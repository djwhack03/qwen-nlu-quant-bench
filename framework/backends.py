import os
import json
import torch
import torch.nn as nn

from framework.config import DEVICE, RANDOM_SEED, MODELS_DIR


# ==========================
# HELPER
# ==========================
def _model_dtype(model) -> torch.dtype:
    for p in model.parameters():
        if p.is_floating_point():
            return p.dtype
    return torch.float16


# ==========================
# OAQ LINEAR LAYER
# ==========================
class OAQLinear(nn.Module):
    def __init__(self, quantized_weight, scales, outlier_weight,
                 bias, in_features, out_features, group_size, bits):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.group_size   = group_size
        self.bits         = bits
        self.register_buffer("quantized_weight", quantized_weight)
        self.register_buffer("scales",           scales)
        self.register_buffer("outlier_weight",   outlier_weight)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None
        self._dequantized_cache = None

    def dequantize(self) -> torch.Tensor:
        if self._dequantized_cache is not None:
            return self._dequantized_cache
        w_q        = self.quantized_weight.float()
        s          = self.scales.float()
        s_expanded = s.repeat_interleave(self.group_size, dim=1)
        s_expanded = s_expanded[:, :self.in_features]
        w_deq      = (w_q * s_expanded +
                      self.outlier_weight.float()).to(torch.float16)
        self._dequantized_cache = w_deq
        return w_deq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w   = self.dequantize()
        out = nn.functional.linear(x, w, None)
        if self.bias is not None:
            out = out + self.bias
        return out


def _oaq_replace_layers(model, bits, group_size, outlier_ratio):
    import torch.nn.functional as F

    def _replace(parent):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and "lm_head" not in child_name:
                w = child.weight.data.float()
                out_f, in_f  = w.shape
                threshold    = torch.quantile(w.abs().flatten(),
                                              1.0 - outlier_ratio)
                outlier_mask = w.abs() >= threshold
                outlier_w    = (w * outlier_mask.float()).half()
                remaining    = w * (~outlier_mask).float()
                pad = (group_size - in_f % group_size) % group_size
                if pad > 0:
                    remaining = F.pad(remaining, (0, pad))
                _, in_pad = remaining.shape
                n_groups  = in_pad // group_size
                r_grouped = remaining.reshape(out_f, n_groups, group_size)
                max_val   = r_grouped.abs().amax(
                    dim=2, keepdim=True).clamp(min=1e-8)
                qmax      = 2 ** (bits - 1) - 1
                scales    = (max_val / qmax).squeeze(2)
                scale_exp = scales.unsqueeze(2).expand_as(r_grouped)
                q_grouped = torch.round(
                    r_grouped / scale_exp).clamp(-qmax - 1, qmax)
                q_flat = q_grouped.reshape(
                    out_f, in_pad)[:, :in_f].to(torch.int8)
                b = (child.bias.data.half().cpu()
                     if child.bias is not None else None)
                setattr(parent, child_name, OAQLinear(
                    quantized_weight = q_flat.cpu(),
                    scales           = scales.half().cpu(),
                    outlier_weight   = outlier_w.cpu(),
                    bias             = b,
                    in_features      = child.in_features,
                    out_features     = child.out_features,
                    group_size       = group_size,
                    bits             = bits,
                ))
            else:
                _replace(child)
    _replace(model)


# ==========================
# HF BACKEND
# Handles: FP16, bnb-4bit, bnb-8bit, AWQ, GPTQ, HQQ, TorchAO, Quanto
# ==========================
class HFBackend:
    def __init__(self, path: str, bnb_bits: int = 0,
                 hqq_bits: int = 0, hqq_group: int = 64,
                 torchao_dtype: str = "", quanto_dtype: str = ""):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.tokenizer = AutoTokenizer.from_pretrained(
            path, trust_remote_code=True)
        self.gen_dtype = torch.float16

        # ── HQQ ──────────────────────────────────────────────────────────
        if hqq_bits > 0:
            from hqq.models.hf.base import AutoHQQHFModel
            from hqq.core.quantize import BaseQuantizeConfig
            print(f"    Loading with HQQ {hqq_bits}-bit "
                  f"(group_size={hqq_group})")
            self.model = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=torch.float16,
                device_map="cuda:0", trust_remote_code=True)
            quant_config = BaseQuantizeConfig(
                nbits=hqq_bits, group_size=hqq_group)
            AutoHQQHFModel.quantize_model(
                self.model, quant_config=quant_config,
                compute_dtype=torch.float16, device=DEVICE)
            self.model.eval()
            self.gen_dtype = _model_dtype(self.model)
            return

        # ── TorchAO ───────────────────────────────────────────────────────
        if torchao_dtype:
            from torchao.quantization import quantize_
            print(f"    Loading with TorchAO {torchao_dtype}")
            self.model = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=torch.bfloat16,
                device_map="auto", trust_remote_code=True)
            if torchao_dtype == "int4_weight_only":
                from torchao.quantization import Int4WeightOnlyConfig
                quantize_(self.model, Int4WeightOnlyConfig(group_size=64))
            elif torchao_dtype == "int8_weight_only":
                from torchao.quantization import Int8WeightOnlyConfig
                quantize_(self.model, Int8WeightOnlyConfig())
            else:
                raise ValueError(f"Unknown torchao_dtype: {torchao_dtype}")
            self.model.eval()
            self.gen_dtype = torch.bfloat16
            return

        # ── Quanto ────────────────────────────────────────────────────────
        if quanto_dtype:
            from transformers import QuantoConfig
            print(f"    Loading with Quanto {quanto_dtype}")
            self.model = AutoModelForCausalLM.from_pretrained(
                path, dtype=torch.float16, device_map="auto",
                quantization_config=QuantoConfig(weights=quanto_dtype),
                trust_remote_code=True)
            self.model.eval()
            return

        # ── Auto-detect AWQ / GPTQ ────────────────────────────────────────
        _config_file = os.path.join(path, "config.json")
        _quant_bits  = bnb_bits

        if _quant_bits == 0 and os.path.exists(_config_file):
            with open(_config_file, encoding="utf-8") as _f:
                _cfg = _f.read()
            _cfg  = json.loads(_cfg)
            _quant = _cfg.get("quantization_config", {}) or {}
            _method = (_quant.get("quant_method", "") or
                       _quant.get("quantization_algo", ""))
            if _method in ("awq", "gptq"):
                print(f"    Detected {_method.upper()} quantization")
                self.model = AutoModelForCausalLM.from_pretrained(
                    path, dtype=torch.float16,
                    device_map="auto", trust_remote_code=True)
                self.model.eval()
                return
            elif (_quant.get("quant_type") == "nf4" or
                  _quant.get("load_in_4bit", False)):
                _quant_bits = 4
            elif _quant.get("load_in_8bit", False):
                _quant_bits = 8

        # ── bitsandbytes ──────────────────────────────────────────────────
        if _quant_bits in (4, 8) and DEVICE != "cuda":
            raise EnvironmentError(
                f"bnb-{_quant_bits}bit requires CUDA.")

        if _quant_bits == 4:
            self.model = AutoModelForCausalLM.from_pretrained(
                path, load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                device_map="auto", trust_remote_code=True)
        elif _quant_bits == 8:
            from transformers import BitsAndBytesConfig
            self.model = AutoModelForCausalLM.from_pretrained(
                path,
                quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                device_map="auto", trust_remote_code=True)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                path,
                dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                device_map="auto" if DEVICE == "cuda" else None,
                trust_remote_code=True)
            if DEVICE == "cpu":
                self.model = self.model.to(torch.float32)

        self.model.eval()

    def _run_generate(self, inputs, **kwargs):
        if DEVICE == "cuda":
            with torch.no_grad(), torch.autocast(
                    "cuda", dtype=self.gen_dtype):
                return self.model.generate(**inputs, **kwargs)
        with torch.no_grad():
            return self.model.generate(**inputs, **kwargs)

    def generate(self, messages: list, max_new_tokens: int = 400,
                 do_sample: bool = False,
                 repetition_penalty: float = 1.0) -> str:
        prompt  = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs  = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        outputs = self._run_generate(
            inputs, max_new_tokens=max_new_tokens,
            do_sample=do_sample, repetition_penalty=repetition_penalty)
        return self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True).strip()

    def unload(self):
        del self.model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


# ==========================
# GGUF BACKEND
# ==========================
class GGUFBackend:
    def __init__(self, path: str, pattern: str = ""):
        from llama_cpp import Llama
        if os.path.isdir(path):
            gguf_files = sorted(
                f for f in os.listdir(path) if f.endswith(".gguf"))
            if not gguf_files:
                raise FileNotFoundError(
                    f"No .gguf file found in: {path}")
            matched = ([f for f in gguf_files
                        if pattern.lower() in f.lower()]
                       if pattern else gguf_files)
            chosen = matched[0] if matched else gguf_files[0]
            path   = os.path.join(path, chosen)
            print(f"    Auto-selected GGUF: {os.path.basename(path)}")
        self.llm = Llama(
            model_path=path, n_ctx=4096,
            n_gpu_layers=-1 if DEVICE == "cuda" else 0,
            seed=RANDOM_SEED, verbose=False)

    def generate(self, messages: list, max_new_tokens: int = 400,
                 do_sample: bool = False,
                 repetition_penalty: float = 1.0) -> str:
        response = self.llm.create_chat_completion(
            messages=messages, max_tokens=max_new_tokens,
            temperature=0.0 if not do_sample else 0.7,
            repeat_penalty=repetition_penalty)
        return response["choices"][0]["message"]["content"].strip()

    def unload(self):
        del self.llm
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


# ==========================
# OAQ BACKEND
# ==========================
class OAQBackend:
    def __init__(self, path: str):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        oaq_cfg_path = os.path.join(path, "oaq_config.json")
        if not os.path.exists(oaq_cfg_path):
            raise FileNotFoundError(
                f"No oaq_config.json in {path}.")

        with open(oaq_cfg_path) as f:
            oaq_cfg = json.load(f)

        print(f"    OAQ: {oaq_cfg['bits']}-bit, "
              f"group={oaq_cfg['group_size']}, "
              f"outlier={oaq_cfg['outlier_ratio']:.1%}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            path, trust_remote_code=True)

        base_model = AutoModelForCausalLM.from_pretrained(
            oaq_cfg["base_model"], torch_dtype=torch.float16,
            trust_remote_code=True)
        _oaq_replace_layers(
            base_model,
            bits=oaq_cfg["bits"],
            group_size=oaq_cfg["group_size"],
            outlier_ratio=oaq_cfg["outlier_ratio"])

        state_dict_path = os.path.join(path, "oaq_model.pt")
        if not os.path.exists(state_dict_path):
            raise FileNotFoundError(
                f"oaq_model.pt not found in {path}")

        state_dict = torch.load(state_dict_path, map_location="cpu")
        base_model.load_state_dict(state_dict)

        if DEVICE == "cuda":
            base_model = base_model.to(DEVICE)
        base_model.eval()
        self.model = base_model

        print("    Pre-warming OAQ dequantization cache...")
        dummy = torch.zeros(1, 1, dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            self.model(dummy)
        print("    Cache ready.")

    def generate(self, messages: list, max_new_tokens: int = 400,
                 do_sample: bool = False,
                 repetition_penalty: float = 1.0) -> str:
        prompt  = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs  = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty)
        return self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True).strip()

    def unload(self):
        del self.model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


# ==========================
# CLASSICAL NER BACKEND
# ==========================
class ClassicalNERBackend:
    def __init__(self, kind: str, model_name: str = ""):
        self.kind = kind
        if kind == "spacy":
            import spacy
            import sys, subprocess
            name = model_name or "ru_core_news_sm"
            try:
                self.nlp = spacy.load(name)
            except OSError:
                print(f"    Downloading spaCy model {name}...")
                subprocess.check_call(
                    [sys.executable, "-m", "spacy", "download", name])
                self.nlp = spacy.load(name)
            print(f"    Loaded spaCy: {name}")
        elif kind == "hf_ner":
            from transformers import pipeline
            name = model_name or "surdan/LaBSE_ner_nerel"
            print(f"    Loading HF NER pipeline: {name}")
            self.pipe = pipeline(
                "token-classification", model=name,
                aggregation_strategy="simple",
                device=0 if DEVICE == "cuda" else -1)
        else:
            raise ValueError(f"Unknown classical NER kind: {kind}")

    def extract_persons(self, sentence: str) -> list:
        if self.kind == "spacy":
            doc = self.nlp(sentence)
            return [ent.text.strip()
                    for ent in doc.ents if ent.label_ == "PER"]
        elif self.kind == "hf_ner":
            entities = self.pipe(sentence)
            return [e["word"].strip() for e in entities
                    if e["entity_group"] in ("PER", "PERSON")]

    def generate(self, messages, **kwargs):
        raise NotImplementedError(
            "Use extract_persons() for classical backends")

    def unload(self):
        if hasattr(self, "nlp"):
            del self.nlp
        if hasattr(self, "pipe"):
            del self.pipe
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


# ==========================
# CLASSICAL SENTIMENT BACKEND
# ==========================
class ClassicalSABackend:
    def __init__(self, kind: str, model_name: str = ""):
        self.kind = kind
        if kind == "textblob":
            from textblob import TextBlob
            self._cls = TextBlob
        elif kind == "vader":
            from nltk.sentiment import SentimentIntensityAnalyzer
            import nltk
            nltk.download("vader_lexicon", quiet=True)
            self.sia = SentimentIntensityAnalyzer()
        elif kind == "hf_sentiment":
            from transformers import pipeline
            self.pipe = pipeline(
                "text-classification", model=model_name,
                device=0 if DEVICE == "cuda" else -1)
        else:
            raise ValueError(f"Unknown classical SA kind: {kind}")

    def predict(self, text: str) -> str:
        if self.kind == "textblob":
            pol = self._cls(text).sentiment.polarity
            if pol > 0.1:   return "positive"
            if pol < -0.1:  return "negative"
            return "neutral"
        elif self.kind == "vader":
            score = self.sia.polarity_scores(text)["compound"]
            if score >= 0.05:  return "positive"
            if score <= -0.05: return "negative"
            return "neutral"
        elif self.kind == "hf_sentiment":
            result = self.pipe(text[:512])[0]["label"].lower()
            if "pos" in result: return "positive"
            if "neg" in result: return "negative"
            return "neutral"

    def unload(self):
        if hasattr(self, "pipe"):
            del self.pipe
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


# ==========================
# BACKEND FACTORY
# ==========================
def load_backend(model_cfg: dict):
    mtype = model_cfg["type"]
    path  = model_cfg.get("path", "")

    if mtype == "oaq":
        return OAQBackend(path)
    elif mtype == "hf":
        return HFBackend(path, bnb_bits=model_cfg.get("bnb_bits", 0))
    elif mtype == "hqq":
        return HFBackend(path,
                         hqq_bits=model_cfg.get("hqq_bits", 4),
                         hqq_group=model_cfg.get("hqq_group", 64))
    elif mtype == "torchao":
        return HFBackend(path,
                         torchao_dtype=model_cfg.get("torchao_dtype", ""))
    elif mtype == "quanto":
        return HFBackend(path,
                         quanto_dtype=model_cfg.get("quanto_dtype", "int8"))
    elif mtype == "gguf":
        return GGUFBackend(path,
                           pattern=model_cfg.get("gguf_pattern", ""))
    elif mtype == "spacy":
        return ClassicalNERBackend("spacy",
                                   model_cfg.get("model_name", ""))
    elif mtype == "hf_ner":
        return ClassicalNERBackend("hf_ner",
                                   model_cfg.get("model_name", ""))
    elif mtype in ("textblob", "vader", "hf_sentiment"):
        return ClassicalSABackend(mtype,
                                  model_cfg.get("model_name", ""))
    else:
        raise ValueError(f"Unknown model type: {mtype}")
