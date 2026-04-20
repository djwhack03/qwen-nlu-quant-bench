import os
import json
import re
import time

os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
sys.modules["tensorflow"] = None
sys.modules["tensorflow_core"] = None

import torch
from tqdm import tqdm

# ==========================
# CONFIG
# ==========================
OUTPUT_DIR      = r"E:/quant/outputs"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL_PATH = r"E:/quant/models/qwen2-1.5b-instruct"

print("Device:", DEVICE)

# ==========================
# SENTIMENT CONFIG
# ==========================
SENTIMENT_SAMPLES = 500

SENTIMENT_DATASET = {
    "type":   "file",
    "format": "rusentiment",
    "path":   r"E:/quant/data/rusentiment.csv",
}

SENTIMENT_MODELS = [
    # ── Classical baselines ───────────────────────────────────────────────
    {"label": "HF-sentiment-rubert", "type": "hf_sentiment",
     "model_name": "blanchefort/rubert-base-cased-sentiment"},

    # ── LLM variants (matching photo) ────────────────────────────────────
    {"label": "Qwen2-1.5B-FP16",
     "type": "hf", "path": BASE_MODEL_PATH},

    {"label": "Qwen2-1.5B-GGUF-Q4",
     "type": "gguf", "path": r"E:/quant/models/qwen2-1.5b-gguf",
     "gguf_pattern": "Q4_K_M"},

    {"label": "Qwen2-1.5B-BnB-4bit",
     "type": "bnb", "path": BASE_MODEL_PATH, "bnb_bits": 4},

    {"label": "Qwen2-1.5B-Quanto-4bit",
     "type": "quanto", "path": BASE_MODEL_PATH, "quanto_weight_bits": 4},

    {"label": "Qwen2-1.5B-HQQ-4bit",
     "type": "hqq", "path": BASE_MODEL_PATH, "hqq_bits": 4, "hqq_group": 64},
]

SENTIMENT_PROMPT = """Classify the sentiment of Russian social media text.
Reply with EXACTLY one word: positive, negative, or neutral.

positive = author feels happy, excited, grateful, in love, amused, proud.
negative = author feels sad, angry, frustrated, disgusted, lonely, disappointed, sarcastic.
neutral  = factual, informational, descriptive — no personal emotion expressed.

Short emotional text, laughter (ахаха, )))))), emoji, or exclamations are NEVER neutral.
When the author clearly feels something personal -> positive or negative, never neutral.

Text: ТЫ МОЕ МАЛЕНЬКОЕ СЧАСТЬЕ !!!!!! Я ТЕБЯ ЛЮБЛЮ ОЧЕНЬ СИЛЬНО
Label: positive

Text: иихуу))) ещё на год постарел)))
Label: positive

Text: 4 года без... Помним
Label: negative

Text: соскучилась по тебе, дружище
Label: negative

Text: Торт Бильярдный стол
Label: neutral

Text: У задержанных за беспорядки в Москве изъято 16 травматических пистолетов
Label: neutral

Now classify this text. Reply with one word only."""

# ==========================
# DATASET LOADERS
# ==========================
def load_rusentiment(path: str):
    import csv
    keep = {"positive", "negative", "neutral"}
    samples = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("label", row.get("sentiment", "")).strip().lower()
            if label in keep:
                samples.append((row["text"].strip(), label))
    return samples


def load_sentiment_hf(dataset_name, split="validation",
                      text_col="sentence", label_col="label"):
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split=split)
    label_map = {0: "negative", 1: "positive", 2: "neutral"}
    samples = []
    for row in ds:
        text  = row[text_col]
        label = row[label_col]
        if isinstance(label, int):
            label = label_map.get(label, str(label))
        samples.append((text, label))
    return samples


def normalize_label(label) -> str:
    if isinstance(label, int):
        return "positive" if label == 1 else "negative"
    label = str(label).lower()
    if "pos" in label: return "positive"
    if "neg" in label: return "negative"
    return "neutral"


def _detect_model_dtype(model) -> torch.dtype:
    for p in model.parameters():
        if p.is_floating_point():
            return p.dtype
    return torch.float16

# ==========================
# HF BACKEND
# ==========================
class HFBackend:
    def __init__(self, path: str, bnb_bits: int = 0,
                 hqq_bits: int = 0, hqq_group: int = 64,
                 torchao_dtype: str = ""):
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import json as _json

        self.tokenizer   = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.model_dtype = torch.float16

        if hqq_bits > 0:
            from hqq.models.hf.base import AutoHQQHFModel
            from hqq.core.quantize import BaseQuantizeConfig
            print(f"    Loading with HQQ {hqq_bits}-bit (group_size={hqq_group})")
            self.model = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=torch.float16,
                device_map="cuda:0", trust_remote_code=True)
            quant_config = BaseQuantizeConfig(nbits=hqq_bits, group_size=hqq_group)
            AutoHQQHFModel.quantize_model(
                self.model, quant_config=quant_config,
                compute_dtype=torch.float16, device=DEVICE)
            self.model.eval()
            self.model_dtype = _detect_model_dtype(self.model)
            return

        if torchao_dtype:
            print(f"    Loading with TorchAO {torchao_dtype}")
            self.model = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=torch.bfloat16,
                device_map=None, trust_remote_code=True)
            if torchao_dtype == "int8_weight_only":
                try:
                    from torchao.quantization import quantize_, Int8WeightOnlyConfig
                    quantize_(self.model, Int8WeightOnlyConfig())
                except (ImportError, TypeError):
                    from torchao.quantization import quantize_
                    from torchao.quantization.quant_api import int8_weight_only
                    quantize_(self.model, int8_weight_only())
            if DEVICE == "cuda":
                self.model = self.model.to("cuda")
            self.model.eval()
            self.model_dtype = torch.bfloat16
            return

        _config_file = os.path.join(path, "config.json")
        _quant_bits  = bnb_bits

        if _quant_bits == 0 and os.path.exists(_config_file):
            with open(_config_file, encoding="utf-8") as _f:
                _cfg = _json.load(_f)
            _quant = _cfg.get("quantization_config", {}) or {}
            _method = (_quant.get("quant_method", "") or _quant.get("quantization_algo", ""))
            if _method in ("awq", "gptq"):
                self.model = AutoModelForCausalLM.from_pretrained(
                    path, torch_dtype=torch.float16,
                    device_map="auto", trust_remote_code=True)
                self.model.eval()
                self.model_dtype = _detect_model_dtype(self.model)
                return
            elif _quant.get("quant_type") == "nf4" or _quant.get("load_in_4bit"):
                _quant_bits = 4
            elif _quant.get("load_in_8bit"):
                _quant_bits = 8

        if _quant_bits == 4:
            print("    Loading with BitsAndBytes NF4 4-bit")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
            self.model = AutoModelForCausalLM.from_pretrained(
                path, quantization_config=bnb_config,
                device_map="auto", trust_remote_code=True)
        elif _quant_bits == 8:
            print("    Loading with BitsAndBytes 8-bit")
            self.model = AutoModelForCausalLM.from_pretrained(
                path, quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                device_map="auto", trust_remote_code=True)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                device_map="auto" if DEVICE == "cuda" else None,
                trust_remote_code=True)
            if DEVICE == "cpu":
                self.model = self.model.to(torch.float32)

        self.model.eval()
        self.model_dtype = _detect_model_dtype(self.model)

    def _input_device(self):
        for p in self.model.parameters():
            if p.device.type != "meta":
                return p.device
        return torch.device(DEVICE)

    def generate(self, messages, max_new_tokens=8, do_sample=False, repetition_penalty=1.0):
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs_raw = self.tokenizer(prompt, return_tensors="pt")
        target_device = self._input_device()
        inputs = {k: v.to(target_device) for k, v in inputs_raw.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=do_sample, repetition_penalty=repetition_penalty)
        return self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True).strip()

    def unload(self):
        del self.model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


class QuantoBackend:
    def __init__(self, path: str, weight_bits: int = 4):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from optimum.quanto import quantize, freeze, qint4, qint8
        print(f"    Loading with Quanto {weight_bits}-bit weight-only")
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map=None, trust_remote_code=True)
        w_dtype = qint4 if weight_bits == 4 else qint8
        quantize(self.model, weights=w_dtype, activations=None)
        freeze(self.model)
        if DEVICE == "cuda":
            self.model = self.model.to("cuda")
        self.model.eval()
        self.model_dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    def generate(self, messages, max_new_tokens=8, do_sample=False, repetition_penalty=1.0):
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs_raw = self.tokenizer(prompt, return_tensors="pt")
        target_device = next(self.model.parameters()).device
        inputs = {k: v.to(target_device) for k, v in inputs_raw.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=do_sample, repetition_penalty=repetition_penalty)
        return self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True).strip()

    def unload(self):
        del self.model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


class GGUFBackend:
    def __init__(self, path: str, pattern: str = ""):
        from llama_cpp import Llama
        if os.path.isdir(path):
            gguf_files = sorted(f for f in os.listdir(path) if f.endswith(".gguf"))
            if not gguf_files:
                raise FileNotFoundError(f"No .gguf file found in: {path}")
            matched = [f for f in gguf_files if pattern.lower() in f.lower()] if pattern else gguf_files
            chosen  = matched[0] if matched else gguf_files[0]
            path    = os.path.join(path, chosen)
            print(f"  Auto-selected GGUF: {os.path.basename(path)}")
        self.llm = Llama(model_path=path, n_ctx=2048,
                         n_gpu_layers=-1 if DEVICE == "cuda" else 0, verbose=False)

    def generate(self, messages, max_new_tokens=8, do_sample=False, repetition_penalty=1.0):
        response = self.llm.create_chat_completion(
            messages=messages, max_tokens=max_new_tokens,
            temperature=0.0 if not do_sample else 0.7,
            repeat_penalty=repetition_penalty)
        return response["choices"][0]["message"]["content"].strip()

    def unload(self):
        del self.llm
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


class SentimentClassicalBackend:
    def __init__(self, kind: str, model_name: str = ""):
        self.kind = kind
        if kind == "hf_sentiment":
            from transformers import pipeline
            self.pipe = pipeline("text-classification", model=model_name,
                                 device=0 if DEVICE == "cuda" else -1)

    def predict(self, text: str) -> str:
        if self.kind == "hf_sentiment":
            result = self.pipe(text[:512])[0]["label"].lower()
            if "pos" in result: return "positive"
            if "neg" in result: return "negative"
            return "neutral"

    def unload(self):
        if hasattr(self, "pipe"):
            del self.pipe
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


def load_backend(model_cfg: dict):
    mtype = model_cfg["type"]
    path  = model_cfg.get("path", "")
    if mtype == "hf":
        return HFBackend(path)
    elif mtype == "bnb":
        return HFBackend(path, bnb_bits=model_cfg.get("bnb_bits", 4))
    elif mtype == "hqq":
        return HFBackend(path, hqq_bits=model_cfg.get("hqq_bits", 4),
                         hqq_group=model_cfg.get("hqq_group", 64))
    elif mtype == "torchao":
        return HFBackend(path, torchao_dtype=model_cfg.get("torchao_dtype", ""))
    elif mtype == "quanto":
        return QuantoBackend(path, weight_bits=model_cfg.get("quanto_weight_bits", 4))
    elif mtype == "gguf":
        return GGUFBackend(path, pattern=model_cfg.get("gguf_pattern", ""))
    elif mtype == "hf_sentiment":
        return SentimentClassicalBackend(mtype, model_cfg.get("model_name", ""))
    else:
        raise ValueError(f"Unknown model type: {mtype}")


_parse_failures: list = []

def predict_sentiment_llm(backend, text: str) -> str:
    messages = [
        {"role": "system", "content": SENTIMENT_PROMPT},
        {"role": "user",   "content": text},
    ]
    raw = backend.generate(messages, max_new_tokens=8, do_sample=False).lower().strip()
    for prefix in ("label:", "ответ:", "sentiment:", "answer:", "output:"):
        if prefix in raw:
            raw = raw.split(prefix, 1)[-1].strip()
    if "pos" in raw:   return "positive"
    if "neg" in raw:   return "negative"
    if "neu" in raw:   return "neutral"
    if "позит" in raw: return "positive"
    if "негат" in raw: return "negative"
    if "нейтр" in raw: return "neutral"
    _parse_failures.append({"text": text[:80], "raw": raw})
    return "neutral"


# ==========================
# MAIN
# ==========================
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n" + "=" * 70)
print("  SENTIMENT ANALYSIS BENCHMARK")
print("=" * 70)

cfg = SENTIMENT_DATASET
if cfg["type"] == "hf":
    samples = load_sentiment_hf(cfg["name"], cfg.get("split", "validation"),
                                cfg.get("text_col", "sentence"), cfg.get("label_col", "label"))
elif cfg["type"] == "file" and cfg["format"] == "rusentiment":
    samples = load_rusentiment(cfg["path"])
else:
    raise ValueError(f"Unknown dataset config: {cfg}")

samples = samples[:SENTIMENT_SAMPLES]
from collections import Counter
dist = Counter(normalize_label(s[1]) for s in samples)
print(f"\n  Dataset : {cfg.get('name', cfg.get('path'))}")
print(f"  Samples : {len(samples)}")
print(f"  Labels  : {dict(dist)}\n")

summary = []

for model_cfg in SENTIMENT_MODELS:
    label        = model_cfg["label"]
    mtype        = model_cfg["type"]
    is_classical = mtype in ("hf_sentiment",)

    print(f"\n{'='*60}")
    print(f"  Model: {label}")
    print(f"{'='*60}")

    try:
        backend = load_backend(model_cfg)
    except Exception as e:
        print(f"  [SKIPPED] {e}")
        summary.append({"model": label, "accuracy": None, "f1_macro": None,
                        "skipped": True, "skip_reason": str(e)})
        continue

    preds, golds, inference_times = [], [], []

    for text, gold in tqdm(samples, desc=label):
        gold_norm = normalize_label(gold)
        t0   = time.perf_counter()
        pred = backend.predict(text) if is_classical else predict_sentiment_llm(backend, text)
        inference_times.append((time.perf_counter() - t0) * 1000)
        preds.append(pred)
        golds.append(gold_norm)

    correct  = sum(p == g for p, g in zip(preds, golds))
    accuracy = correct / len(golds)
    classes  = sorted(set(golds))
    f1s, per_class = [], {}
    for cls in classes:
        tp = sum(p == cls and g == cls for p, g in zip(preds, golds))
        fp = sum(p == cls and g != cls for p, g in zip(preds, golds))
        fn = sum(p != cls and g == cls for p, g in zip(preds, golds))
        p_cls  = tp / (tp + fp + 1e-8)
        r_cls  = tp / (tp + fn + 1e-8)
        f1_cls = 2 * p_cls * r_cls / (p_cls + r_cls + 1e-8)
        f1s.append(f1_cls)
        per_class[cls] = {"tp": tp, "fp": fp, "fn": fn,
                          "precision": round(p_cls, 4),
                          "recall":    round(r_cls, 4),
                          "f1":        round(f1_cls, 4)}
    f1_macro = sum(f1s) / len(f1s)
    avg_ms   = sum(inference_times) / len(inference_times)
    total_s  = sum(inference_times) / 1000

    print(f"\n  Accuracy   : {accuracy:.4f}")
    print(f"  F1 macro   : {f1_macro:.4f}")
    print(f"  Avg latency: {avg_ms:.1f} ms/sample")
    print(f"  Total time : {total_s:.1f}s")

    if not is_classical and _parse_failures:
        print(f"  [!] Parse failures: {len(_parse_failures)}")
        for pf in _parse_failures[:3]:
            print(f"      raw={pf['raw']!r:20}  text={pf['text'][:40]!r}")
        _parse_failures.clear()

    safe_label = re.sub(r"[^\w\-]", "_", label)
    out_path   = os.path.join(OUTPUT_DIR, f"sentiment_{safe_label}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": label, "accuracy": round(accuracy, 4),
            "f1_macro": round(f1_macro, 4),
            "avg_inference_ms": round(avg_ms, 2),
            "total_inference_s": round(total_s, 2),
            "per_class": per_class,
            "predictions": [
                {"text": s[0], "gold": normalize_label(s[1]), "pred": p}
                for s, p in zip(samples, preds)],
        }, f, ensure_ascii=False, indent=2)
    print(f"  Saved → {out_path}")

    summary.append({"model": label, "accuracy": round(accuracy, 4),
                    "f1_macro": round(f1_macro, 4), "avg_ms": round(avg_ms, 2)})
    backend.unload()

print("\n" + "=" * 70)
print("  FINAL SUMMARY")
print("=" * 70)
print(f"{'Model':<45} {'Acc':>7} {'F1-mac':>8} {'ms/s':>8}")
print("-" * 70)
for s in summary:
    if s.get("skipped"):
        print(f"{s['model']:<45} SKIPPED  {s.get('skip_reason', '')}")
    else:
        print(f"{s['model']:<45} {s['accuracy']:>7.4f} {s['f1_macro']:>8.4f} {s['avg_ms']:>8.1f}")

summ_path = os.path.join(OUTPUT_DIR, "summary_sentiment.json")
with open(summ_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"\nSummary saved → {summ_path}")
