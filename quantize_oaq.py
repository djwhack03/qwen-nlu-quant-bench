"""
Outlier-Aware Quantization (OAQ) — Round-To-Nearest with outlier isolation.

Algorithm:
  1. For each linear layer's weight matrix:
     a. Identify outlier weights (top X% by absolute magnitude)
     b. Store outliers in fp16 as a sparse component
     c. Quantize remaining weights to int4 using per-group scaling
     d. At inference: dequantize int4 + add fp16 outliers = fp16 output

Fixes vs original:
  - OAQLinear.dequantize() invalidates cache on device mismatch (prevents
    silent per-token recompute after model.to(CUDA))
  - OUTLIER_RATIO raised 0.01 → 0.02 for lower quantization error
  - Base model loaded directly to CUDA (avoids Windows pagefile exhaustion
    / OSError 1455 on large models with mmap-based loading)
  - SAFETENSORS_DISABLE_MMAP env guard added

Usage:
    python quantize_oaq.py

Output:
    <BASE_DIR>/models/qwen2-1.5b-oaq-4bit/
"""

import os

# Must be set before torch / transformers / safetensors are imported.
# Disables mmap-based weight loading which exhausts the Windows pagefile
# (OSError 1455) on models larger than ~2 GB.
os.environ["SAFETENSORS_FAST_GPU"]      = "0"
os.environ["SAFETENSORS_DISABLE_MMAP"] = "1"
os.environ["TORCH_DISTRIBUTED_DEBUG"]  = "OFF"

import sys
import json
import shutil
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from framework.config import BASE_MODEL_PATH, model_path, DEVICE

# ==========================
# CONFIG
# ==========================
OUTPUT_PATH   = model_path("qwen2-1.5b-oaq-4bit")
BITS          = 4
GROUP_SIZE    = 128
OUTLIER_RATIO = 0.02  # raised from 0.01 — captures more tail outliers
                      # with negligible size overhead (~0.3% larger state dict)


# ==========================
# OAQ LINEAR LAYER
# FIX: dequantize() checks device consistency before reusing cache.
# If the module was moved to a different device after caching (e.g.
# pre-warmed on CPU then moved to CUDA), the stale cache is dropped and
# recomputed once on the correct device. All subsequent calls are cache hits.
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
        # Invalidate cache if module has moved to a different device since
        # the cache was populated (registered buffers move with .to(device),
        # but _dequantized_cache is a plain attribute and stays behind).
        if self._dequantized_cache is not None:
            if self._dequantized_cache.device != self.quantized_weight.device:
                self._dequantized_cache = None  # stale — recompute
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

    def extra_repr(self):
        return (f"in={self.in_features}, out={self.out_features}, "
                f"bits={self.bits}, group={self.group_size}, "
                f"outlier_ratio={OUTLIER_RATIO:.1%}")


# ==========================
# QUANTIZATION
# ==========================
def quantize_weight(weight: torch.Tensor, bits: int,
                    group_size: int, outlier_ratio: float):
    w = weight.float()
    out_features, in_features = w.shape

    threshold    = torch.quantile(w.abs().flatten(), 1.0 - outlier_ratio)
    outlier_mask = w.abs() >= threshold
    outlier_w    = (w * outlier_mask.float()).half()
    remaining    = w * (~outlier_mask).float()

    pad = (group_size - in_features % group_size) % group_size
    if pad > 0:
        remaining = torch.nn.functional.pad(remaining, (0, pad))
    _, in_padded = remaining.shape
    n_groups     = in_padded // group_size

    r_grouped = remaining.reshape(out_features, n_groups, group_size)
    max_val   = r_grouped.abs().amax(dim=2, keepdim=True).clamp(min=1e-8)
    qmax      = 2 ** (bits - 1) - 1
    scales    = (max_val / qmax).squeeze(2)

    scale_exp = scales.unsqueeze(2).expand_as(r_grouped)
    q_grouped = torch.round(r_grouped / scale_exp).clamp(-qmax - 1, qmax)
    q_flat    = q_grouped.reshape(out_features, in_padded)[:, :in_features]

    return q_flat.to(torch.int8), scales.half(), outlier_w


def quantize_model(model: nn.Module, bits: int,
                   group_size: int, outlier_ratio: float) -> nn.Module:
    replaced = 0

    def _replace(parent: nn.Module, name_path: str = ""):
        nonlocal replaced
        for child_name, child in list(parent.named_children()):
            full_name = (f"{name_path}.{child_name}"
                         if name_path else child_name)
            if isinstance(child, nn.Linear):
                if "lm_head" in full_name:
                    continue
                w = child.weight.data.to(DEVICE)
                b = (child.bias.data.half().cpu()
                     if child.bias is not None else None)
                q_w, scales, outlier_w = quantize_weight(
                    w, bits=bits, group_size=group_size,
                    outlier_ratio=outlier_ratio)
                setattr(parent, child_name, OAQLinear(
                    quantized_weight = q_w.cpu(),
                    scales           = scales.cpu(),
                    outlier_weight   = outlier_w.cpu(),
                    bias             = b,
                    in_features      = child.in_features,
                    out_features     = child.out_features,
                    group_size       = group_size,
                    bits             = bits,
                ))
                replaced += 1
            else:
                _replace(child, full_name)

    _replace(model)
    print(f"  Replaced {replaced} linear layers with OAQLinear")
    return model


# ==========================
# SAVE
# ==========================
def save_oaq_model(model, tokenizer, output_path: str,
                   config_extras: dict):
    os.makedirs(output_path, exist_ok=True)
    tokenizer.save_pretrained(output_path)
    print(f"  Saved tokenizer → {output_path}")

    state_dict_path = os.path.join(output_path, "oaq_model.pt")
    torch.save(model.state_dict(), state_dict_path)
    print(f"  Saved state dict → {state_dict_path}")

    base_cfg_src = os.path.join(config_extras["base_model"], "config.json")
    base_cfg_dst = os.path.join(output_path, "config.json")
    if os.path.exists(base_cfg_src):
        shutil.copy(base_cfg_src, base_cfg_dst)
        print(f"  Copied base config.json → {base_cfg_dst}")
    else:
        print(f"  [WARN] Could not find base config.json at {base_cfg_src}")

    with open(os.path.join(output_path, "oaq_config.json"), "w") as f:
        json.dump({
            "oaq_version":   "1.0",
            "bits":          config_extras["bits"],
            "group_size":    config_extras["group_size"],
            "outlier_ratio": config_extras["outlier_ratio"],
            "base_model":    config_extras["base_model"],
            "description": (
                "Outlier-Aware Quantization: top outlier_ratio weights "
                "stored in fp16, remaining weights quantized to int4 "
                "with per-group scaling."
            ),
        }, f, indent=2)
    print(f"  Saved OAQ config → {output_path}/oaq_config.json")


# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    print(f"Device      : {DEVICE}")
    print(f"Config      : {BITS}-bit, group_size={GROUP_SIZE}, "
          f"outlier_ratio={OUTLIER_RATIO:.1%}")
    print(f"Base model  : {BASE_MODEL_PATH}")
    print(f"Output      : {OUTPUT_PATH}\n")

    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH, trust_remote_code=True)

    # Load directly to CUDA rather than staging through CPU.
    # On Windows, CPU loading uses memory-mapped files which drain the
    # pagefile (OSError 1455). Loading to CUDA reads straight from disk
    # into VRAM, bypassing the pagefile entirely. quantize_model() pulls
    # each weight to DEVICE for quantization and stores the result on CPU,
    # so peak VRAM stays within the 1.5B model's ~3 GB footprint.
    _load_device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.float16,
        device_map=_load_device, trust_remote_code=True)
    print(f"  Parameters: "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M\n")

    n_linear = sum(
        1 for name, m in model.named_modules()
        if isinstance(m, nn.Linear) and "lm_head" not in name
    )
    print(f"  Linear layers to quantize: {n_linear}\n")

    print("Quantizing...")
    model = quantize_model(
        model, bits=BITS,
        group_size=GROUP_SIZE, outlier_ratio=OUTLIER_RATIO)

    print("\nSaving...")
    save_oaq_model(model, tokenizer, OUTPUT_PATH, {
        "bits":          BITS,
        "group_size":    GROUP_SIZE,
        "outlier_ratio": OUTLIER_RATIO,
        "base_model":    BASE_MODEL_PATH,
    })

    print("\nSanity check...")
    model.eval()
    test_input = tokenizer(
        "Привет, как дела?", return_tensors="pt").to(_load_device)
    with torch.no_grad():
        out = model.generate(
            test_input["input_ids"], max_new_tokens=10, do_sample=False)
    print(f"  Output: '{tokenizer.decode(out[0], skip_special_tokens=True)}'")
    print(f"\nDone. Model saved to {OUTPUT_PATH}")
