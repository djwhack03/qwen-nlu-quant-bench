import os
import re
import json
import time
import random

os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
sys.modules["tensorflow"] = None
sys.modules["tensorflow_core"] = None

from collections import Counter, defaultdict
from tqdm import tqdm

from framework.config import (
    OUTPUT_DIR, MAX_SA_SAMPLES, STRATIFY_SAMPLES, STRATIFY_SEED,
    SENTIMENT_DATASET, SA_MODELS,
)
from framework.datasets import (
    load_rusentiment, load_sst2, load_sentiment_hf,
)
from framework.backends import load_backend, ClassicalSABackend
from framework.inference import (
    predict_sentiment_llm, predict_sentiment_classical, _parse_failures,
)
from framework.postprocess import normalize_sentiment_label
from framework.evaluate import sa_metrics


# ==========================
# STRATIFIED SAMPLING
# ==========================
def stratified_sample(samples: list, n: int, seed: int = 42) -> list:
    random.seed(seed)
    by_class = defaultdict(list)
    for text, label in samples:
        by_class[normalize_sentiment_label(label)].append((text, label))

    num_classes = len(by_class)
    per_class   = n // num_classes
    result      = []
    for cls, cls_samples in by_class.items():
        chosen = random.sample(cls_samples, min(per_class, len(cls_samples)))
        result.extend(chosen)
        if len(cls_samples) < per_class:
            print(f"  [WARN] Class '{cls}' has only {len(cls_samples)} "
                  f"samples, requested {per_class}. Using all.")

    random.shuffle(result)
    print(f"  Stratified: {per_class} per class × {num_classes} classes "
          f"= {len(result)} total (requested {n})")
    return result


# ==========================
# QUANTIZATION FIDELITY REPORT
# ==========================
def quantization_fidelity_report(
    fp16_preds: list, quant_preds: list, golds: list, label: str
) -> dict:
    diffs = [
        (g, fp16_p, q_p)
        for g, fp16_p, q_p in zip(golds, fp16_preds, quant_preds)
        if fp16_p != q_p
    ]
    neutral_collapse = [
        (g, fp16_p, q_p)
        for g, fp16_p, q_p in zip(golds, fp16_preds, quant_preds)
        if g != "neutral" and fp16_p != "neutral" and q_p == "neutral"
    ]
    spurious_gains = [
        (g, fp16_p, q_p)
        for g, fp16_p, q_p in zip(golds, fp16_preds, quant_preds)
        if fp16_p != g and q_p == g and q_p == "neutral"
    ]

    total         = len(golds)
    diff_rate     = len(diffs) / total
    collapse_rate = len(neutral_collapse) / total

    print(f"\n  [Fidelity vs FP16] {label}")
    print(f"    Diverging samples     : {len(diffs)} / {total}  "
          f"({diff_rate:.1%})")
    print(f"    Neutral-collapse rate : {len(neutral_collapse)} / {total}  "
          f"({collapse_rate:.1%})")
    if spurious_gains:
        print(f"    Spurious gains via neutral collapse: {len(spurious_gains)}")

    return {
        "diverging":              len(diffs),
        "diverging_rate":         round(diff_rate, 4),
        "neutral_collapse":       len(neutral_collapse),
        "neutral_collapse_rate":  round(collapse_rate, 4),
        "spurious_neutral_gains": len(spurious_gains),
    }


# ==========================
# MAIN
# ==========================
def run_sentiment():
    print("=" * 70)
    print("  SENTIMENT ANALYSIS BENCHMARK")
    print("=" * 70)
    print(f"  Primary metric: F1-macro  (accuracy shown for reference only)\n")

    # ── load data ─────────────────────────────────────────────────────────
    cfg = SENTIMENT_DATASET
    if cfg["type"] == "hf":
        all_samples = load_sentiment_hf(
            cfg["name"], cfg.get("split", "validation"),
            cfg.get("text_col", "sentence"),
            cfg.get("label_col", "label"))
    elif cfg["type"] == "file" and cfg["format"] == "rusentiment":
        all_samples = load_rusentiment(cfg["path"])
    elif cfg["type"] == "file" and cfg["format"] == "sst2":
        all_samples = load_sst2(cfg["path"])
    else:
        raise ValueError(f"Unknown dataset config: {cfg}")

    print(f"  Dataset     : {cfg.get('name', cfg.get('path'))}")
    print(f"  Total rows  : {len(all_samples)}")

    # ── stratified or raw sampling ────────────────────────────────────────
    if STRATIFY_SAMPLES:
        samples = stratified_sample(all_samples, MAX_SA_SAMPLES,
                                    seed=STRATIFY_SEED)
    else:
        samples = all_samples[:MAX_SA_SAMPLES]
        print(f"  Samples     : {len(samples)}  "
              f"(raw slice, no stratification)")

    dist = Counter(normalize_sentiment_label(s[1]) for s in samples)
    print(f"  Label dist  : {dict(dist)}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary          = []
    fp16_preds_store = None   # kept for fidelity comparison

    for model_cfg in SA_MODELS:
        label        = model_cfg["label"]
        is_classical = model_cfg["type"] in (
            "textblob", "vader", "hf_sentiment")
        is_fp16_base = label == "Qwen2-1.5B-FP16"

        print(f"\n{'='*60}")
        print(f"  Model: {label}")
        print(f"{'='*60}")

        try:
            backend = load_backend(model_cfg)
        except Exception as e:
            print(f"  [SKIPPED] {e}")
            summary.append({
                "model": label, "accuracy": None,
                "f1_macro": None, "skipped": True,
                "skip_reason": str(e),
            })
            continue

        preds, golds, inference_times = [], [], []

        for text, gold in tqdm(samples, desc=label):
            gold_norm = normalize_sentiment_label(gold)
            t0 = time.perf_counter()
            if is_classical:
                pred = predict_sentiment_classical(backend, text)
            else:
                pred = predict_sentiment_llm(backend, text)
            inference_times.append(
                (time.perf_counter() - t0) * 1000)
            preds.append(pred)
            golds.append(gold_norm)

        accuracy, f1_macro, per_class = sa_metrics(preds, golds)
        avg_ms  = sum(inference_times) / len(inference_times)
        total_s = sum(inference_times) / 1000

        print(f"\n  F1 macro   : {f1_macro:.4f}   <- PRIMARY")
        print(f"  Accuracy   : {accuracy:.4f}   (reference only)")
        print(f"  Avg latency: {avg_ms:.1f} ms/sample")
        print(f"  Total time : {total_s:.1f}s")

        print(f"\n  Per-class breakdown:")
        for cls, m in per_class.items():
            print(f"    {cls:10s}  P={m['precision']:.4f}  "
                  f"R={m['recall']:.4f}  F1={m['f1']:.4f}  "
                  f"tp={m['tp']} fp={m['fp']} fn={m['fn']}")

        print(f"\n  Pred dist  : {dict(Counter(preds))}")
        print(f"  Gold dist  : {dict(Counter(golds))}")

        # Parse failure report
        if not is_classical and _parse_failures:
            print(f"  [!] Parse failures (fell back to neutral): "
                  f"{len(_parse_failures)}")
            for pf in _parse_failures[:5]:
                print(f"      raw={pf['raw']!r:20}  "
                      f"text={pf['text'][:50]!r}")
            _parse_failures.clear()

        # Fidelity vs FP16
        fidelity = None
        if is_fp16_base:
            fp16_preds_store = preds[:]
        elif not is_classical and fp16_preds_store is not None:
            fidelity = quantization_fidelity_report(
                fp16_preds_store, preds, golds, label)

        safe_label = re.sub(r"[^\w\-]", "_", label)
        out_path   = os.path.join(
            OUTPUT_DIR, f"sentiment_{safe_label}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "model":              label,
                "f1_macro":           round(f1_macro, 4),
                "accuracy":           round(accuracy, 4),
                "avg_inference_ms":   round(avg_ms, 2),
                "total_inference_s":  round(total_s, 2),
                "n_samples":          len(samples),
                "stratified":         STRATIFY_SAMPLES,
                "label_distribution": dict(Counter(golds)),
                "pred_distribution":  dict(Counter(preds)),
                "per_class":          per_class,
                "fidelity_vs_fp16":   fidelity,
                "predictions": [
                    {
                        "text": s[0],
                        "gold": normalize_sentiment_label(s[1]),
                        "pred": p,
                    }
                    for s, p in zip(samples, preds)
                ],
            }, f, ensure_ascii=False, indent=2)
        print(f"  Saved → {out_path}")

        summary.append({
            "model":    label,
            "f1_macro": round(f1_macro, 4),
            "accuracy": round(accuracy, 4),
            "avg_ms":   round(avg_ms, 2),
            "neutral_collapse_rate": (
                fidelity["neutral_collapse_rate"] if fidelity else None),
        })
        backend.unload()

    # ── summary — sorted by F1-macro ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY  (sorted by F1-macro)")
    print("=" * 70)
    print(f"{'Model':<40} {'F1-mac':>8} {'Acc':>7} "
          f"{'ms/s':>8} {'NeutCollapse':>13}")
    print("-" * 70)

    ranked = sorted(
        summary, key=lambda s: s.get("f1_macro") or -1, reverse=True)
    for s in ranked:
        if s.get("skipped"):
            print(f"{s['model']:<40} SKIPPED  "
                  f"{s.get('skip_reason', '')}")
        else:
            nc = s["neutral_collapse_rate"]
            nc_str = f"{nc:.1%}" if nc is not None else "  n/a"
            print(f"{s['model']:<40} {s['f1_macro']:>8.4f} "
                  f"{s['accuracy']:>7.4f} {s['avg_ms']:>8.1f} "
                  f"{nc_str:>13}")

    summary_path = os.path.join(OUTPUT_DIR, "summary_sentiment.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(ranked, f, ensure_ascii=False, indent=2)
    print(f"\nSummary saved → {summary_path}")
    return ranked


if __name__ == "__main__":
    run_sentiment()
