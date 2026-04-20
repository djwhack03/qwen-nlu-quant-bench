import os
import re
import json
import time

os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
sys.modules["tensorflow"] = None
sys.modules["tensorflow_core"] = None

from collections import Counter
from tqdm import tqdm

from framework.config import (
    OUTPUT_DIR, MAX_SA_SAMPLES, SENTIMENT_DATASET, SA_MODELS,
)
from framework.datasets import (
    load_rusentiment, load_sst2, load_sentiment_hf,
)
from framework.backends import load_backend, ClassicalSABackend
from framework.inference import (
    predict_sentiment_llm, predict_sentiment_classical,
)
from framework.postprocess import normalize_sentiment_label
from framework.evaluate import sa_metrics


def run_sentiment():
    print("=" * 70)
    print("  SENTIMENT ANALYSIS BENCHMARK")
    print("=" * 70)

    # ── load data ─────────────────────────────────────────────────────────
    cfg = SENTIMENT_DATASET
    if cfg["type"] == "hf":
        samples = load_sentiment_hf(
            cfg["name"], cfg.get("split", "validation"),
            cfg.get("text_col", "sentence"),
            cfg.get("label_col", "label"))
    elif cfg["type"] == "file" and cfg["format"] == "rusentiment":
        samples = load_rusentiment(cfg["path"])
    elif cfg["type"] == "file" and cfg["format"] == "sst2":
        samples = load_sst2(cfg["path"])
    else:
        raise ValueError(f"Unknown dataset config: {cfg}")

    samples = samples[:MAX_SA_SAMPLES]
    dist    = Counter(normalize_sentiment_label(s[1]) for s in samples)
    print(f"\n  Dataset : {cfg.get('name', cfg.get('path'))}")
    print(f"  Samples : {len(samples)}")
    print(f"  Labels  : {dict(dist)}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary = []

    for model_cfg in SA_MODELS:
        label        = model_cfg["label"]
        is_classical = model_cfg["type"] in (
            "textblob", "vader", "hf_sentiment")

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

        print(f"\n  Accuracy   : {accuracy:.4f}")
        print(f"  F1 macro   : {f1_macro:.4f}")
        print(f"  Avg latency: {avg_ms:.1f} ms/sample")
        print(f"  Total time : {total_s:.1f}s")

        safe_label = re.sub(r"[^\w\-]", "_", label)
        out_path   = os.path.join(
            OUTPUT_DIR, f"sentiment_{safe_label}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "model":             label,
                "accuracy":          accuracy,
                "f1_macro":          f1_macro,
                "avg_inference_ms":  round(avg_ms, 2),
                "total_inference_s": round(total_s, 2),
                "per_class":         per_class,
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
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "avg_ms":   round(avg_ms, 2),
        })
        backend.unload()

    # ── summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SENTIMENT SUMMARY")
    print("=" * 70)
    print(f"{'Model':<45} {'Acc':>7} {'F1-mac':>8} {'ms/s':>8}")
    print("-" * 70)
    for s in summary:
        if s.get("skipped"):
            print(f"{s['model']:<45} SKIPPED  "
                  f"{s.get('skip_reason', '')}")
        else:
            print(f"{s['model']:<45} "
                  f"{s['accuracy']:>7.4f} "
                  f"{s['f1_macro']:>8.4f} "
                  f"{s['avg_ms']:>8.1f}")

    summary_path = os.path.join(OUTPUT_DIR, "summary_sentiment.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSummary saved → {summary_path}")
    return summary


if __name__ == "__main__":
    run_sentiment()
