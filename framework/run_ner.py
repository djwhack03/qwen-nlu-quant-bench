import os
import re
import json
import time

os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
sys.modules["tensorflow"] = None
sys.modules["tensorflow_core"] = None

from tqdm import tqdm

from framework.config import (
    DATASET_NER_PATH, OUTPUT_DIR, MAX_NER_SAMPLES,
    SHUFFLE_DATASET, RANDOM_SEED, NER_MODELS,
)
from framework.datasets import load_wikiner, extract_gold_persons
from framework.backends import load_backend, ClassicalNERBackend
from framework.inference import generate_persons, generate_persons_classical
from framework.postprocess import soft_match
from framework.evaluate import ner_metrics


def run_ner():
    print("=" * 70)
    print("  NER BENCHMARK")
    print("=" * 70)

    # ── load data ─────────────────────────────────────────────────────────
    print("\nLoading WikiNER dataset...")
    sentences, labels = load_wikiner(DATASET_NER_PATH)

    if SHUFFLE_DATASET:
        import random
        random.seed(RANDOM_SEED)
        paired = list(zip(sentences, labels))
        random.shuffle(paired)
        sentences, labels = zip(*paired)
        sentences, labels = list(sentences), list(labels)
        print(f"  Shuffled (seed={RANDOM_SEED})")
    else:
        print("  Using original file order")

    sentences = sentences[:MAX_NER_SAMPLES]
    labels    = labels[:MAX_NER_SAMPLES]
    print(f"  Samples: {len(sentences)}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary = []

    for model_cfg in NER_MODELS:
        label = model_cfg["label"]

        # OAQ pre-check
        if model_cfg["type"] == "oaq":
            oaq_check = os.path.join(model_cfg["path"], "oaq_config.json")
            if not os.path.exists(oaq_check):
                print(f"\n  [SKIPPED] {label} — run quantize_oaq.py first")
                summary.append({
                    "model": label, "precision": None,
                    "recall": None, "f1": None,
                    "skipped": True,
                    "skip_reason": "OAQ model not generated yet",
                })
                continue

        print(f"\n{'='*60}")
        print(f"  Model: {label}")
        print(f"{'='*60}")

        try:
            backend = load_backend(model_cfg)
        except Exception as e:
            print(f"  [SKIPPED] {e}")
            summary.append({
                "model": label, "precision": None,
                "recall": None, "f1": None,
                "skipped": True, "skip_reason": str(e),
            })
            continue

        is_classical = isinstance(backend, ClassicalNERBackend)
        global_tp = global_fp = global_fn = 0
        all_results, inference_times = [], []

        for tokens, tags in tqdm(
                zip(sentences, labels),
                total=len(sentences), desc=label):
            sentence     = " ".join(tokens)
            gold_persons = extract_gold_persons(tokens, tags)

            t0 = time.perf_counter()
            if is_classical:
                pred_persons = generate_persons_classical(
                    backend, sentence)
            else:
                pred_persons = generate_persons(backend, sentence)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            tp, fp, fn = soft_match(gold_persons, pred_persons)
            global_tp += tp
            global_fp += fp
            global_fn += fn
            inference_times.append(elapsed_ms)

            all_results.append({
                "sentence":          sentence,
                "gold_persons":      gold_persons,
                "predicted_persons": pred_persons,
                "tp": tp, "fp": fp, "fn": fn,
                "inference_ms":      round(elapsed_ms, 2),
            })

        precision, recall, f1 = ner_metrics(
            global_tp, global_fp, global_fn)
        avg_ms  = sum(inference_times) / len(inference_times)
        total_s = sum(inference_times) / 1000

        print(f"\n  Precision : {precision:.4f}")
        print(f"  Recall    : {recall:.4f}")
        print(f"  F1        : {f1:.4f}")
        print(f"  TP={global_tp}  FP={global_fp}  FN={global_fn}")
        print(f"  Avg       : {avg_ms:.1f} ms/sentence")
        print(f"  Total     : {total_s:.1f}s")

        safe_label  = re.sub(r"[^\w\-]", "_", label)
        output_path = os.path.join(
            OUTPUT_DIR, f"predictions_{safe_label}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "model":             label,
                "precision":         precision,
                "recall":            recall,
                "f1":                f1,
                "tp":                global_tp,
                "fp":                global_fp,
                "fn":                global_fn,
                "avg_inference_ms":  round(avg_ms, 2),
                "total_inference_s": round(total_s, 2),
                "results":           all_results,
            }, f, ensure_ascii=False, indent=2)
        print(f"  Saved → {output_path}")

        summary.append({
            "model":     label,
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
            "tp":        global_tp,
            "fp":        global_fp,
            "fn":        global_fn,
            "avg_ms":    round(avg_ms, 2),
        })

        backend.unload()

    # ── summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  NER SUMMARY")
    print("=" * 70)
    print(f"{'Model':<45} {'P':>7} {'R':>7} {'F1':>7} {'ms/s':>8}")
    print("-" * 70)
    for s in summary:
        if s.get("skipped"):
            print(f"{s['model']:<45} SKIPPED  "
                  f"{s.get('skip_reason', '')}")
        else:
            print(f"{s['model']:<45} "
                  f"{s['precision']:>7.4f} "
                  f"{s['recall']:>7.4f} "
                  f"{s['f1']:>7.4f} "
                  f"{s['avg_ms']:>8.1f}")

    summary_path = os.path.join(OUTPUT_DIR, "summary_ner.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSummary saved → {summary_path}")
    return summary


if __name__ == "__main__":
    run_ner()
