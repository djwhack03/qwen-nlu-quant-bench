"""
Single entry point for the full framework.

Usage:
    python -m framework.run_all              # runs NER + SA + merges
    python -m framework.run_all --ner-only   # NER only
    python -m framework.run_all --sa-only    # SA only
    python -m framework.run_all --merge-only # re-merge existing results
"""

import os
import sys
import json
import argparse

from framework.config import OUTPUT_DIR


def merge_summaries():
    ner_path = os.path.join(OUTPUT_DIR, "summary_ner.json")
    sa_path  = os.path.join(OUTPUT_DIR, "summary_sentiment.json")

    unified = {"ner": [], "sentiment": []}

    if os.path.exists(ner_path):
        with open(ner_path, encoding="utf-8") as f:
            unified["ner"] = json.load(f)
    else:
        print("  [WARN] summary_ner.json not found — run NER first")

    if os.path.exists(sa_path):
        with open(sa_path, encoding="utf-8") as f:
            unified["sentiment"] = json.load(f)
    else:
        print("  [WARN] summary_sentiment.json not found — run SA first")

    out = os.path.join(OUTPUT_DIR, "summary_unified.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(unified, f, ensure_ascii=False, indent=2)

    # ── combined thesis table ─────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  UNIFIED RESULTS — NER F1 · SA Accuracy · SA F1-macro")
    print("=" * 75)
    print(f"{'Model':<42} {'NER-F1':>8} {'SA-Acc':>8} {'SA-F1':>8}")
    print("-" * 75)

    ner_map = {
        r["model"]: r for r in unified["ner"]
        if not r.get("skipped") and r.get("f1") is not None
    }
    sa_map = {
        r["model"]: r for r in unified["sentiment"]
        if not r.get("skipped") and r.get("accuracy") is not None
    }
    all_models = sorted(set(list(ner_map) + list(sa_map)))

    for m in all_models:
        ner_f1 = (f"{ner_map[m]['f1']:.4f}"
                  if m in ner_map else "   —   ")
        sa_acc = (f"{sa_map[m]['accuracy']:.4f}"
                  if m in sa_map else "   —   ")
        sa_f1  = (f"{sa_map[m]['f1_macro']:.4f}"
                  if m in sa_map else "   —   ")
        print(f"{m:<42} {ner_f1:>8} {sa_acc:>8} {sa_f1:>8}")

    print(f"\nUnified summary saved → {out}")
    return unified


def main():
    parser = argparse.ArgumentParser(
        description="NLU Quantization Framework")
    parser.add_argument("--ner-only",   action="store_true",
                        help="Run NER benchmark only")
    parser.add_argument("--sa-only",    action="store_true",
                        help="Run SA benchmark only")
    parser.add_argument("--merge-only", action="store_true",
                        help="Merge existing results only")
    args = parser.parse_args()

    run_ner_flag = not args.sa_only  and not args.merge_only
    run_sa_flag  = not args.ner_only and not args.merge_only

    if args.merge_only:
        merge_summaries()
        return

    if run_ner_flag:
        print("\n" + "=" * 70)
        print("  PHASE 1 / 2 — NER BENCHMARK")
        print("=" * 70 + "\n")
        from framework.run_ner import run_ner
        run_ner()

    if run_sa_flag:
        print("\n" + "=" * 70)
        print("  PHASE 3 — SENTIMENT ANALYSIS BENCHMARK")
        print("=" * 70 + "\n")
        from framework.run_sentiment import run_sentiment
        run_sentiment()

    print("\n" + "=" * 70)
    print("  MERGING ALL RESULTS")
    print("=" * 70 + "\n")
    merge_summaries()


if __name__ == "__main__":
    main()
