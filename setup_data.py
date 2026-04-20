"""
First-time setup: downloads all models and datasets.

Usage:
    python setup_data.py

Optional — override base directory:
    Windows : set QUANT_BASE_DIR=D:/myproject && python setup_data.py
    Linux   : QUANT_BASE_DIR=/data/quant python setup_data.py

What gets downloaded:
    models/qwen2-1.5b-instruct   — Qwen/Qwen2-1.5B-Instruct        (~3 GB)
    models/qwen2-1.5b-awq        — Qwen/Qwen2-1.5B-Instruct-AWQ    (~1 GB)
    models/qwen2-1.5b-gguf       — Q4_K_M GGUF file                 (~1 GB)
    data/aij-wikiner-ru-wp3      — Russian WikiNER dataset           (small)

    RuSentiment must be downloaded manually (see instructions below).
"""

import os
import sys


# ---------------------------------------------------------------------------
# Bootstrap: make sure framework.config can be imported before any deps load
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from framework.config import (
    BASE_DIR, MODELS_DIR, DATA_DIR, OUTPUT_DIR,
    BASE_MODEL_PATH, DATASET_NER_PATH, DATASET_SA_PATH,
    model_path,
)

# ---------------------------------------------------------------------------
# Models to download via snapshot_download
# ---------------------------------------------------------------------------
HF_MODELS = {
    BASE_MODEL_PATH:             "Qwen/Qwen2-1.5B-Instruct",
    model_path("qwen2-1.5b-awq"): "Qwen/Qwen2-1.5B-Instruct-AWQ",
}

GGUF_MODELS = {
    model_path("qwen2-1.5b-gguf"): {
        "repo":    "MaziyarPanahi/Qwen2-1.5B-Instruct-GGUF",
        "pattern": "Q4_K_M",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _has_weights(path: str) -> bool:
    return (
        os.path.isdir(path) and
        any(f.endswith((".safetensors", ".bin"))
            for f in os.listdir(path))
    )


def _has_gguf(path: str, pattern: str) -> bool:
    return (
        os.path.isdir(path) and
        any(
            f.endswith(".gguf") and pattern.lower() in f.lower()
            for f in os.listdir(path)
        )
    )


# ---------------------------------------------------------------------------
# Step 1 — create directory structure
# ---------------------------------------------------------------------------
def create_dirs():
    print("Creating directory structure...")
    for d in [MODELS_DIR, DATA_DIR, OUTPUT_DIR]:
        os.makedirs(d, exist_ok=True)
        print(f"  {d}")
    print()


# ---------------------------------------------------------------------------
# Step 2 — download HF models
# ---------------------------------------------------------------------------
def download_hf_models():
    print("=== Downloading HuggingFace models ===")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        import subprocess
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import snapshot_download

    for local_path, repo_id in HF_MODELS.items():
        if _has_weights(local_path):
            print(f"  [exists]  {repo_id}")
            continue
        print(f"  Downloading {repo_id} ...")
        os.makedirs(local_path, exist_ok=True)
        snapshot_download(repo_id=repo_id, local_dir=local_path)
        print(f"  Saved  →  {local_path}\n")
    print()


# ---------------------------------------------------------------------------
# Step 3 — download GGUF models
# ---------------------------------------------------------------------------
def download_gguf_models():
    print("=== Downloading GGUF models ===")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        pass  # already installed above

    from huggingface_hub import snapshot_download

    for local_path, cfg in GGUF_MODELS.items():
        pattern = cfg["pattern"]
        if _has_gguf(local_path, pattern):
            print(f"  [exists]  {cfg['repo']} ({pattern})")
            continue
        print(f"  Downloading {cfg['repo']} — {pattern} ...")
        os.makedirs(local_path, exist_ok=True)
        snapshot_download(
            repo_id=cfg["repo"],
            local_dir=local_path,
            allow_patterns=[f"*{pattern}*.gguf"],
        )
        print(f"  Saved  →  {local_path}\n")
    print()


# ---------------------------------------------------------------------------
# Step 4 — download WikiNER dataset
# ---------------------------------------------------------------------------
def download_wikiner():
    print("=== Downloading WikiNER dataset ===")

    if os.path.isfile(DATASET_NER_PATH):
        print(f"  [exists]  {DATASET_NER_PATH}\n")
        return

    print("  Downloading aij-wikiner-ru from HuggingFace datasets...")
    try:
        from datasets import load_dataset
    except ImportError:
        import subprocess
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "datasets"])
        from datasets import load_dataset

    ds = load_dataset("ayakiri/aij-wikiner-ru", split="train")

    # Convert to the pipe-separated format expected by load_wikiner()
    # Format per token: WORD|POS|NER  — one sentence per line
    os.makedirs(os.path.dirname(DATASET_NER_PATH), exist_ok=True)

    # Tag index → string mapping for WikiNER
    ner_map = {
        0: "O",
        1: "B-PER", 2: "I-PER",
        3: "B-ORG", 4: "I-ORG",
        5: "B-LOC", 6: "I-LOC",
        7: "B-MISC", 8: "I-MISC",
    }

    written = 0
    with open(DATASET_NER_PATH, "w", encoding="utf-8") as f:
        for row in ds:
            tokens   = row["tokens"]
            ner_ids  = row["ner_tags"]
            pos_tags = row.get("pos_tags", None)

            ner_strs = [ner_map.get(n, "O") for n in ner_ids]

            if pos_tags and len(pos_tags) == len(tokens):
                parts = [f"{t}|{p}|{n}"
                         for t, p, n in zip(tokens, pos_tags, ner_strs)]
            else:
                parts = [f"{t}|NN|{n}"
                         for t, n in zip(tokens, ner_strs)]

            f.write(" ".join(parts) + "\n")
            written += 1

    print(f"  Written {written} sentences → {DATASET_NER_PATH}\n")


# ---------------------------------------------------------------------------
# Step 5 — check RuSentiment (manual download required)
# ---------------------------------------------------------------------------
def check_rusentiment():
    print("=== RuSentiment dataset ===")
    if os.path.isfile(DATASET_SA_PATH):
        print(f"  [exists]  {DATASET_SA_PATH}\n")
        return

    print("  [MANUAL DOWNLOAD REQUIRED]")
    print()
    print("  RuSentiment cannot be downloaded automatically.")
    print("  Please follow these steps:")
    print()
    print("  1. Go to: https://github.com/text-machine-lab/rusentiment")
    print("  2. Download 'rusentiment_random_posts.csv'")
    print(f"  3. Place it at: {DATASET_SA_PATH}")
    print()
    print("  The SA benchmark will be skipped until this file is present.")
    print()


# ---------------------------------------------------------------------------
# Step 6 — install spaCy Russian models
# ---------------------------------------------------------------------------
def install_spacy_models():
    print("=== Installing spaCy Russian models ===")
    import subprocess
    for model in ("ru_core_news_sm", "ru_core_news_lg"):
        try:
            import spacy
            spacy.load(model)
            print(f"  [exists]  {model}")
        except Exception:
            print(f"  Installing {model} ...")
            subprocess.check_call(
                [sys.executable, "-m", "spacy", "download", model])
    print()


# ---------------------------------------------------------------------------
# Step 7 — OAQ reminder
# ---------------------------------------------------------------------------
def oaq_reminder():
    oaq_path = model_path("qwen2-1.5b-oaq-4bit")
    oaq_cfg  = os.path.join(oaq_path, "oaq_config.json")
    print("=== OAQ model ===")
    if os.path.isfile(oaq_cfg):
        print(f"  [exists]  {oaq_path}\n")
    else:
        print("  OAQ model must be generated manually.")
        print("  Run:  python quantize_oaq.py")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"\nBase directory: {BASE_DIR}")
    print(f"Models dir    : {MODELS_DIR}")
    print(f"Data dir      : {DATA_DIR}")
    print(f"Outputs dir   : {OUTPUT_DIR}")
    print()

    create_dirs()
    download_hf_models()
    download_gguf_models()
    download_wikiner()
    check_rusentiment()
    install_spacy_models()
    oaq_reminder()

    print("=" * 60)
    print("  Setup complete.")
    print()
    print("  Next steps:")
    print("  1. (if needed) Download RuSentiment manually — see above")
    print("  2. python quantize_oaq.py   — build OAQ model")
    print("  3. python -m framework.run_all  — run full benchmark")
    print("=" * 60)


if __name__ == "__main__":
    main()
