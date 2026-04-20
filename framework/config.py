import os
import torch

# ==========================
# BASE DIRECTORY
# ==========================
# Defaults to the repo root (parent of framework/).
# Override by setting the QUANT_BASE_DIR environment variable,
# or by editing this file directly.
#
# Example (Windows):   set QUANT_BASE_DIR=E:/quant
# Example (Linux/Mac): export QUANT_BASE_DIR=/home/user/quant
# ==========================
BASE_DIR = os.environ.get(
    "QUANT_BASE_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

MODELS_DIR  = os.path.join(BASE_DIR, "models")
DATA_DIR    = os.path.join(BASE_DIR, "data")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")

# ==========================
# DATASET PATHS
# ==========================
DATASET_NER_PATH = os.environ.get(
    "DATASET_NER_PATH",
    os.path.join(DATA_DIR, "aij-wikiner-ru-wp3")
)
DATASET_SA_PATH = os.environ.get(
    "DATASET_SA_PATH",
    os.path.join(DATA_DIR, "rusentiment.csv")
)

# ==========================
# MODEL PATHS
# ==========================
BASE_MODEL_PATH = os.environ.get(
    "BASE_MODEL_PATH",
    os.path.join(MODELS_DIR, "qwen2-1.5b-instruct")
)

def model_path(name: str) -> str:
    """Return absolute path for a named model subdirectory."""
    return os.path.join(MODELS_DIR, name)

# ==========================
# RUN SETTINGS
# ==========================
MAX_NER_SAMPLES  = 500
MAX_SA_SAMPLES   = 500
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED      = 42
SHUFFLE_DATASET  = False

SENTIMENT_DATASET = {
    "type":   "file",
    "format": "rusentiment",
    "path":   DATASET_SA_PATH,
}

# ==========================
# NER MODEL LIST
# ==========================
NER_MODELS = [
    # ── classical baselines ──────────────────────────────────────────────
    {
        "label":      "spaCy-ru_core_news_sm",
        "type":       "spacy",
        "path":       "",
        "model_name": "ru_core_news_sm",
    },
    {
        "label":      "spaCy-ru_core_news_lg",
        "type":       "spacy",
        "path":       "",
        "model_name": "ru_core_news_lg",
    },
    {
        "label":      "HF-NER-LaBSE-NEREL",
        "type":       "hf_ner",
        "path":       "",
        "model_name": "surdan/LaBSE_ner_nerel",
    },

    # ── base model ───────────────────────────────────────────────────────
    {
        "label": "Qwen2-1.5B-FP16",
        "type":  "hf",
        "path":  BASE_MODEL_PATH,
    },

    # ── quantized variants ───────────────────────────────────────────────
    {
        "label":         "Qwen2-1.5B-TorchAO-int8",
        "type":          "torchao",
        "path":          BASE_MODEL_PATH,
        "torchao_dtype": "int8_weight_only",
    },
    {
        "label":        "Qwen2-1.5B-GGUF-Q4_K_M",
        "type":         "gguf",
        "path":         model_path("qwen2-1.5b-gguf"),
        "gguf_pattern": "Q4_K_M",
    },
    {
        "label": "Qwen2-1.5B-AWQ",
        "type":  "hf",
        "path":  model_path("qwen2-1.5b-awq"),
    },
    {
        "label":    "Qwen2-1.5B-bnb-4bit",
        "type":     "hf",
        "path":     BASE_MODEL_PATH,
        "bnb_bits": 4,
    },
    {
        "label":     "Qwen2-1.5B-HQQ-4bit",
        "type":      "hqq",
        "path":      BASE_MODEL_PATH,
        "hqq_bits":  4,
        "hqq_group": 64,
    },
    {
        "label":        "Qwen2-1.5B-Quanto-int4",
        "type":         "quanto",
        "path":         BASE_MODEL_PATH,
        "quanto_dtype": "int4",
    },
    {
        "label": "Qwen2-1.5B-OAQ-4bit",
        "type":  "oaq",
        "path":  model_path("qwen2-1.5b-oaq-4bit"),
    },
]

# ==========================
# SENTIMENT MODEL LIST
# ==========================
SA_MODELS = [
    # ── classical baselines ──────────────────────────────────────────────
    {
        "label": "TextBlob",
        "type":  "textblob",
    },
    {
        "label": "VADER",
        "type":  "vader",
    },
    {
        "label":      "HF-rubert-sentiment",
        "type":       "hf_sentiment",
        "model_name": "blanchefort/rubert-base-cased-sentiment",
    },

    # ── LLM variants ─────────────────────────────────────────────────────
    {
        "label": "Qwen2-1.5B-FP16",
        "type":  "hf",
        "path":  BASE_MODEL_PATH,
    },
    {
        "label":         "Qwen2-1.5B-TorchAO-int8",
        "type":          "torchao",
        "path":          BASE_MODEL_PATH,
        "torchao_dtype": "int8_weight_only",
    },
    {
        "label":        "Qwen2-1.5B-GGUF-Q4_K_M",
        "type":         "gguf",
        "path":         model_path("qwen2-1.5b-gguf"),
        "gguf_pattern": "Q4_K_M",
    },
    {
        "label":     "Qwen2-1.5B-HQQ-4bit",
        "type":      "hqq",
        "path":      BASE_MODEL_PATH,
        "hqq_bits":  4,
        "hqq_group": 64,
    },
    {
        "label":        "Qwen2-1.5B-Quanto-int4",
        "type":         "quanto",
        "path":         BASE_MODEL_PATH,
        "quanto_dtype": "int4",
    },
    {
        "label": "Qwen2-1.5B-OAQ-4bit",
        "type":  "oaq",
        "path":  model_path("qwen2-1.5b-oaq-4bit"),
    },
]
