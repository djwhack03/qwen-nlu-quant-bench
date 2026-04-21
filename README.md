# qwen-nlu-quant-bench

A reproducible framework for evaluating LLM quantization methods on classical NLU tasks (NER and Sentiment Analysis).

Built as part of a bachelor's thesis on applying adaptive LLM distillation to classical NLP problems.

---

## What this is

This framework benchmarks **Qwen2-1.5B-Instruct** across 7 quantization methods against classical NLP baselines on two tasks:

- **NER** — Named Entity Recognition (person extraction) on Russian WikiNER
- **SA** — Sentiment Analysis on RuSentiment

The core research question: *does quantized LLM zero-shot inference approach or match classical task-specific methods on structured NLU tasks?*

---

## Results (NER · WikiNER · 500 samples)

| Model | Precision | Recall | F1 |
|---|---|---|---|
| spaCy-ru_core_news_lg | 0.913 | 0.861 | **0.886** |
| spaCy-ru_core_news_sm | 0.832 | 0.873 | 0.852 |
| Qwen2-1.5B FP16 | 0.912 | 0.725 | 0.808 |
| Qwen2-1.5B TorchAO int8 | 0.869 | 0.734 | 0.796 |
| Qwen2-1.5B bnb-8bit | 0.856 | 0.730 | 0.788 |
| Qwen2-1.5B Quanto int8 | 0.851 | 0.725 | 0.783 |
| Qwen2-1.5B GGUF Q4_K_M | 0.701 | 0.701 | 0.701 |
| Qwen2-1.5B Quanto int4 | 0.636 | 0.689 | 0.661 |
| Qwen2-1.5B AWQ 4bit | 0.590 | 0.738 | 0.656 |
| Qwen2-1.5B OAQ 4bit | 0.554 | 0.734 | 0.631 |
| Qwen2-1.5B bnb-4bit | 0.991 | 0.443 | 0.612 |
| Qwen2-1.5B HQQ 4bit | 0.469 | 0.721 | 0.569 |

**Key finding:** INT8 quantization (TorchAO) loses only 1.5% F1 vs FP16. INT4 methods drop 13–30%. No quantized LLM variant beats spaCy-lg in zero-shot mode.

---

## Results (SA · RuSentiment · 500 samples)

| Model | Accuracy | F1_macro |
|---|---|---|
| HF-rubert-sentiment | 0.79 | 0.783 |
| Qwen2-1.5B FP16 | 0.726 | 0.722 |
| Qwen2-1.5B GGUF Q4_K_M | 0.654 | 0.656 |
| Qwen2-1.5B bnb-4bit | 0.632 | 0.635 |
| Qwen2-1.5B Quanto int4 | 0.602 | 0.604 |
| Qwen2-1.5B HQQ 4bit | 0.582 | 0.502 |

---

## Repository structure

```
qwen-nlu-quant-bench/
│
├── framework/                  # main package
│   ├── __init__.py
|   ├── backends.py             # HF / GGUF / OAQ / Classical backends
│   ├── config.py               # all paths, model lists, settings
│   ├── datasets.py             # WikiNER, RuSentiment, SST-2 loaders
│   ├── evaluate.py             # F1, accuracy, macro-F1, model size
│   ├── inference.py            # generate_persons, predict_sentiment
│   ├── postprocess.py          # filtering, normalization, soft matching
│   ├── prompts.py              # NER and SA prompts
│   └── run_all.py              # unified entry point
│   ├── run_ner.py              # NER benchmark loop
│   ├── run_sentiment.py        # SA benchmark loop
│── results/                    # framework outputs
├── quantize_oaq.py             # OAQ quantization script
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/djwhack03/qwen-nlu-quant-bench
cd qwen-nlu-quant-bench
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### Data

- **WikiNER (Russian):** download `aij-wikiner-ru-wp3` from [Hugging Face](https://huggingface.co/datasets/ayakiri/aij-wikiner-ru) and place at `E:/quant/data/aij-wikiner-ru-wp3`
- **RuSentiment:** download CSV from [text-machine-lab/rusentiment](https://github.com/text-machine-lab/rusentiment) and place at `E:/quant/data/rusentiment.csv`

### Models

Base model and most quantized variants download automatically on first run.  
OAQ requires manual quantization first:

```bash
python quantize_oaq.py
```

---

## Running

```bash
# full benchmark (NER + SA)
python -m framework.run_all

# NER only
python -m framework.run_all --ner-only

# SA only
python -m framework.run_all --sa-only

# re-merge existing results
python -m framework.run_all --merge-only
```

Results are saved to `E:/quant/outputs/` as per-model JSON files and summary JSONs:
- `summary_ner.json`
- `summary_sentiment.json`
- `summary_unified.json`

---

## Quantization methods compared

| Method | Bits | Library |
|---|---|---|
| FP16 baseline | 16 | transformers |
| TorchAO | 8 | torchao |
| GGUF Q4_K_M | 4 | llama.cpp |
| AWQ | 4 | autoawq |
| bitsandbytes | 4 | bitsandbytes |
| HQQ | 4 | hqq |
| Quanto | 4 | optimum-quanto |
| OAQ (custom) | 4 | this repo |

---

## Classical baselines compared

| Model | Task |
|---|---|
| spaCy ru_core_news_sm | NER |
| spaCy ru_core_news_lg | NER |
| rubert-base-cased-sentiment | SA |

---

## Requirements

```
torch>=2.4.0
transformers>=4.40.0
peft>=0.10.0
bitsandbytes
torchao>=0.16.0
hqq
optimum-quanto
llama-cpp-python
spacy
tqdm
huggingface_hub
sentencepiece
textblob
nltk
datasets
```

---

## Citation

If you use this framework, please cite:

```
@misc{qwen-nlu-quant-bench,
  author = {djwhack03},
  title  = {qwen-nlu-quant-bench: LLM Quantization Benchmark for NLU Tasks},
  year   = {2026},
  url    = {https://github.com/djwhack03/qwen-nlu-quant-bench}
}
```
