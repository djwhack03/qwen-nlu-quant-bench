import csv


# ==========================
# NER — WikiNER
# ==========================
def load_wikiner(path: str):
    sentences, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            words = line.split()
            tokens, tags = [], []
            for word in words:
                parts = word.split("|")
                if len(parts) == 3:
                    tokens.append(parts[0])
                    tags.append(parts[2])
            sentences.append(tokens)
            labels.append(tags)
    return sentences, labels


def extract_gold_persons(tokens: list, tags: list) -> list:
    persons, current = [], []
    for token, tag in zip(tokens, tags):
        if tag == "B-PER":
            if current:
                persons.append(" ".join(current))
            current = [token]
        elif tag == "I-PER":
            current.append(token)
        else:
            if current:
                persons.append(" ".join(current))
                current = []
    if current:
        persons.append(" ".join(current))
    return persons


# ==========================
# SENTIMENT — RuSentiment
# ==========================
def load_rusentiment(path: str) -> list:
    keep = {"positive", "negative", "neutral"}
    samples = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("label", row.get("sentiment", "")).strip().lower()
            if label in keep:
                samples.append((row["text"].strip(), label))
    return samples


# ==========================
# SENTIMENT — SST-2
# ==========================
def load_sst2(path: str) -> list:
    samples = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            samples.append((row["sentence"].strip(), int(row["label"])))
    return samples


# ==========================
# SENTIMENT — HuggingFace datasets
# ==========================
def load_sentiment_hf(dataset_name: str, split: str = "validation",
                      text_col: str = "sentence",
                      label_col: str = "label") -> list:
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
