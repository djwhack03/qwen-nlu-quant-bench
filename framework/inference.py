from framework.prompts import SYSTEM_PROMPT, VERIFY_PROMPT, SENTIMENT_PROMPT
from framework.postprocess import (
    extract_json_persons, is_valid_person,
    all_words_in_sentence, appears_capitalized_in_sentence,
    deduplicate_predictions, normalize_sentiment_label,
)
from framework.backends import ClassicalNERBackend, ClassicalSABackend


# ==========================
# NER — LLM PATH
# ==========================
def verify_person(backend, name: str, sentence: str) -> bool:
    messages = [
        {"role": "system", "content": VERIFY_PROMPT},
        {"role": "user",
         "content": (f'Sentence: {sentence}\n'
                     f'Is "{name}" a person\'s name? '
                     f'Answer YES or NO only.')},
    ]
    response = backend.generate(
        messages, max_new_tokens=5, do_sample=False).upper()
    return response.startswith("YES")


def generate_persons(backend, sentence: str) -> list:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": sentence},
    ]
    response = backend.generate(
        messages, max_new_tokens=400,
        do_sample=False, repetition_penalty=1.0)
    raw      = extract_json_persons(response)
    filtered = [p for p in raw if is_valid_person(p)]
    filtered = [p for p in filtered
                if all_words_in_sentence(p, sentence)]
    filtered = [p for p in filtered
                if appears_capitalized_in_sentence(p, sentence)]
    filtered = deduplicate_predictions(filtered)
    return [p for p in filtered
            if verify_person(backend, p, sentence)]


# ==========================
# NER — CLASSICAL PATH
# ==========================
def generate_persons_classical(backend: ClassicalNERBackend,
                                sentence: str) -> list:
    raw      = backend.extract_persons(sentence)
    filtered = [p for p in raw if is_valid_person(p)]
    filtered = [p for p in filtered
                if all_words_in_sentence(p, sentence)]
    filtered = [p for p in filtered
                if appears_capitalized_in_sentence(p, sentence)]
    return deduplicate_predictions(filtered)


# ==========================
# SENTIMENT — LLM PATH
# FIX: strip label prefixes before keyword matching, add Russian keywords,
# and track parse failures (responses that fell back to neutral).
# ==========================

# Populated during a run; cleared by run_sentiment after each model.
_parse_failures: list = []


def predict_sentiment_llm(backend, text: str) -> str:
    messages = [
        {"role": "system", "content": SENTIMENT_PROMPT},
        {"role": "user",   "content": text},
    ]
    raw = backend.generate(
        messages, max_new_tokens=8, do_sample=False).lower().strip()

    # Strip common label prefixes the model may prepend
    for prefix in ("label:", "ответ:", "sentiment:", "answer:", "output:"):
        if prefix in raw:
            raw = raw.split(prefix, 1)[-1].strip()

    if "pos" in raw:    return "positive"
    if "neg" in raw:    return "negative"
    if "neu" in raw:    return "neutral"
    if "позит" in raw:  return "positive"
    if "негат" in raw:  return "negative"
    if "нейтр" in raw:  return "neutral"

    _parse_failures.append({"text": text[:80], "raw": raw})
    return "neutral"


# ==========================
# SENTIMENT — CLASSICAL PATH
# ==========================
def predict_sentiment_classical(backend: ClassicalSABackend,
                                 text: str) -> str:
    return backend.predict(text)
