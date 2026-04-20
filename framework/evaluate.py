# ==========================
# NER METRICS
# ==========================
def ner_metrics(global_tp: int, global_fp: int, global_fn: int):
    precision = global_tp / (global_tp + global_fp + 1e-8)
    recall    = global_tp / (global_tp + global_fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    return round(precision, 4), round(recall, 4), round(f1, 4)


# ==========================
# SENTIMENT METRICS
# ==========================
def sa_metrics(preds: list, golds: list):
    correct  = sum(p == g for p, g in zip(preds, golds))
    accuracy = correct / len(golds)

    classes   = sorted(set(golds))
    f1s       = []
    per_class = {}

    for cls in classes:
        tp = sum(p == cls and g == cls for p, g in zip(preds, golds))
        fp = sum(p == cls and g != cls for p, g in zip(preds, golds))
        fn = sum(p != cls and g == cls for p, g in zip(preds, golds))
        p_cls  = tp / (tp + fp + 1e-8)
        r_cls  = tp / (tp + fn + 1e-8)
        f1_cls = 2 * p_cls * r_cls / (p_cls + r_cls + 1e-8)
        f1s.append(f1_cls)
        per_class[cls] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(p_cls, 4),
            "recall":    round(r_cls, 4),
            "f1":        round(f1_cls, 4),
        }

    f1_macro = sum(f1s) / len(f1s) if f1s else 0.0
    return round(accuracy, 4), round(f1_macro, 4), per_class


# ==========================
# MODEL SIZE ESTIMATE
# ==========================
def model_size_mb(model) -> float:
    """Estimate in-memory model size from parameters + buffers."""
    total  = sum(p.numel() * p.element_size() for p in model.parameters())
    total += sum(b.numel() * b.element_size() for b in model.buffers())
    return round(total / (1024 ** 2), 1)
