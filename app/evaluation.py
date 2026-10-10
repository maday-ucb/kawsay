"""Retrieval metrics and gate threshold selection."""
import math
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def ranking_metrics(ranking: list[str], relevant: list[str], k: int = 5) -> dict:
    """
    Compute NDCG@k, Recall@k, Precision@k and MRR@k.
    Only call this for answerable queries (relevant must be non-empty).
    """
    if not relevant:
        raise ValueError("ranking_metrics requires at least one relevant chunk.")

    relevant_set = set(relevant)
    hits = [int(item in relevant_set) for item in ranking[:k]]

    dcg   = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant_set))))
    mrr   = next((1 / (i + 1) for i, h in enumerate(hits) if h), 0.0)

    return {
        "ndcg_at_5":      dcg / ideal,
        "recall_at_5":    sum(hits) / len(relevant_set),
        "precision_at_5": sum(hits) / k,
        "mrr_at_5":       mrr,
    }


def gate_metrics(labels: list[bool], scores: list[float],
                 threshold: float) -> dict:
    """Evaluate the binary gate that decides whether to attempt generation."""
    predicted = np.asarray(scores) >= threshold
    tn, fp, fn, tp = confusion_matrix(
        labels, predicted, labels=[False, True]
    ).ravel()
    return {
        "f1":        float(f1_score(labels, predicted, zero_division=0)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall":    float(recall_score(labels, predicted, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def select_gate(labels: list[bool], scores: list[float]) -> dict:
    """
    Choose the threshold that maximises F1 on the validation set.
    Ties broken by higher precision, then higher threshold.
    """
    thresholds = np.append(
        np.unique(scores), np.nextafter(float(max(scores)), np.inf)
    )
    candidates = [
        {"threshold": float(t), **gate_metrics(labels, scores, float(t))}
        for t in thresholds
    ]
    return max(candidates, key=lambda r: (r["f1"], r["precision"], r["threshold"]))
