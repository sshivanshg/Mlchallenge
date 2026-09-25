"""Macro F0.5 scorer matching the challenge definition (incl. singletons)."""

from __future__ import annotations


def f05(precision: float, recall: float) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)


def entity_f05(pred: set[str], truth: set[str]) -> float:
    if not truth and not pred:
        return 1.0
    if not truth and pred:
        return 0.0
    if truth and not pred:
        return 0.0
    tp = len(pred & truth)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(truth) if truth else 0.0
    return f05(precision, recall)


def macro_f05(
    predictions: dict[str, set[str]],
    ground_truth: dict[str, set[str]],
) -> float:
    scores = []
    for s1, truth in ground_truth.items():
        pred = predictions.get(s1, set())
        scores.append(entity_f05(pred, truth))
    return sum(scores) / len(scores) if scores else 0.0
