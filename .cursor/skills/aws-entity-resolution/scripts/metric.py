"""Offline set-level reference metric; no data loading, fitting, or networking."""

from collections.abc import Mapping
from math import fsum


def _ids(value):
    if not isinstance(value, (set, frozenset)):
        raise TypeError("Match collections must be sets, not strings or lists")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError("IDs must be nonempty opaque strings")
    return value


def entity_f05(truth, prediction):
    """Return singleton-aware F0.5 for one S1 entity."""
    truth, prediction = _ids(truth), _ids(prediction)
    if not truth and not prediction:
        return 1.0
    tp = len(truth & prediction)
    fp = len(prediction - truth)
    fn = len(truth - prediction)
    return 5 * tp / (5 * tp + 4 * fp + fn)


def evaluate(truth, predictions):
    """Require complete S1 maps; return macro target and micro diagnostics.

    Keys and target IDs are opaque strings; callers must source-qualify target
    IDs when needed. Truth must include every evaluated S1, even singletons.
    Zero-denominator diagnostic rates are None, not invented perfect scores.
    """
    if not isinstance(truth, Mapping) or not isinstance(predictions, Mapping):
        raise TypeError("Inputs must map S1 IDs to match sets")
    if not truth:
        raise ValueError("Cannot evaluate an empty S1 universe")
    _ids(set(truth))
    _ids(set(predictions))
    if set(truth) != set(predictions):
        raise ValueError("Prediction keys must exactly equal the S1 universe")
    scores = []
    tp = fp = fn = singletons = singleton_correct = 0
    for key in sorted(truth):
        actual, predicted = _ids(truth[key]), _ids(predictions[key])
        scores.append(entity_f05(actual, predicted))
        tp += len(actual & predicted)
        fp += len(predicted - actual)
        fn += len(actual - predicted)
        if not actual:
            singletons += 1
            singleton_correct += not predicted
    return {
        "macro_F0.5": fsum(scores) / len(scores),
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "singleton_accuracy": (
            singleton_correct / singletons if singletons else None
        ),
        "n_s1": len(truth),
        "n_singletons": singletons,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
