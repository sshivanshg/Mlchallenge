"""Macro F0.5 — thin wrapper over the installed aws-entity-resolution skill metric.

Canonical implementation: ``.agents/skills/aws-entity-resolution/scripts/metric.py``.
Argument order for ``entity_f05`` / ``evaluate`` follows the skill: (truth, prediction).
"""

from __future__ import annotations

import sys
from pathlib import Path

_SKILL_SCRIPTS = (
    Path(__file__).resolve().parents[3]
    / ".agents"
    / "skills"
    / "aws-entity-resolution"
    / "scripts"
)
if _SKILL_SCRIPTS.is_dir() and str(_SKILL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SKILL_SCRIPTS))

from metric import entity_f05, evaluate  # noqa: E402  (skill script)


def macro_f05(
    predictions: dict[str, set[str]],
    ground_truth: dict[str, set[str]],
) -> float:
    """End-to-end macro F0.5; requires complete S1 key coverage in both maps."""
    preds = {k: set(predictions.get(k, set())) for k in ground_truth}
    return float(evaluate(ground_truth, preds)["macro_F0.5"])


def evaluate_predictions(
    predictions: dict[str, set[str]],
    ground_truth: dict[str, set[str]],
) -> dict:
    preds = {k: set(predictions.get(k, set())) for k in ground_truth}
    return evaluate(ground_truth, preds)


__all__ = ["entity_f05", "evaluate", "evaluate_predictions", "macro_f05"]
