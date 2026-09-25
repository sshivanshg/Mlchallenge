"""Unit tests for competition macro F0.5 (skill-aligned)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(REPO / ".agents/skills/aws-entity-resolution/scripts"))
sys.path.insert(0, str(SRC))

from evaluation.metric_ext import entity_f05, score_predictions  # noqa: E402
from metric import evaluate  # noqa: E402


class CompetitionMetricTests(unittest.TestCase):
    def test_required_edges(self):
        self.assertEqual(entity_f05(set(), set()), 1.0)
        self.assertEqual(entity_f05(set(), {"x"}), 0.0)
        self.assertEqual(entity_f05({"x"}, {"x"}), 1.0)
        self.assertEqual(entity_f05({"x"}, set()), 0.0)
        self.assertAlmostEqual(entity_f05({"x"}, {"x", "y"}), 5 / 9)
        self.assertAlmostEqual(entity_f05({"x", "y"}, {"x"}), 5 / 6)

    def test_macro_complete_universe(self):
        truth = {"a": set(), "b": {"1", "2"}, "c": {"3"}}
        pred = {"a": set(), "b": {"1"}, "c": {"3", "4"}}
        m = score_predictions(truth, pred)
        self.assertAlmostEqual(m["macro_F0.5"], (1 + 5 / 6 + 5 / 9) / 3)
        self.assertEqual(m["singleton_accuracy"], 1.0)

    def test_skill_evaluate_agreement(self):
        truth = {"a": set(), "b": {"z"}}
        pred = {"a": set(), "b": {"z"}}
        self.assertEqual(score_predictions(truth, pred), evaluate(truth, pred))


if __name__ == "__main__":
    unittest.main()
