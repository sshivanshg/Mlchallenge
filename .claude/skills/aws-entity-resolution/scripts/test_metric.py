"""Run with unittest discover; fixtures contain no external entity data."""

import unittest
from metric import entity_f05, evaluate


class MetricTests(unittest.TestCase):
    def test_singletons_and_empty_prediction(self):
        self.assertEqual(entity_f05(set(), set()), 1.0)
        self.assertEqual(entity_f05(set(), {"S2:a"}), 0.0)
        self.assertEqual(entity_f05({"S2:a"}, set()), 0.0)

    def test_exact_disjoint_and_precision_weight(self):
        self.assertEqual(entity_f05({"a"}, {"a"}), 1.0)
        self.assertEqual(entity_f05({"a"}, {"b"}), 0.0)
        self.assertAlmostEqual(entity_f05({"a"}, {"a", "b"}), 5 / 9)
        self.assertAlmostEqual(entity_f05({"a", "b"}, {"a"}), 5 / 6)

    def test_multiple_sources_remain_distinct(self):
        self.assertAlmostEqual(
            entity_f05({"S2:001", "S3:001"}, {"S2:001"}), 5 / 6
        )

    def test_macro_is_not_micro(self):
        truth = {"001": set(), "002": {"a", "b"}, "003": {"c"}}
        prediction = {"001": set(), "002": {"a"}, "003": {"c", "d"}}
        result = evaluate(truth, prediction)
        self.assertAlmostEqual(result["macro_F0.5"], (1 + 5 / 6 + 5 / 9) / 3)
        self.assertAlmostEqual(result["precision"], 2 / 3)
        self.assertAlmostEqual(result["recall"], 2 / 3)
        self.assertEqual(result["singleton_accuracy"], 1.0)
        self.assertNotAlmostEqual(result["macro_F0.5"], 2 / 3)

    def test_undefined_diagnostics(self):
        result = evaluate({"x": set()}, {"x": set()})
        self.assertEqual(result["macro_F0.5"], 1.0)
        self.assertIsNone(result["precision"])
        self.assertIsNone(result["recall"])
        result = evaluate({"x": {"a"}}, {"x": {"a"}})
        self.assertIsNone(result["singleton_accuracy"])

    def test_complete_universe_required(self):
        for prediction in ({}, {"x": set(), "extra": set()}):
            with self.assertRaises(ValueError):
                evaluate({"x": set()}, prediction)
        with self.assertRaises(ValueError):
            evaluate({}, {})

    def test_strict_input_contract(self):
        for bad in ("abc", ["a", "a"]):
            with self.assertRaises(TypeError):
                entity_f05(set(), bad)
        for bad in ({1}, {""}):
            with self.assertRaises(ValueError):
                entity_f05(set(), bad)

    def test_permutation_and_known_counts(self):
        first = evaluate({"b": {"q"}, "a": set()}, {"a": {"x"}, "b": {"q"}})
        second = evaluate({"a": set(), "b": {"q"}}, {"b": {"q"}, "a": {"x"}})
        self.assertEqual(first, second)
        self.assertEqual((first["tp"], first["fp"], first["fn"]), (1, 1, 0))
        self.assertEqual(first["macro_F0.5"], 0.5)


if __name__ == "__main__":
    unittest.main()
