"""Isolated parser tests; these fixtures are never used for competition scoring."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from run_official_audit import _scan_verified_dataset  # noqa: E402


class OfficialAuditTests(unittest.TestCase):
    def test_empty_baseline_and_label_integrity(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            train = root / "train"
            train.mkdir()
            header = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            (train / "train_source1.tsv").write_text(
                header + "S1-a\tAlpha\tRoad\tUS\nS1-b\tBeta\t\tIndia\n"
            )
            (train / "train_source2.tsv").write_text(header + "S2-a\tAlpha\tRoad\tUS\n")
            (train / "train_source3.tsv").write_text(header + "S3-a\tBeta\tLane\tIndia\n")
            gt = train / "train_ground_truth.tsv"
            gt.write_text(
                "source1_entity_id\tmatched_entity_ids\n"
                "S1-a\tS2-a,S3-a\nS1-b\t\n"
            )
            result = _scan_verified_dataset(root, root / "index.sqlite")
            self.assertEqual(result["sources"]["s1"]["records"], 2)
            self.assertEqual(result["ground_truth"]["links"], 2)
            self.assertEqual(result["ground_truth"]["singleton_rate"], 0.5)
            self.assertEqual(result["empty_baseline"]["macro_F0.5"], 0.5)

            gt.write_text(
                "source1_entity_id\tmatched_entity_ids\n"
                "S1-a\tS2-a\nS1-a\t\n"
            )
            with self.assertRaisesRegex(ValueError, "repeated S1"):
                _scan_verified_dataset(root, root / "duplicate.sqlite")


if __name__ == "__main__":
    unittest.main()
