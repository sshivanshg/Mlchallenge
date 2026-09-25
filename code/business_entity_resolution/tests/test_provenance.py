"""Provenance guard unit tests (no competition fitting)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from data.provenance import (  # noqa: E402
    ProvenanceError,
    assert_not_synthetic_output_target,
    require_official_dataset,
    verify_official_dataset,
)


def _write_tsv(path: Path, header: list[str], rows: list[list[str]] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(header)]
    for row in rows or []:
        lines.append("\t".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class ProvenanceGuardTests(unittest.TestCase):
    def test_banned_path_token_fails(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_fixture_") as td:
            root = Path(td) / "student_resource" / "dataset"
            root.mkdir(parents=True)
            report = verify_official_dataset(root, require=("train",), check_hashes=False)
            self.assertFalse(report.ok)
            self.assertTrue(any("banned" in e.lower() for e in report.errors))

    def test_non_official_location_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "not_student_resource" / "dataset"
            root.mkdir(parents=True)
            report = verify_official_dataset(root, require=("train",), check_hashes=False)
            self.assertFalse(report.ok)
            self.assertTrue(any("official competition path" in e for e in report.errors))

    def test_header_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "student_resource" / "dataset"
            for name in ("train_source1", "train_source2", "train_source3"):
                _write_tsv(
                    root / "train" / f"{name}.tsv",
                    ["entity_id", "business_name", "WRONG", "country"],
                    [["S1-1", "A", "B", "US"]],
                )
            _write_tsv(
                root / "train" / "train_ground_truth.tsv",
                ["source1_entity_id", "matched_entity_ids"],
                [["S1-1", ""]],
            )
            report = verify_official_dataset(root, require=("train",), check_hashes=False)
            self.assertFalse(report.ok)
            self.assertTrue(any("header mismatch" in e for e in report.errors))

    def test_hash_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "student_resource" / "dataset"
            hdr = ["entity_id", "business_name", "business_address", "country"]
            for name in ("train_source1", "train_source2", "train_source3"):
                _write_tsv(root / "train" / f"{name}.tsv", hdr, [["S1-1", "A", "B", "US"]])
            _write_tsv(
                root / "train" / "train_ground_truth.tsv",
                ["source1_entity_id", "matched_entity_ids"],
                [["S1-1", ""]],
            )
            report = verify_official_dataset(root, require=("train",), check_hashes=True)
            self.assertFalse(report.ok)
            self.assertTrue(any("sha256 mismatch" in e for e in report.errors))

    def test_synthetic_writer_blocked_from_competition_root(self):
        target = REPO / "student_resource" / "dataset"
        with self.assertRaises(ProvenanceError):
            assert_not_synthetic_output_target(target)

    def test_synthetic_writer_requires_empty_tests_directory(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaisesRegex(ProvenanceError, "tests/ directory"):
                assert_not_synthetic_output_target(root / "data")
            allowed = root / "tests" / "fixture"
            assert_not_synthetic_output_target(allowed)
            (allowed / "train").mkdir(parents=True)
            (allowed / "train" / "train_source1.tsv").write_text("existing")
            with self.assertRaisesRegex(ProvenanceError, "overwrite"):
                assert_not_synthetic_output_target(allowed)

    def test_official_dataset_passes_when_present(self):
        root = REPO / "student_resource" / "dataset"
        if not (root / "train" / "train_source1.tsv").is_file():
            self.skipTest("official dataset not installed in this environment")
        report = require_official_dataset(root, require=("train", "test"), check_hashes=True)
        self.assertTrue(report.ok)
        self.assertEqual(report.manifest_version, "dataset-v1")
        self.assertGreaterEqual(len(report.checked_files), 7)


if __name__ == "__main__":
    unittest.main()
