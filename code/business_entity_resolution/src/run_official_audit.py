"""Streaming first measurement on the verified competition dump.

This intentionally stops before retrieval or model fitting. It reads only local
challenge TSVs and keeps S1 ID integrity state in a temporary SQLite database.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import tempfile
from collections import Counter
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.provenance import load_manifest, require_official_dataset

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


def _rows(path: Path, columns: list[str]):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != columns:
            raise ValueError(f"Unexpected columns in {path}: {reader.fieldnames}")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed TSV row in {path} at line {reader.line_num}")
            yield row


def _scan_verified_dataset(root: Path, db_path: Path) -> dict:
    """Internal scanner; only the public entrypoint may invoke it on real data."""
    db = sqlite3.connect(db_path)
    try:
        db.execute("CREATE TABLE s1 (id TEXT PRIMARY KEY, seen INTEGER NOT NULL DEFAULT 0) WITHOUT ROWID")
        sources = {}
        for source in (1, 2, 3):
            counts = Counter()
            countries = Counter()
            path = root / "train" / f"train_source{source}.tsv"
            batch = []
            for row in _rows(path, SOURCE_COLUMNS):
                entity_id = row["entity_id"]
                if not entity_id:
                    raise ValueError(f"Empty ID in {path}")
                counts["records"] += 1
                countries[row["country"] or "<missing>"] += 1
                for field in ("business_name", "business_address", "country"):
                    counts[f"missing_{field}"] += not row[field].strip()
                if source == 1:
                    batch.append((entity_id,))
                    if len(batch) >= 10000:
                        with db:
                            db.executemany("INSERT INTO s1 (id) VALUES (?)", batch)
                        batch.clear()
            if batch:
                with db:
                    db.executemany("INSERT INTO s1 (id) VALUES (?)", batch)
            sources[f"s{source}"] = {**counts, "countries": dict(countries)}

        matches = Counter()
        path = root / "train" / "train_ground_truth.tsv"
        for row in _rows(path, TRUTH_COLUMNS):
            sid = row["source1_entity_id"]
            if not sid:
                raise ValueError(f"Empty S1 ID in {path}")
            changed = db.execute("UPDATE s1 SET seen=1 WHERE id=? AND seen=0", (sid,)).rowcount
            if changed != 1:
                raise ValueError(f"Unknown or repeated S1 ground-truth ID: {sid}")
            raw = row["matched_entity_ids"]
            ids = raw.split(",") if raw else []
            if any(not item for item in ids) or len(set(ids)) != len(ids):
                raise ValueError(f"Malformed or duplicate target ID for S1 {sid}")
            matches["truth_rows"] += 1
            if matches["truth_rows"] % 10000 == 0:
                db.commit()
            matches["links"] += len(ids)
            matches[f"cardinality_{len(ids)}"] += 1
            matches["s2_links"] += sum(item.startswith("S2-") for item in ids)
            matches["s3_links"] += sum(item.startswith("S3-") for item in ids)
            if any(not (item.startswith("S2-") or item.startswith("S3-")) for item in ids):
                raise ValueError(f"Unexpected target ID namespace for S1 {sid}")
        db.commit()
        unseen = db.execute("SELECT COUNT(*) FROM s1 WHERE seen=0").fetchone()[0]
        if unseen:
            raise ValueError(f"Ground truth omitted {unseen} S1 records")
        n_s1 = sources["s1"]["records"]
        if n_s1 != matches["truth_rows"]:
            raise ValueError("Ground-truth and S1 record counts disagree")
        if n_s1 == 0:
            raise ValueError("Empty S1 dataset")
        singleton = matches["cardinality_0"]
        return {
            "sources": sources,
            "ground_truth": {
                **matches,
                "singleton_rate": singleton / n_s1,
                "match_count_histogram": {
                    key.removeprefix("cardinality_"): value
                    for key, value in matches.items() if key.startswith("cardinality_")
                },
            },
            "empty_baseline": {
                "macro_F0.5": singleton / n_s1,
                "singleton_accuracy": 1.0 if singleton else None,
                "precision": None,
                "recall": 0.0 if matches["links"] else None,
            },
            "limitations": [
                "No blocking, matcher, threshold, or test-label metrics were measured.",
                "Target-ID existence and S2/S3 ID uniqueness were not audited here.",
            ],
        }
    finally:
        db.close()


def run(dataset_root: Path, output: Path, work_dir: Path | None = None) -> dict:
    dataset_root = dataset_root.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite audit: {output}")
    report = require_official_dataset(dataset_root, require=("train", "test"), check_hashes=True)
    manifest = load_manifest()
    with tempfile.TemporaryDirectory(prefix="official_audit_", dir=work_dir) as tmp:
        result = _scan_verified_dataset(dataset_root, Path(tmp) / "ids.sqlite")
    result["provenance"] = {
        "version": report.manifest_version,
        "verified_files": {name: manifest["files"][name]["sha256"] for name in report.checked_files},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return result


def main() -> None:
    repo = SRC.parents[2]
    parser = argparse.ArgumentParser(description="Audit the verified official dump with bounded RAM")
    parser.add_argument("--dataset-root", type=Path, default=repo / "student_resource" / "dataset")
    parser.add_argument("--output", type=Path, default=repo / "reports" / "eda" / "official_audit.json")
    parser.add_argument("--work-dir", type=Path, help="Directory with free disk space for temporary S1 ID index")
    args = parser.parse_args()
    result = run(args.dataset_root, args.output, args.work_dir)
    print(json.dumps({"output": str(args.output), "ground_truth": result["ground_truth"], "empty_baseline": result["empty_baseline"]}, indent=2))


if __name__ == "__main__":
    main()
