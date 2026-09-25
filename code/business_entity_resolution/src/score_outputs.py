"""Score writer outputs (matching + candidate TSVs) on frozen train-fold IDs.

Reports exact macro F0.5, precision, recall, singleton accuracy, candidate recall,
and candidate-oracle macro F0.5, overall and by country and source.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from evaluation.metric_ext import score_predictions
from official_core import SOURCE_H, iter_tsv
from run_official_bounded import GT_H

REPO = SRC.parents[2]


def read_lists(path: Path) -> dict[str, set[str]]:
    out = {}
    with path.open(encoding="utf-8") as f:
        next(f)
        for line in f:
            s, ids = line.rstrip("\n").split("\t")
            out[s] = set(ids.split(",")) if ids else set()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ids", type=Path, required=True)
    p.add_argument("--outputs", nargs="+", type=Path, required=True, help="writer output dirs")
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    ds = REPO / "student_resource" / "dataset"
    spec = json.loads(args.ids.read_text())
    ids = spec["ids"]
    need = set(ids)
    gt = {i: set() for i in ids}
    for row in iter_tsv(ds / "train" / "train_ground_truth.tsv", GT_H):
        if row["source1_entity_id"] in need and row["matched_entity_ids"]:
            gt[row["source1_entity_id"]] = set(row["matched_entity_ids"].split(","))
    country = {r["entity_id"]: r["country"] for r in iter_tsv(ds / "train" / "train_source1.tsv", SOURCE_H) if r["entity_id"] in need}
    report = {"ids_file": str(args.ids), "ids_sha256": spec.get("sha256_of_sorted_ids"), "n": len(ids), "results": {}}
    for out in args.outputs:
        pred = read_lists(out / "matching_results.tsv")
        cand = read_lists(out / "candidate_pairs.tsv")
        if set(pred) != need or set(cand) != need:
            raise SystemExit(f"{out}: S1 coverage differs from ids file")
        if any(not pred[s] <= cand[s] for s in ids):
            raise SystemExit(f"{out}: match outside candidates")
        m = score_predictions(gt, pred)
        oracle = score_predictions(gt, {s: gt[s] & cand[s] for s in ids})["macro_F0.5"]
        tot = sum(len(gt[s]) for s in ids)
        rec = sum(len(gt[s] & cand[s]) for s in ids) / tot
        by_c = defaultdict(list)
        for s in ids:
            by_c[country[s]].append(s)
        slices = {}
        for c, ss in sorted(by_c.items()):
            mc = score_predictions({s: gt[s] for s in ss}, {s: pred[s] for s in ss})
            slices[c] = {"n": len(ss), "macro_F0.5": mc["macro_F0.5"], "precision": mc["precision"],
                         "recall": mc["recall"], "singleton_accuracy": mc["singleton_accuracy"]}
        src = {}
        for pre in ("S2-", "S3-"):
            t = sum(1 for s in ids for x in gt[s] if x.startswith(pre))
            h = sum(1 for s in ids for x in gt[s] & cand[s] if x.startswith(pre))
            tp = sum(1 for s in ids for x in gt[s] & pred[s] if x.startswith(pre))
            fp = sum(1 for s in ids for x in pred[s] - gt[s] if x.startswith(pre))
            src[pre[:2]] = {"candidate_recall": h / t if t else None, "recall": tp / t if t else None,
                            "precision": tp / (tp + fp) if tp + fp else None}
        report["results"][str(out)] = {**m, "candidate_recall": rec, "candidate_oracle_macro_F0.5": oracle,
                                       "by_country": slices, "by_source": src}
        print(f"{out}: F0.5={m['macro_F0.5']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} "
              f"sing={m['singleton_accuracy']:.4f} cand_recall={rec:.4f} oracle={oracle:.4f}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
