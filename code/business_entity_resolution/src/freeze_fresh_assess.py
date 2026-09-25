"""Freeze a fresh assessment slice disjoint from the already-inspected assess IDs.

The Stage-4/4e assessment sample (3,000 IDs: seed 42, drawn after select and fit)
has influenced development and is relabelled 'assess_dev_used'. This picks new
assess-partition S1s whose positive components share no target with that sample.
Ground truth is read only to build components; no label statistics are written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from data.provenance import load_manifest, require_official_dataset
from official_core import iter_tsv
from run_official_bounded import GT_H, UnionFind

REPO = SRC.parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=REPO / "student_resource" / "dataset")
    p.add_argument("--work-dir", type=Path, default=REPO / "artifacts" / "official_index")
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260925)
    args = p.parse_args()
    prov = require_official_dataset(args.dataset_root, require=("train",), check_hashes=True)
    gt_hash = load_manifest()["files"]["train/train_ground_truth.tsv"]["sha256"]
    split_version = f"official_{gt_hash[:12]}"
    out = REPO / "experiments" / "splits" / f"{split_version}_assess_fresh_v1.json"
    if out.exists():
        raise SystemExit(f"already frozen: {out}")

    fold_db = sqlite3.connect(args.work_dir / f"{split_version}_folds.sqlite")
    rng = random.Random(42)

    def sample(fold, n):
        rows = list(fold_db.execute("SELECT id FROM fold WHERE fold=? ORDER BY id", (fold,)))
        rng.shuffle(rows)
        return [r[0] for r in rows[:n]]

    sample("select", 5000)
    sample("fit", 8000)
    used = set(sample("assess", 3000))
    assess_all = sorted(r[0] for r in fold_db.execute("SELECT id FROM fold WHERE fold='assess'"))
    fold_db.close()
    assess_set = set(assess_all)

    uf = UnionFind()
    owners = defaultdict(list)
    for row in iter_tsv(args.dataset_root / "train" / "train_ground_truth.tsv", GT_H):
        sid = row["source1_entity_id"]
        if sid not in assess_set:
            continue
        uf.add(sid)
        for t in (row["matched_entity_ids"].split(",") if row["matched_entity_ids"] else []):
            owners[t].append(sid)
    for o in owners.values():
        for x in o[1:]:
            uf.union(o[0], x)
    used_roots = {uf.find(s) for s in used}
    comps = defaultdict(list)
    for s in assess_all:
        r = uf.find(s)
        if r not in used_roots:
            comps[r].append(s)
    comp_list = [sorted(v) for _, v in sorted(comps.items(), key=lambda kv: kv[1][0])]
    random.Random(args.seed).shuffle(comp_list)
    fresh = []
    for c in comp_list:
        if len(fresh) >= args.n:
            break
        fresh.extend(c)
    fresh = sorted(fresh)
    digest = hashlib.sha256("\n".join(fresh).encode()).hexdigest()
    out.write_text(json.dumps({
        "split_version": split_version,
        "provenance": f"official_challenge_dataset:{prov.manifest_version}",
        "role": "assess_fresh_v1: untouched final assessment; evaluate each frozen config once",
        "excluded": "3,000 assess IDs used in Stage 4/4e (relabelled assess_dev_used) and their positive components",
        "seed": args.seed,
        "n": len(fresh),
        "sha256_of_sorted_ids": digest,
        "ids": fresh,
    }, indent=1) + "\n")
    print(json.dumps({"out": str(out), "n": len(fresh), "sha256": digest, "overlap_with_used": len(set(fresh) & used)}))


if __name__ == "__main__":
    main()
