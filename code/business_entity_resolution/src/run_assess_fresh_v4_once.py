"""One-shot assessment of frozen v5 vs v6 on assess_fresh_v4 (never reopen).

Scores both policies on the same 5,000 assess IDs. Does not retune thresholds.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import pickle
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from evaluation.metric_ext import score_predictions
from frozen_v5 import policy as P5
from frozen_v6 import policy as P6
from official_core import SOURCE_H, iter_tsv
from run_official_bounded import GT_H

REPO = SRC.parents[2]
WD = REPO / "artifacts" / "official_index"
REP = REPO / "reports" / "official" / "official_70bc1d8a16c6"
IDS = REPO / "experiments" / "splits" / "official_70bc1d8a16c6_assess_fresh_v4.json"


def load_booster(path, key):
    with path.open("rb") as f:
        b = pickle.load(f)
    m = b[key]
    return m.booster_ if hasattr(m, "booster_") else m, b


def score_policy(label, ids, s1, gt, retrieve_fn, features_fn, booster, thr):
    t0 = time.time()
    pred = {s: set() for s in ids}
    cands = {s: set() for s in ids}
    for i, sid in enumerate(ids):
        r = s1[sid]
        c, sc, rt = retrieve_fn(r)
        cands[sid] = set(c)
        if not c:
            continue
        # load_records from the shared db inside retrieve closures
        X = features_fn(r, c, sc, rt)
        if X is None:
            continue
        pr = booster.predict(X, num_threads=1)
        for t, q in zip(c, pr):
            if q >= thr:
                pred[sid].add(t)
        if (i + 1) % 500 == 0:
            print(f"  [{label}] {i+1}/{len(ids)} {time.time()-t0:.0f}s", flush=True)
    m = score_predictions(gt, pred)
    oracle = score_predictions(gt, {s: gt[s] & cands[s] for s in ids})
    m["candidate_oracle_macro_F0.5"] = oracle["macro_F0.5"]
    true_total = sum(len(gt[s]) for s in ids)
    true_hit = sum(len(gt[s] & cands[s]) for s in ids)
    m["candidate_recall"] = true_hit / true_total if true_total else None
    m["runtime_sec"] = time.time() - t0
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-file", type=Path, default=IDS)
    args = ap.parse_args()
    out_path = REP / "assess_fresh_v4_once.json"
    if out_path.exists():
        raise SystemExit(f"already assessed once: {out_path}")

    meta = json.loads(args.ids_file.read_text())
    ids = meta["ids"]
    need = set(ids)
    print(f"assess_fresh_v4 n={len(ids)} sha={meta['sha256_of_sorted_ids'][:16]}...", flush=True)

    s1 = {}
    root = REPO / "student_resource" / "dataset" / "train"
    for row in iter_tsv(root / "train_source1.tsv", SOURCE_H):
        if row["entity_id"] in need:
            s1[row["entity_id"]] = row
    gt = {i: set() for i in ids}
    for row in iter_tsv(root / "train_ground_truth.tsv", GT_H):
        sid = row["source1_entity_id"]
        if sid in need and row["matched_entity_ids"]:
            gt[sid] = set(row["matched_entity_ids"].split(","))

    db = sqlite3.connect(f"file:{WD / 'targets_index_v2.sqlite'}?mode=ro", uri=True)
    pair = sqlite3.connect(f"file:{WD / 'train_pair_index_v3.sqlite'}?mode=ro", uri=True)
    aux = sqlite3.connect(f"file:{WD / 'train_aux_index_v4.sqlite'}?mode=ro", uri=True)
    v5b = sqlite3.connect(f"file:{WD / 'train_aux_index_v5b.sqlite'}?mode=ro", uri=True)
    n_targets = db.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
    bounds = P5.s3_boundaries(db, P5.ROUTE_TABLES)
    pair_bounds = P5.s3_boundaries(pair, P5.PAIR_TABLES)
    aux_bounds = P5.s3_boundaries(aux, P5.AUX_TABLES)
    v5b_bounds = P6.s3_boundaries(v5b, P6.V5B_TABLES)

    ret5 = P5.Retriever(db, pair, aux, n_targets, bounds, pair_bounds, aux_bounds)
    ret6 = P6.Retriever(db, pair, aux, v5b, n_targets, bounds, pair_bounds, aux_bounds, v5b_bounds)

    sel5 = json.loads((SRC / "frozen_v5" / "selected.json").read_text())
    sel6 = json.loads((SRC / "frozen_v6" / "selected.json").read_text())
    b5, _ = load_booster(REPO / sel5["model_path"], sel5["model_key"])
    b6, _ = load_booster(REPO / sel6["model_path"], sel6["model_key"])

    def feat5(r, c, sc, rt):
        recs = P5.load_records(db, c)
        return P5.features(P5.s1_view(r), c, sc, rt, recs)

    def feat6(r, c, sc, rt):
        recs = P6.load_records(db, c)
        return P6.features(P6.s1_view(r), c, sc, rt, recs)

    m5 = score_policy(
        "v5", ids, s1, gt,
        lambda r: ret5.retrieve(r["business_name"], r["business_address"], r["country"]),
        feat5, b5, sel5["threshold"],
    )
    print(json.dumps({"v5": m5["macro_F0.5"], "oracle": m5["candidate_oracle_macro_F0.5"]}, indent=2), flush=True)

    m6 = score_policy(
        "v6", ids, s1, gt,
        lambda r: ret6.retrieve(r["business_name"], r["business_address"], r["country"]),
        feat6, b6, sel6["threshold"],
    )
    print(json.dumps({"v6": m6["macro_F0.5"], "oracle": m6["candidate_oracle_macro_F0.5"]}, indent=2), flush=True)

    out = {
        "ids_file": str(args.ids_file.relative_to(REPO)),
        "ids_sha256": meta["sha256_of_sorted_ids"],
        "n": len(ids),
        "opened_once_utc": time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()),
        "results": {
            "v5_v4all_43f": {
                "policy": sel5,
                **m5,
            },
            "v6_addrtok_h4": {
                "policy": sel6,
                **m6,
            },
        },
        "delta_v6_minus_v5": m6["macro_F0.5"] - m5["macro_F0.5"],
        "note": "one-shot only; do not reopen assess_fresh_v4",
    }
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", out_path)
    print(json.dumps({"v5": m5["macro_F0.5"], "v6": m6["macro_F0.5"], "delta": out["delta_v6_minus_v5"],
                      "oracle_v5": m5["candidate_oracle_macro_F0.5"], "oracle_v6": m6["candidate_oracle_macro_F0.5"]}, indent=2))


if __name__ == "__main__":
    main()
