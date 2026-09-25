"""Parity check: frozen_v4 vs research path (run_blocking_diag v4all + run_matcher_v4 features)."""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import pickle
import random
import sqlite3
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from evaluation.metric_ext import score_predictions
from frozen_v3b import policy as P3
from frozen_v4 import policy as P
from official_core import SOURCE_H, iter_tsv
from run_blocking_diag import ROUTE_TABLES, Retriever, s3_boundaries
from run_matcher_v4 import block_features as research_block_features
from run_official_bounded import GT_H

REPO = SRC.parents[2]
WD = REPO / "artifacts" / "official_index"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--writer-outputs", nargs="*", type=Path, default=[])
    args = p.parse_args()
    ds = REPO / "student_resource" / "dataset"
    fold_db = sqlite3.connect(WD / "official_70bc1d8a16c6_folds.sqlite")
    rows = list(fold_db.execute("SELECT id FROM fold WHERE fold='select' ORDER BY id"))
    random.Random(42).shuffle(rows)
    ids = [r[0] for r in rows[: args.n]]
    need = set(ids)
    s1 = {r["entity_id"]: r for r in iter_tsv(ds / "train" / "train_source1.tsv", SOURCE_H) if r["entity_id"] in need}
    with (WD / "matcher_v4all_m2.pkl").open("rb") as f:
        clf = pickle.load(f)["models"]["M2_allneg_extra_fit15k"]
    booster = clf.booster_

    db = sqlite3.connect(f"file:{WD / 'targets_index_v2.sqlite'}?mode=ro", uri=True)
    pdb = sqlite3.connect(f"file:{WD / 'train_pair_index_v3.sqlite'}?mode=ro", uri=True)
    adb = sqlite3.connect(f"file:{WD / 'train_aux_index_v4.sqlite'}?mode=ro", uri=True)
    n_targets = db.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
    b, pb, ab = s3_boundaries(db, ROUTE_TABLES), s3_boundaries(pdb, P.PAIR_TABLES), s3_boundaries(adb, P.AUX_TABLES)
    ref = Retriever(db, n_targets, "v4all", b, pdb, pb, adb, ab)
    frz = P.Retriever(db, pdb, adb, n_targets, b, pb, ab)

    cand_mm = feat_mm = dec_mm = 0
    max_dp = 0.0
    ref_pred, frz_pred = {}, {}
    for sid in ids:
        r = s1[sid]
        ranked, sc_r, rt_r, _ = ref.retrieve(r["business_name"], r["business_address"], r["country"])
        c_r = ranked[: P.CAP]
        c_f, sc_f, rt_f = frz.retrieve(r["business_name"], r["business_address"], r["country"])
        cand_mm += int(c_r != c_f)
        if not c_r:
            ref_pred[sid] = frz_pred[sid] = set()
            continue
        recs = P3.load_records(db, c_r)
        base, extra = research_block_features(P3.s1_view(r), c_r, sc_r, rt_r, recs)
        X_r = np.hstack([base, extra])
        X_f = P.block_features(P.s1_view(r), c_f, sc_f, rt_f, P.load_records(db, c_f))
        if X_r.shape != X_f.shape or not np.array_equal(X_r, X_f):
            feat_mm += 1
        p_r = clf.predict_proba(X_r)[:, 1]
        p_f = booster.predict(X_f, num_threads=1)
        if p_r.shape == p_f.shape:
            max_dp = max(max_dp, float(np.max(np.abs(p_r - p_f))))
        ref_pred[sid] = {t for t, q in zip(c_r, p_r) if q >= P.THRESHOLD}
        frz_pred[sid] = {t for t, q in zip(c_f, p_f) if q >= P.THRESHOLD}
        dec_mm += int(ref_pred[sid] != frz_pred[sid])

    gt = {i: set() for i in ids}
    for row in iter_tsv(ds / "train" / "train_ground_truth.tsv", GT_H):
        if row["source1_entity_id"] in need and row["matched_entity_ids"]:
            gt[row["source1_entity_id"]] = set(row["matched_entity_ids"].split(","))
    report = {"n_select_ids": len(ids), "candidate_list_mismatches": cand_mm, "feature_matrix_mismatches": feat_mm,
              "max_abs_prob_diff": max_dp, "decision_mismatches": dec_mm,
              "ref_macro_F0.5": score_predictions(gt, ref_pred)["macro_F0.5"],
              "frozen_macro_F0.5": score_predictions(gt, frz_pred)["macro_F0.5"], "writer_results": []}
    for out in args.writer_outputs:
        pred = {}
        with (out / "matching_results.tsv").open(encoding="utf-8") as f:
            next(f)
            for line in f:
                s, m = line.rstrip("\n").split("\t")
                pred[s] = set(m.split(",")) if m else set()
        report["writer_results"].append({"output": str(out), "ids_equal": set(pred) == need,
                                         "decision_mismatches_vs_frozen": sum(pred.get(s, set()) != frz_pred[s] for s in ids)})
    if len(args.writer_outputs) >= 2:
        blobs = [((o / "matching_results.tsv").read_bytes(), (o / "candidate_pairs.tsv").read_bytes()) for o in args.writer_outputs]
        report["writer_outputs_byte_identical"] = all(x == blobs[0] for x in blobs[1:])
    print(json.dumps(report, indent=2))
    ok = cand_mm == 0 and feat_mm == 0 and dec_mm == 0 and max_dp < 1e-9
    ok = ok and all(w["ids_equal"] and w["decision_mismatches_vs_frozen"] == 0 for w in report["writer_results"])
    ok = ok and report.get("writer_outputs_byte_identical", True)
    if not ok:
        raise SystemExit("VERIFY FAILED")
    print("VERIFY PASS")


if __name__ == "__main__":
    main()
