"""Check the frozen policy against the research reference path on selection IDs.

Reference = run_blocking_diag.Retriever(v3b) + official_core features + the saved
LGBMClassifier.predict_proba, exactly as used in run_matcher_v3. Compares
candidate lists, feature vectors, probabilities, and decisions, then scores
macro F0.5 on the selection IDs (a development set, already used for tuning).
Optionally compares writer outputs (e.g. 1 vs N workers) for byte equality.
"""

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
from frozen_v3b import policy as P
from official_core import SOURCE_H, feat_vec, iter_tsv, load_records, norm_addr, norm_name
from run_blocking_diag import ROUTE_TABLES, Retriever, s3_boundaries
from run_official_bounded import GT_H

REPO = SRC.parents[2]
WD = REPO / "artifacts" / "official_index"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--write-ids", type=Path, default=None)
    p.add_argument("--writer-outputs", nargs="*", type=Path, default=[])
    args = p.parse_args()
    ds = REPO / "student_resource" / "dataset"

    fold_db = sqlite3.connect(WD / "official_70bc1d8a16c6_folds.sqlite")
    rows = list(fold_db.execute("SELECT id FROM fold WHERE fold='select' ORDER BY id"))
    random.Random(42).shuffle(rows)
    ids = [r[0] for r in rows[: args.n]]
    need = set(ids)
    if args.write_ids:
        args.write_ids.write_text(json.dumps({"ids": sorted(ids)}) + "\n")
    s1 = {r["entity_id"]: r for r in iter_tsv(ds / "train" / "train_source1.tsv", SOURCE_H) if r["entity_id"] in need}

    with (WD / "matcher_v3b_cap300.pkl").open("rb") as f:
        bundle = pickle.load(f)
    clf = bundle["lightgbm"]
    booster = clf.booster_

    db = sqlite3.connect(f"file:{WD / 'targets_index_v2.sqlite'}?mode=ro", uri=True)
    pdb = sqlite3.connect(f"file:{WD / 'train_pair_index_v3.sqlite'}?mode=ro", uri=True)
    n_targets = db.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
    b, pb = s3_boundaries(db, ROUTE_TABLES), s3_boundaries(pdb, ("by_pair", "by_toknum"))
    ref = Retriever(db, n_targets, "v3b", b, pdb, pb)
    frz = P.Retriever(db, pdb, n_targets, P.s3_boundaries(db, P.ROUTE_TABLES), P.s3_boundaries(pdb, P.PAIR_TABLES))
    assert b == frz.bounds and pb == frz.pair_bounds

    cand_mismatch = feat_mismatch = dec_mismatch = 0
    max_dp = 0.0
    ref_pred, frz_pred = {}, {}
    for sid in ids:
        r = s1[sid]
        c_ref = ref.retrieve(r["business_name"], r["business_address"], r["country"])[0][: P.CAP]
        c_frz = frz.candidates(r["business_name"], r["business_address"], r["country"])
        cand_mismatch += int(c_ref != c_frz)
        a_ref = {**r, "name_norm": norm_name(r["business_name"]), "addr_norm": norm_addr(r["business_address"])}
        a_frz = P.s1_view(r)
        rec_ref = load_records(db, c_ref)
        rec_frz = P.load_records(db, c_frz)
        if not c_ref:
            ref_pred[sid] = frz_pred[sid] = set()
            continue
        F_ref = np.stack([feat_vec(a_ref, rec_ref[t]) for t in c_ref])
        F_frz = np.stack([P.feat_vec(a_frz, rec_frz[t]) for t in c_frz])
        if F_ref.shape != F_frz.shape or not np.array_equal(F_ref, F_frz):
            feat_mismatch += 1
        p_ref = clf.predict_proba(F_ref)[:, 1]
        p_frz = booster.predict(F_frz, num_threads=1)
        if p_ref.shape == p_frz.shape:
            max_dp = max(max_dp, float(np.max(np.abs(p_ref - p_frz))))
        ref_pred[sid] = {t for t, q in zip(c_ref, p_ref) if q >= P.THRESHOLD}
        frz_pred[sid] = {t for t, q in zip(c_frz, p_frz) if q >= P.THRESHOLD}
        dec_mismatch += int(ref_pred[sid] != frz_pred[sid])

    gt = {i: set() for i in ids}
    for row in iter_tsv(ds / "train" / "train_ground_truth.tsv", GT_H):
        if row["source1_entity_id"] in need and row["matched_entity_ids"]:
            gt[row["source1_entity_id"]] = set(row["matched_entity_ids"].split(","))
    report = {
        "n_select_ids": len(ids),
        "candidate_list_mismatches": cand_mismatch,
        "feature_matrix_mismatches": feat_mismatch,
        "max_abs_prob_diff": max_dp,
        "decision_mismatches": dec_mismatch,
        "ref_macro_F0.5": score_predictions(gt, ref_pred)["macro_F0.5"],
        "frozen_macro_F0.5": score_predictions(gt, frz_pred)["macro_F0.5"],
    }
    writer_results = []
    for out in args.writer_outputs:
        pred = {}
        with (out / "matching_results.tsv").open(encoding="utf-8") as f:
            next(f)
            for line in f:
                s, m = line.rstrip("\n").split("\t")
                pred[s] = set(m.split(",")) if m else set()
        writer_results.append({
            "output": str(out),
            "ids_equal": set(pred) == need,
            "decision_mismatches_vs_frozen": sum(pred.get(s, set()) != frz_pred[s] for s in ids),
            "macro_F0.5": score_predictions(gt, {s: pred.get(s, set()) for s in ids})["macro_F0.5"],
        })
    if len(args.writer_outputs) >= 2:
        blobs = [((o / "matching_results.tsv").read_bytes(), (o / "candidate_pairs.tsv").read_bytes()) for o in args.writer_outputs]
        report["writer_outputs_byte_identical"] = all(x == blobs[0] for x in blobs[1:])
    report["writer_results"] = writer_results
    print(json.dumps(report, indent=2))
    ok = cand_mismatch == 0 and feat_mismatch == 0 and dec_mismatch == 0 and max_dp < 1e-9
    ok = ok and all(w["ids_equal"] and w["decision_mismatches_vs_frozen"] == 0 for w in writer_results)
    ok = ok and report.get("writer_outputs_byte_identical", True)
    if not ok:
        raise SystemExit("VERIFY FAILED")
    print("VERIFY PASS")


if __name__ == "__main__":
    main()
