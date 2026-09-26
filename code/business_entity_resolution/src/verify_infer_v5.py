"""Parity check: frozen_v5 vs the research path used for selection (run_matcher_v5)."""

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

from frozen_v4 import policy as P4
from frozen_v5 import policy as P5
from official_core import SOURCE_H, iter_tsv
from run_matcher_v5 import extra2 as research_extra2

REPO = SRC.parents[2]
WD = REPO / "artifacts" / "official_index"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--writer-outputs", nargs="*", type=Path, default=[])
    args = p.parse_args()
    sel = json.loads((SRC / "frozen_v5" / "selected.json").read_text())
    ds = REPO / "student_resource" / "dataset"
    fold_db = sqlite3.connect(WD / "official_70bc1d8a16c6_folds.sqlite")
    rows = list(fold_db.execute("SELECT id FROM fold WHERE fold='select' ORDER BY id"))
    random.Random(42).shuffle(rows)
    ids = [r[0] for r in rows[: args.n]]
    need = set(ids)
    s1 = {r["entity_id"]: r for r in iter_tsv(ds / "train" / "train_source1.tsv", SOURCE_H) if r["entity_id"] in need}
    with (REPO / sel["model_path"]).open("rb") as f:
        model = pickle.load(f)[sel["model_key"]]
    booster = model.booster_
    db = sqlite3.connect(f"file:{WD / 'targets_index_v2.sqlite'}?mode=ro", uri=True)
    pdb = sqlite3.connect(f"file:{WD / 'train_pair_index_v3.sqlite'}?mode=ro", uri=True)
    adb = sqlite3.connect(f"file:{WD / 'train_aux_index_v4.sqlite'}?mode=ro", uri=True)
    n_targets = db.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
    ret = P4.Retriever(db, pdb, adb, n_targets, P4.s3_boundaries(db, P4.ROUTE_TABLES),
                       P4.s3_boundaries(pdb, P4.PAIR_TABLES), P4.s3_boundaries(adb, P4.AUX_TABLES))
    feat_mm = dec_mm = 0
    max_dp = 0.0
    frz_pred = {}
    for sid in ids:
        r = s1[sid]
        c, sc, rt = ret.retrieve(r["business_name"], r["business_address"], r["country"])
        if not c:
            frz_pred[sid] = set()
            continue
        recs = P4.load_records(db, c)
        a = P4.s1_view(r)
        base = P4.block_features(a, c, sc, rt, recs)
        X_r = np.hstack([base, research_extra2(a, recs, c)]) if sel["n_features"] == 43 else base
        X_f = P5.features(a, c, sc, rt, recs) if sel["n_features"] == 43 else P5.block_features(a, c, sc, rt, recs)
        feat_mm += int(not np.array_equal(X_r, X_f))
        p_r = model.predict_proba(X_r)[:, 1]
        p_f = booster.predict(X_f, num_threads=1)
        max_dp = max(max_dp, float(np.max(np.abs(p_r - p_f))))

        def dec(pr):
            mx = float(pr.max())
            return {t for t, q in zip(c, pr) if q >= sel["threshold"] and mx >= sel["gate"] and q >= sel["rel"] * mx}

        frz_pred[sid] = dec(p_f)
        dec_mm += int(dec(p_r) != frz_pred[sid])
    report = {"n": len(ids), "feature_matrix_mismatches": feat_mm, "max_abs_prob_diff": max_dp,
              "decision_mismatches": dec_mm, "writer_results": []}
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
    ok = feat_mm == 0 and dec_mm == 0 and max_dp < 1e-9
    ok = ok and all(w["ids_equal"] and w["decision_mismatches_vs_frozen"] == 0 for w in report["writer_results"])
    ok = ok and report.get("writer_outputs_byte_identical", True)
    if not ok:
        raise SystemExit("VERIFY FAILED")
    print("VERIFY PASS")


if __name__ == "__main__":
    main()
