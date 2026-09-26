"""Select-fold combine screen: v5b addrtok retrieval + transferred v5 / H4 matchers.

Builds 43-d features on addrtok candidates for the fixed 5,000 select S1s, scores
with frozen v5 and H4 boosters (no retraining), retunes thresholds for macro F0.5.
Does not open assess_fresh_v4.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import pickle
import random
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from evaluation.metric_ext import score_predictions
from evaluation.skill_metric import entity_f05
from frozen_v6 import policy as P
from official_core import SOURCE_H, iter_tsv
from run_official_bounded import GT_H

REPO = SRC.parents[2]
WD = REPO / "artifacts" / "official_index"
REP = REPO / "reports" / "official" / "official_70bc1d8a16c6" / "matcher_exp3"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_booster(path: Path, key: str):
    with path.open("rb") as f:
        bundle = pickle.load(f)
    if key in bundle:
        model = bundle[key]
    elif "F1N1" in bundle:
        model = bundle["F1N1"]
    else:
        model = bundle["model"]
    return model.booster_ if hasattr(model, "booster_") else model, bundle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    REP.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    fold = sqlite3.connect(f"file:{WD / 'official_70bc1d8a16c6_folds.sqlite'}?mode=ro", uri=True)
    rows = [r[0] for r in fold.execute("SELECT id FROM fold WHERE fold='select' ORDER BY id")]
    random.Random(args.seed).shuffle(rows)
    ids = rows[: args.n_eval]
    need = set(ids)
    fold.close()

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
    ret = P.Retriever(
        db, pair, aux, v5b, n_targets,
        P.s3_boundaries(db, P.ROUTE_TABLES),
        P.s3_boundaries(pair, P.PAIR_TABLES),
        P.s3_boundaries(aux, P.AUX_TABLES),
        P.s3_boundaries(v5b, P.V5B_TABLES),
    )

    print(f"building features for {len(ids)} select S1s...", flush=True)
    X_rows, owners, tids, oracle_pred = [], [], [], {s: set() for s in ids}
    for i, sid in enumerate(ids):
        r = s1[sid]
        c, sc, rt = ret.retrieve(r["business_name"], r["business_address"], r["country"])
        if not c:
            continue
        recs = P.load_records(db, c)
        a = P.s1_view(r)
        X_rows.append(P.features(a, c, sc, rt, recs))
        owners.extend([sid] * len(c))
        tids.extend(c)
        oracle_pred[sid] = gt[sid] & set(c)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(ids)} {time.time()-t0:.0f}s", flush=True)
    X = np.vstack(X_rows)
    owners = np.asarray(owners)
    tids = np.asarray(tids)
    assert X.shape[1] == 43, X.shape
    m_oracle = score_predictions(gt, oracle_pred)
    print(f"features {X.shape} oracle_F0.5={m_oracle['macro_F0.5']:.4f} {time.time()-t0:.0f}s", flush=True)

    v5_path = REPO / "artifacts" / "official_index" / "matcher_v5.pkl"
    h4_path = REPO / "artifacts" / "official_index" / "matcher_v5_capacity_l127_t600.pkl"
    v5_booster, v5_bundle = load_booster(v5_path, "F1N1")
    h4_booster, h4_bundle = load_booster(h4_path, "model")
    p_v5 = v5_booster.predict(X, num_threads=1)
    p_h4 = h4_booster.predict(X, num_threads=1)

    grid = np.round(np.concatenate([np.linspace(0.5, 0.9, 9), np.linspace(0.91, 0.995, 18)]), 4)

    def decide(probs, thr):
        pred = {s: set() for s in ids}
        for s, t, q in zip(owners, tids, probs):
            if q >= thr:
                pred[s].add(t)
        return pred

    def tune(probs, label, default_thr):
        best_thr, best_m = default_thr, score_predictions(gt, decide(probs, default_thr))
        for t in grid:
            m = score_predictions(gt, decide(probs, float(t)))
            if m["macro_F0.5"] > best_m["macro_F0.5"]:
                best_thr, best_m = float(t), m
        return {"label": label, "threshold": best_thr, "metrics": best_m,
                "default_threshold": default_thr,
                "default_metrics": score_predictions(gt, decide(probs, default_thr))}

    v5_sel = json.loads((SRC / "frozen_v5" / "selected.json").read_text())
    res_v5 = tune(p_v5, "v5_matcher_on_addrtok", v5_sel["threshold"])
    res_h4 = tune(p_h4, "h4_capacity_on_addrtok", float(h4_bundle.get("threshold", 0.7)))
    print(json.dumps({"v5": {k: res_v5[k] for k in res_v5 if k != "metrics"} | {"macro_F0.5": res_v5["metrics"]["macro_F0.5"]},
                      "h4": {k: res_h4[k] for k in res_h4 if k != "metrics"} | {"macro_F0.5": res_h4["metrics"]["macro_F0.5"]}}, indent=2), flush=True)

    # paired deltas vs each other on retuned thresholds
    pred_v5 = decide(p_v5, res_v5["threshold"])
    pred_h4 = decide(p_h4, res_h4["threshold"])
    d = np.asarray([entity_f05(gt[s], pred_h4[s]) - entity_f05(gt[s], pred_v5[s]) for s in ids])
    rng = np.random.default_rng(0)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
    paired = {
        "mean_delta_h4_minus_v5": float(d.mean()),
        "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "n_better": int((d > 0).sum()),
        "n_worse": int((d < 0).sum()),
        "n_tie": int((d == 0).sum()),
    }

    h4_sha = sha256_file(h4_path)
    out = {
        "hypothesis": "addrtok retrieval (oracle 0.9596) + H4 capacity matcher beats v5-on-addrtok and unlocks >0.95 ceiling",
        "fold": f"select_bounded_n{args.n_eval}",
        "seed": args.seed,
        "n_target_universe": n_targets,
        "feature_dim": 43,
        "candidate_oracle": m_oracle,
        "v5_on_addrtok": res_v5,
        "h4_on_addrtok": res_h4,
        "paired_h4_vs_v5_on_addrtok": paired,
        "h4_model_sha256": h4_sha,
        "runtime_sec": time.time() - t0,
        "assess_fresh_v4": "frozen, not opened",
    }
    outp = REP / "v6_addrtok_h4_select.json"
    outp.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", outp)

    # update frozen_v6/selected.json with measured threshold + sha
    sel = {
        "model_path": "artifacts/official_index/matcher_v5_capacity_l127_t600.pkl",
        "model_sha256": h4_sha,
        "model_key": "model",
        "n_features": 43,
        "threshold": res_h4["threshold"],
        "gate": 0.0,
        "rel": 0.0,
        "selection_macro_F0.5": res_h4["metrics"]["macro_F0.5"],
        "retrieval": "v5b_addrtok_only",
        "ngram_rejected": True,
        "oracle_macro_F0.5": m_oracle["macro_F0.5"],
        "selected_from": str(outp.relative_to(REPO)),
        "v5_on_addrtok_macro_F0.5": res_v5["metrics"]["macro_F0.5"],
        "paired_delta_vs_v5_on_addrtok": paired["mean_delta_h4_minus_v5"],
    }
    (SRC / "frozen_v6" / "selected.json").write_text(json.dumps(sel, indent=2) + "\n")
    print(json.dumps(sel, indent=2))


if __name__ == "__main__":
    main()
