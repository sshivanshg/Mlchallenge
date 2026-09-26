"""H4: higher-capacity LightGBM on the cached v5 features (same fit/select S1s, 43 features).

Targets the largest matcher loss under v5: retrieved true links rejected (median
probability 0.34). Paired per-S1 comparison against the frozen v5 model on the
selection fold. Low-priority, single-threaded by default to avoid slowing inference.
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
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from evaluation.metric_ext import score_predictions
from evaluation.skill_metric import entity_f05
from official_core import iter_tsv
from run_official_bounded import GT_H

REPO = SRC.parents[2]
WD = REPO / "artifacts" / "official_index"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--leaves", type=int, default=127)
    p.add_argument("--trees", type=int, default=600)
    p.add_argument("--min-child", type=int, default=100)
    args = p.parse_args()
    t0 = time.time()
    sel = json.loads((SRC / "frozen_v5" / "selected.json").read_text())
    cache = sorted((REPO / "artifacts" / "cache_v5").glob("features_*.npz"), key=lambda q: q.stat().st_size)[-1]
    z = np.load(cache, allow_pickle=True)
    fold_db = sqlite3.connect(WD / "official_70bc1d8a16c6_folds.sqlite")
    rng = random.Random(42)

    def sample(fold, n):
        rows = list(fold_db.execute("SELECT id FROM fold WHERE fold=? ORDER BY id", (fold,)))
        rng.shuffle(rows)
        return [r[0] for r in rows[:n]]

    select_ids = sample("select", 5000)
    fit_ids = set(sample("fit", 30000))
    sel_set = set(select_ids)
    owner = z["owner"]
    m_sel = np.fromiter((o in sel_set for o in owner), bool, len(owner))
    m_fit = np.fromiter((o in fit_ids for o in owner), bool, len(owner))
    X = np.hstack([z["X32"], z["X11"]])
    y = z["y"]
    gt = {i: set() for i in select_ids}
    for row in iter_tsv(REPO / "student_resource" / "dataset" / "train" / "train_ground_truth.tsv", GT_H):
        if row["source1_entity_id"] in sel_set and row["matched_entity_ids"]:
            gt[row["source1_entity_id"]] = set(row["matched_entity_ids"].split(","))
    s_own, s_tid, Xs = owner[m_sel], z["tid"][m_sel], X[m_sel]

    def decide(pr, thr):
        pred = {s: set() for s in select_ids}
        for s, t, q in zip(s_own, s_tid, pr):
            if q >= thr:
                pred[s].add(t)
        return pred

    with (REPO / sel["model_path"]).open("rb") as f:
        inc = pickle.load(f)[sel["model_key"]]
    p_inc = inc.booster_.predict(Xs, num_threads=args.threads)
    inc_pred = decide(p_inc, sel["threshold"])
    inc_s = np.asarray([entity_f05(gt[s], inc_pred[s]) for s in select_ids])
    m_inc = score_predictions(gt, inc_pred)

    import lightgbm as lgb

    clf = lgb.LGBMClassifier(n_estimators=args.trees, learning_rate=0.05, num_leaves=args.leaves,
                             min_child_samples=args.min_child, subsample=0.9, subsample_freq=1,
                             colsample_bytree=0.9, random_state=42, verbose=-1, n_jobs=args.threads)
    clf.fit(X[m_fit], y[m_fit])
    p_new = clf.booster_.predict(Xs, num_threads=args.threads)
    grid = np.round(np.concatenate([np.linspace(0.5, 0.9, 9), np.linspace(0.91, 0.995, 18)]), 4)
    best = max(((float(t), score_predictions(gt, decide(p_new, float(t)))) for t in grid), key=lambda x: x[1]["macro_F0.5"])
    new_pred = decide(p_new, best[0])
    d = np.asarray([entity_f05(gt[s], new_pred[s]) for s in select_ids]) - inc_s
    r = np.random.default_rng(0)
    boots = [d[r.integers(0, len(d), len(d))].mean() for _ in range(1000)]
    res = {
        "hypothesis": "higher-capacity LightGBM recovers rejected retrieved true links",
        "config": {"num_leaves": args.leaves, "n_estimators": args.trees, "min_child_samples": args.min_child},
        "incumbent_v5_select": m_inc, "challenger_select": best[1], "challenger_threshold": best[0],
        "paired_delta": float(d.mean()), "delta_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "s1_better": int((d > 1e-12).sum()), "s1_worse": int((d < -1e-12).sum()), "sec": time.time() - t0,
    }
    out = REPO / "reports" / "official" / "official_70bc1d8a16c6" / "matcher_exp3" / f"capacity_l{args.leaves}_t{args.trees}.json"
    out.write_text(json.dumps(res, indent=2) + "\n")
    with (WD / f"matcher_v5_capacity_l{args.leaves}_t{args.trees}.pkl").open("wb") as f:
        pickle.dump({"model": clf, "threshold": best[0], "result": res}, f)
    print(json.dumps({k: res[k] for k in ("challenger_threshold", "paired_delta", "delta_ci95", "s1_better", "s1_worse", "sec")}))
    print("incumbent", round(m_inc["macro_F0.5"], 4), "challenger", round(best[1]["macro_F0.5"], 4))


if __name__ == "__main__":
    main()
