"""Train H4-capacity LightGBM on v6 addrtok candidates (fit 30k / select 5k).

Builds a fresh feature cache keyed by frozen_v6 retrieval (addrtok only), trains
the same capacity config as matcher_v5_capacity_l127_t600, retunes threshold for
macro F0.5. Does not open assess_fresh_v4.
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
from frozen_v5 import policy as P5
from frozen_v6 import policy as P
from official_core import SOURCE_H, iter_tsv
from run_official_bounded import GT_H

REPO = SRC.parents[2]
WD = REPO / "artifacts" / "official_index"
CACHE = REPO / "artifacts" / "cache_v6"
REP = REPO / "reports" / "official" / "official_70bc1d8a16c6" / "matcher_exp3"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-s1", type=int, default=30000)
    ap.add_argument("--eval-s1", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--leaves", type=int, default=127)
    ap.add_argument("--trees", type=int, default=600)
    ap.add_argument("--min-child", type=int, default=100)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    t_all = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    REP.mkdir(parents=True, exist_ok=True)

    fold = sqlite3.connect(f"file:{WD / 'official_70bc1d8a16c6_folds.sqlite'}?mode=ro", uri=True)
    rng = random.Random(args.seed)

    def sample(fold_name, n):
        rows = [r[0] for r in fold.execute("SELECT id FROM fold WHERE fold=? ORDER BY id", (fold_name,))]
        rng.shuffle(rows)
        return rows[:n]

    select_ids = sample("select", args.eval_s1)
    fit_ids = sample("fit", args.fit_s1)
    fold.close()
    need = set(select_ids) | set(fit_ids)

    key = hashlib.sha256(json.dumps({
        "policy": P.POLICY_ID,
        "policy_sha": hashlib.sha256(Path(P.__file__).read_bytes()).hexdigest(),
        "features": P.FEATURE_NAMES,
        "fit": args.fit_s1, "eval": args.eval_s1, "seed": args.seed,
        "retrieval": "addrtok_only",
    }, sort_keys=True).encode()).hexdigest()[:16]
    cache_path = CACHE / f"features_{key}.npz"

    gt = {i: set() for i in need}
    for row in iter_tsv(REPO / "student_resource" / "dataset" / "train" / "train_ground_truth.tsv", GT_H):
        if row["source1_entity_id"] in need and row["matched_entity_ids"]:
            gt[row["source1_entity_id"]] = set(row["matched_entity_ids"].split(","))

    t0 = time.time()
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        data = {k: z[k] for k in z.files}
        print(f"loaded cache {cache_path}", flush=True)
    else:
        s1 = {}
        for row in iter_tsv(REPO / "student_resource" / "dataset" / "train" / "train_source1.tsv", SOURCE_H):
            if row["entity_id"] in need:
                s1[row["entity_id"]] = row
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
        order = select_ids + fit_ids
        Xs, owners, tids, ys = [], [], [], []
        for i, sid in enumerate(order):
            r = s1[sid]
            c, sc, rt = ret.retrieve(r["business_name"], r["business_address"], r["country"])
            if not c:
                continue
            recs = P.load_records(db, c)
            a = P.s1_view(r)
            Xs.append(P.features(a, c, sc, rt, recs))
            owners.extend([sid] * len(c))
            tids.extend(c)
            ys.extend(int(t in gt[sid]) for t in c)
            if (i + 1) % 2000 == 0:
                print(f"  built {i+1}/{len(order)} {time.time()-t0:.0f}s", flush=True)
        data = {
            "X": np.vstack(Xs),
            "owner": np.asarray(owners),
            "tid": np.asarray(tids),
            "y": np.asarray(ys, dtype=np.int8),
        }
        np.savez(cache_path, **data)
        print(f"wrote cache {cache_path} rows={len(data['y'])}", flush=True)
    print(f"features ready {time.time()-t0:.0f}s rows={len(data['y'])}", flush=True)

    owner, tid, y, X = data["owner"], data["tid"], data["y"], data["X"]
    assert X.shape[1] == 43, X.shape
    sel_set, fit_set = set(select_ids), set(fit_ids)
    m_sel = np.fromiter((o in sel_set for o in owner), bool, len(owner))
    m_fit = np.fromiter((o in fit_set for o in owner), bool, len(owner))
    sel_gt = {s: gt[s] for s in select_ids}
    s_own, s_tid, Xs = owner[m_sel], tid[m_sel], X[m_sel]

    # oracle on select
    cands_by = {}
    for s, t in zip(s_own, s_tid):
        cands_by.setdefault(s, set()).add(t)
    oracle = {s: sel_gt[s] & cands_by.get(s, set()) for s in select_ids}
    m_oracle = score_predictions(sel_gt, oracle)

    import lightgbm as lgb

    clf = lgb.LGBMClassifier(
        n_estimators=args.trees, learning_rate=0.05, num_leaves=args.leaves,
        min_child_samples=args.min_child, subsample=0.9, subsample_freq=1,
        colsample_bytree=0.9, random_state=args.seed, verbose=-1, n_jobs=args.threads,
    )
    t_fit = time.time()
    clf.fit(X[m_fit], y[m_fit])
    print(f"trained {time.time()-t_fit:.0f}s fit_pairs={int(m_fit.sum())}", flush=True)
    probs = clf.booster_.predict(Xs, num_threads=args.threads)

    def decide(pr, thr):
        pred = {s: set() for s in select_ids}
        for s, t, q in zip(s_own, s_tid, pr):
            if q >= thr:
                pred[s].add(t)
        return pred

    grid = np.round(np.concatenate([np.linspace(0.5, 0.9, 9), np.linspace(0.91, 0.995, 18)]), 4)
    best = max(((float(t), score_predictions(sel_gt, decide(probs, float(t)))) for t in grid),
               key=lambda x: x[1]["macro_F0.5"])
    new_pred = decide(probs, best[0])

    # incumbent: frozen v5 model on the SAME addrtok candidates
    with (REPO / "artifacts/official_index/matcher_v5.pkl").open("rb") as f:
        inc = pickle.load(f)["F1N1"]
    p_inc = inc.booster_.predict(Xs, num_threads=args.threads)
    # retune incumbent threshold on addrtok too for fair paired compare
    best_inc = max(((float(t), score_predictions(sel_gt, decide(p_inc, float(t)))) for t in grid),
                   key=lambda x: x[1]["macro_F0.5"])
    inc_pred = decide(p_inc, best_inc[0])
    d = np.asarray([entity_f05(sel_gt[s], new_pred[s]) - entity_f05(sel_gt[s], inc_pred[s]) for s in select_ids])
    rngb = np.random.default_rng(0)
    boots = [d[rngb.integers(0, len(d), len(d))].mean() for _ in range(1000)]

    model_path = WD / "matcher_v6_addrtok_h4.pkl"
    bundle = {
        "model": clf,
        "threshold": best[0],
        "policy": P.POLICY_ID,
        "base_features": P5.V4_FEATURES if hasattr(P5, "V4_FEATURES") else P.FEATURE_NAMES[:32],
        "extra2_features": P.EXTRA2_NAMES,
        "feature_names": P.FEATURE_NAMES,
        "n_features": 43,
        "config": {"num_leaves": args.leaves, "n_estimators": args.trees, "min_child_samples": args.min_child},
        "fit_s1": args.fit_s1,
        "select_macro_F0.5": best[1]["macro_F0.5"],
    }
    # align with frozen_v5 feature names for infer compatibility
    from frozen_v5.policy import V4_FEATURES, EXTRA2_NAMES as E2
    bundle["base_features"] = V4_FEATURES
    bundle["extra2_features"] = E2
    with model_path.open("wb") as f:
        pickle.dump(bundle, f)
    model_sha = sha256_file(model_path)

    res = {
        "hypothesis": "retrain H4 capacity on addrtok candidates recovers matcher F0.5 under raised oracle",
        "cache": str(cache_path),
        "oracle_macro_F0.5": m_oracle["macro_F0.5"],
        "challenger": {
            "threshold": best[0],
            "metrics": best[1],
            "config": bundle["config"],
            "model": str(model_path),
            "model_sha256": model_sha,
        },
        "incumbent_v5_retuned_on_addrtok": {
            "threshold": best_inc[0],
            "metrics": best_inc[1],
        },
        "paired_challenger_minus_v5": {
            "mean": float(d.mean()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "n_better": int((d > 0).sum()),
            "n_worse": int((d < 0).sum()),
            "n_tie": int((d == 0).sum()),
        },
        "reference_h4_on_v4all_select": 0.8795,
        "reference_v5_on_v4all_select": 0.871,
        "runtime_sec": time.time() - t_all,
        "assess_fresh_v4": "frozen, not opened",
    }
    outp = REP / "matcher_v6_addrtok_h4.json"
    outp.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({
        "oracle": m_oracle["macro_F0.5"],
        "challenger_F0.5": best[1]["macro_F0.5"],
        "challenger_thr": best[0],
        "v5_on_addrtok_F0.5": best_inc[1]["macro_F0.5"],
        "paired": res["paired_challenger_minus_v5"],
        "model": str(model_path),
        "sha": model_sha,
    }, indent=2), flush=True)

    sel = {
        "model_path": str(model_path.relative_to(REPO)),
        "model_sha256": model_sha,
        "model_key": "model",
        "n_features": 43,
        "threshold": best[0],
        "gate": 0.0,
        "rel": 0.0,
        "selection_macro_F0.5": best[1]["macro_F0.5"],
        "retrieval": "v5b_addrtok_only",
        "ngram_rejected": True,
        "oracle_macro_F0.5": m_oracle["macro_F0.5"],
        "selected_from": str(outp.relative_to(REPO)),
        "paired_delta_vs_v5_on_addrtok": float(d.mean()),
    }
    (SRC / "frozen_v6" / "selected.json").write_text(json.dumps(sel, indent=2) + "\n")
    print("wrote", outp)


if __name__ == "__main__":
    main()
