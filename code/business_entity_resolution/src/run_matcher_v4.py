"""Matcher experiments on the frozen v3b cap-300 candidate pool (official data only).

Isolated changes against the selected baseline (LightGBM, 9 features, sampled
negatives, 8k fit S1):
  M1  same 9 features, every candidate of each fit S1 as training data (all hard negatives)
  M2  M1 + candidate-competition / retrieval / route / missing-field features
Both can use more fit S1s (--fit-s1). Thresholds tuned on the 5k selection IDs only.
Selection scores are development scores, not held-out claims.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse
import json
import pickle
import random
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from data.provenance import load_manifest, require_official_dataset
from evaluation.metric_ext import score_predictions
from frozen_v3b import policy as P
from official_core import SOURCE_H, iter_tsv
from run_blocking_diag import ROUTE_TABLES, Retriever, rss_mb, s3_boundaries
from run_official_bounded import GT_H

REPO = SRC.parents[2]
WD = REPO / "artifacts" / "official_index"
ROUTES = ["name", "sig", "prefix", "token", "tok_intersect", "pair", "toknum", "num"]
EXTRA_NAMES = (
    ["rank_frac", "log_score", "score_ratio_to_top", "n_cands", "name_s_rank", "name_s_gap_to_best_other",
     "n_cands_name_ge_0.9", "addr_s_rank", "s1_addr_missing", "t_addr_missing", "num_conflict", "same_source_count_name_ge_0.9"]
    + [f"route_{r}" for r in ROUTES]
)


def block_features(a, cands, scores, routes, recs):
    """Base 9 features + competition features for all candidates of one S1."""
    base = np.stack([P.feat_vec(a, recs[t]) for t in cands])
    n = len(cands)
    sc = np.asarray([scores[t] for t in cands], dtype=np.float32)
    name_s, addr_s, num_ov = base[:, 0], base[:, 3], base[:, 4]
    order_name = (-name_s).argsort(kind="stable")
    name_rank = np.empty(n, dtype=np.float32)
    name_rank[order_name] = np.arange(n)
    order_addr = (-addr_s).argsort(kind="stable")
    addr_rank = np.empty(n, dtype=np.float32)
    addr_rank[order_addr] = np.arange(n)
    top2 = np.sort(name_s)[::-1][:2]
    best_other = np.where(name_s >= top2[0], top2[1] if n > 1 else 0.0, top2[0])
    strong = name_s >= 0.9
    src = np.asarray([t.startswith("S2-") for t in cands])
    same_src_strong = np.where(src, (strong & src).sum(), (strong & ~src).sum()).astype(np.float32)
    s1_addr_missing = float(not (a.get("addr_norm") or "").strip())
    t_addr_missing = np.asarray([float(not (recs[t]["addr_norm"] or "").strip()) for t in cands], dtype=np.float32)
    a_nums = set(P.nums(a.get("addr_norm") or ""))
    num_conflict = np.asarray(
        [float(bool(a_nums) and bool(tn := set(P.nums(recs[t]["addr_norm"] or ""))) and not (a_nums & tn)) for t in cands],
        dtype=np.float32,
    )
    route_m = np.asarray([[float(r in routes.get(t, ())) for r in ROUTES] for t in cands], dtype=np.float32)
    extra = np.column_stack([
        np.arange(n, dtype=np.float32) / P.CAP,
        np.log1p(sc),
        sc / max(float(sc.max()), 1e-6),
        np.full(n, n / P.CAP, dtype=np.float32),
        name_rank / P.CAP,
        name_s - best_other,
        np.full(n, strong.sum(), dtype=np.float32),
        addr_rank / P.CAP,
        np.full(n, s1_addr_missing, dtype=np.float32),
        t_addr_missing,
        num_conflict,
        same_src_strong,
        route_m,
    ])
    return base, extra


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fit-s1", type=int, default=15000)
    p.add_argument("--eval-s1", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default="v4")
    args = p.parse_args()
    t_all = time.time()
    ds = REPO / "student_resource" / "dataset"
    prov = require_official_dataset(ds, require=("train",), check_hashes=True)
    split_version = f"official_{load_manifest()['files']['train/train_ground_truth.tsv']['sha256'][:12]}"
    out_dir = REPO / "reports" / "official" / split_version / "matcher_exp2"
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_db = sqlite3.connect(WD / f"{split_version}_folds.sqlite")
    rng = random.Random(args.seed)

    def sample(fold, n):
        rows = list(fold_db.execute("SELECT id FROM fold WHERE fold=? ORDER BY id", (fold,)))
        rng.shuffle(rows)
        return [r[0] for r in rows[:n]]

    select_ids = sample("select", args.eval_s1)
    fit_ids = sample("fit", args.fit_s1)
    fit8k = set(fit_ids[:8000])
    fold_db.close()
    need = set(select_ids) | set(fit_ids)
    s1 = {r["entity_id"]: r for r in iter_tsv(ds / "train" / "train_source1.tsv", SOURCE_H) if r["entity_id"] in need}

    db = sqlite3.connect(f"file:{WD / 'targets_index_v2.sqlite'}?mode=ro", uri=True)
    pdb = sqlite3.connect(f"file:{WD / 'train_pair_index_v3.sqlite'}?mode=ro", uri=True)
    n_targets = db.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
    ret = Retriever(db, n_targets, "v3b", s3_boundaries(db, ROUTE_TABLES), pdb, s3_boundaries(pdb, ("by_pair", "by_toknum")))

    def build(ids):
        t0 = time.time()
        rows = {}
        for sid in ids:
            r = s1[sid]
            ranked, scores, routes, _ = ret.retrieve(r["business_name"], r["business_address"], r["country"])
            cands = ranked[: P.CAP]
            if not cands:
                rows[sid] = (cands, None, None)
                continue
            recs = P.load_records(db, cands)
            base, extra = block_features(P.s1_view(r), cands, scores, routes, recs)
            rows[sid] = (cands, base, extra)
        return rows, time.time() - t0

    fit_rows, t_fit = build(fit_ids)
    sel_rows, t_sel = build(select_ids)
    print(f"built fit {t_fit:.0f}s select {t_sel:.0f}s rss={rss_mb():.0f}MB", flush=True)

    gt = {i: set() for i in need}
    for row in iter_tsv(ds / "train" / "train_ground_truth.tsv", GT_H):
        if row["source1_entity_id"] in need and row["matched_entity_ids"]:
            gt[row["source1_entity_id"]] = set(row["matched_entity_ids"].split(","))
    sel_gt = {i: gt[i] for i in select_ids}

    def stack(rows_map, ids, use_extra, neg_policy):
        Xs, ys = [], []
        for sid in ids:
            cands, base, extra = rows_map[sid]
            if base is None:
                continue
            y = np.asarray([t in gt[sid] for t in cands], dtype=np.int8)
            idx = np.arange(len(cands))
            if neg_policy == "sampled":
                pos, neg = idx[y == 1].tolist(), idx[y == 0].tolist()
                rng.shuffle(neg)
                idx = np.asarray(pos + neg[: max(8, 3 * max(len(pos), 1))], dtype=int)
            X = np.hstack([base, extra]) if use_extra else base
            Xs.append(X[idx])
            ys.append(y[idx])
        return np.vstack(Xs), np.concatenate(ys)

    def sel_matrix(use_extra):
        Xs, owners = [], []
        for sid in select_ids:
            cands, base, extra = sel_rows[sid]
            if base is None:
                continue
            Xs.append(np.hstack([base, extra]) if use_extra else base)
            owners.extend((sid, t) for t in cands)
        return np.vstack(Xs), owners

    grid = np.round(np.concatenate([np.linspace(0.5, 0.9, 9), np.linspace(0.91, 0.995, 18)]), 4)

    def tune(probs, owners):
        best = None
        for thr in grid:
            pred = {s: set() for s in select_ids}
            for (s, t), q in zip(owners, probs):
                if q >= thr:
                    pred[s].add(t)
            m = score_predictions(sel_gt, pred)
            if best is None or m["macro_F0.5"] > best[1]["macro_F0.5"]:
                best = (float(thr), m)
        return best

    import lightgbm as lgb

    experiments = [
        ("M0_repro_sampled_9f_fit8k", False, "sampled", [s for s in fit_ids if s in fit8k]),
        ("M1_allneg_9f_fit8k", False, "all", [s for s in fit_ids if s in fit8k]),
        ("M1b_allneg_9f_fit15k", False, "all", fit_ids),
        ("M2_allneg_extra_fit15k", True, "all", fit_ids),
    ]
    mats = {False: sel_matrix(False), True: sel_matrix(True)}
    results = {}
    models = {}
    for name, use_extra, negp, ids in experiments:
        t0 = time.time()
        X, y = stack(fit_rows, ids, use_extra, negp)
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                                 subsample=0.9, subsample_freq=1, colsample_bytree=0.9, random_state=args.seed,
                                 verbose=-1, n_jobs=4)
        clf.fit(X, y)
        Xs, owners = mats[use_extra]
        probs = clf.booster_.predict(Xs, num_threads=4)
        thr, m = tune(probs, owners)
        results[name] = {"threshold": thr, "select": m, "fit_pairs": int(len(y)), "fit_pos": int(y.sum()),
                         "n_fit_s1": len(ids), "features": P.FEATURE_NAMES + (EXTRA_NAMES if use_extra else []),
                         "neg_policy": negp, "sec": time.time() - t0}
        models[name] = clf
        print(f"{name}: F0.5={m['macro_F0.5']:.4f} thr={thr} P={m['precision']:.4f} R={m['recall']:.4f} "
              f"sing={m['singleton_accuracy']:.4f} fit_pairs={len(y)} {time.time()-t0:.0f}s", flush=True)

    best = max(results, key=lambda k: results[k]["select"]["macro_F0.5"])
    with (WD / f"matcher_{args.tag}.pkl").open("wb") as f:
        pickle.dump({"models": models, "results": results, "leader": best, "extra_names": EXTRA_NAMES,
                     "base_names": P.FEATURE_NAMES, "routes": ROUTES, "cap": P.CAP, "variant": "v3b"}, f)
    report = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provenance": f"official_challenge_dataset:{prov.manifest_version}",
        "split_version": split_version,
        "role": "development (fit + select); not a held-out claim",
        "candidate_policy": "frozen v3b cap 300",
        "n_select": len(select_ids),
        "results": results,
        "leader_on_select": best,
        "baseline_selected_model_select_F0.5": 0.796288,
        "runtime_sec": {"fit_build": t_fit, "select_build": t_sel, "total": time.time() - t_all},
        "peak_rss_mb": rss_mb(),
    }
    (out_dir / f"matcher_{args.tag}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"leader={best}")


if __name__ == "__main__":
    main()
