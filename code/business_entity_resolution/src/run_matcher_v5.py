"""Overnight matcher experiments on frozen v4all cap-300 candidates (official data only).

Builds features once for fit/select S1s (cached by configuration hash), then runs
isolated experiments against the incumbent M2 (32 features, all negatives, 15k fit):
  D1  decision rule on incumbent probabilities: per-S1 empty gate on max probability
  D2  decision rule: relative threshold (prob >= r * max prob in the S1's list)
  F1  M2 + extra pairwise agreement/contradiction features (same 15k fit S1s)
  N1  M2 features with 2x fit S1s (30k)
Selection-fold scores only; paired per-S1 differences reported against the incumbent.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse
import hashlib
import json
import random
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from data.provenance import load_manifest, require_official_dataset
from evaluation.metric_ext import score_predictions
from evaluation.skill_metric import entity_f05
from frozen_v4 import policy as P
from official_core import SOURCE_H, iter_tsv
from run_official_bounded import GT_H

REPO = SRC.parents[2]
WD = REPO / "artifacts" / "official_index"
CACHE = REPO / "artifacts" / "cache_v5"
EXTRA2_NAMES = ["name_jw", "name_token_set", "addr_token_set", "addr_partial", "first_num_equal",
                "long_num_shared", "long_num_conflict", "name_len_ratio", "addr_len_ratio", "name_tok_contain_s1", "name_tok_contain_t"]


def extra2(a, recs, cands):
    na = a["name_norm"] or ""
    aa = a["addr_norm"] or ""
    a_nums = P.nums(aa)
    a_long = {n for n in a_nums if len(n) >= 5}
    ta = set(P.sig_tokens(na))
    rows = []
    for t in cands:
        b = recs[t]
        nb, ab = b["name_norm"] or "", b["addr_norm"] or ""
        b_nums = P.nums(ab)
        b_long = {n for n in b_nums if len(n) >= 5}
        tb = set(P.sig_tokens(nb))
        rows.append([
            JaroWinkler.normalized_similarity(na, nb) if (na or nb) else 0.0,
            fuzz.token_set_ratio(na, nb) / 100.0 if (na or nb) else 0.0,
            fuzz.token_set_ratio(aa, ab) / 100.0 if (aa or ab) else 0.0,
            fuzz.partial_ratio(aa, ab) / 100.0 if (aa or ab) else 0.0,
            float(bool(a_nums) and bool(b_nums) and a_nums[0] == b_nums[0]),
            float(bool(a_long & b_long)),
            float(bool(a_long) and bool(b_long) and not (a_long & b_long)),
            min(len(na), len(nb)) / max(len(na), len(nb), 1),
            min(len(aa), len(ab)) / max(len(aa), len(ab), 1),
            len(ta & tb) / len(ta) if ta else 0.0,
            len(ta & tb) / len(tb) if tb else 0.0,
        ])
    return np.asarray(rows, dtype=np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fit-s1", type=int, default=30000)
    p.add_argument("--eval-s1", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default="v5")
    args = p.parse_args()
    t_all = time.time()
    ds = REPO / "student_resource" / "dataset"
    prov = require_official_dataset(ds, require=("train",), check_hashes=True)
    split_version = f"official_{load_manifest()['files']['train/train_ground_truth.tsv']['sha256'][:12]}"
    out_dir = REPO / "reports" / "official" / split_version / "matcher_exp3"
    out_dir.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    fold_db = sqlite3.connect(WD / f"{split_version}_folds.sqlite")
    rng = random.Random(args.seed)

    def sample(fold, n):
        rows = list(fold_db.execute("SELECT id FROM fold WHERE fold=? ORDER BY id", (fold,)))
        rng.shuffle(rows)
        return [r[0] for r in rows[:n]]

    select_ids = sample("select", args.eval_s1)
    fit_ids = sample("fit", args.fit_s1)
    fold_db.close()
    fit15 = fit_ids[:15000]
    need = set(select_ids) | set(fit_ids)

    key = hashlib.sha256(json.dumps({
        "policy": P.POLICY_ID, "policy_sha": hashlib.sha256(Path(P.__file__).read_bytes()).hexdigest(),
        "extra2": EXTRA2_NAMES, "fit": args.fit_s1, "eval": args.eval_s1, "seed": args.seed, "split": split_version,
    }, sort_keys=True).encode()).hexdigest()[:16]
    cache_path = CACHE / f"features_{key}.npz"

    gt = {i: set() for i in need}
    for row in iter_tsv(ds / "train" / "train_ground_truth.tsv", GT_H):
        if row["source1_entity_id"] in need and row["matched_entity_ids"]:
            gt[row["source1_entity_id"]] = set(row["matched_entity_ids"].split(","))

    t0 = time.time()
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        data = {k: z[k] for k in z.files}
        print(f"loaded cache {cache_path}", flush=True)
    else:
        s1 = {r["entity_id"]: r for r in iter_tsv(ds / "train" / "train_source1.tsv", SOURCE_H) if r["entity_id"] in need}
        db = sqlite3.connect(f"file:{WD / 'targets_index_v2.sqlite'}?mode=ro", uri=True)
        pdb = sqlite3.connect(f"file:{WD / 'train_pair_index_v3.sqlite'}?mode=ro", uri=True)
        adb = sqlite3.connect(f"file:{WD / 'train_aux_index_v4.sqlite'}?mode=ro", uri=True)
        n_targets = db.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
        ret = P.Retriever(db, pdb, adb, n_targets, P.s3_boundaries(db, P.ROUTE_TABLES),
                          P.s3_boundaries(pdb, P.PAIR_TABLES), P.s3_boundaries(adb, P.AUX_TABLES))
        order = select_ids + fit_ids
        X32, X11, owner, tid, y = [], [], [], [], []
        for i, sid in enumerate(order):
            r = s1[sid]
            c, sc, rt = ret.retrieve(r["business_name"], r["business_address"], r["country"])
            if not c:
                continue
            recs = P.load_records(db, c)
            a = P.s1_view(r)
            X32.append(P.block_features(a, c, sc, rt, recs))
            X11.append(extra2(a, recs, c))
            owner.extend([sid] * len(c))
            tid.extend(c)
            y.extend(int(t in gt[sid]) for t in c)
            if (i + 1) % 5000 == 0:
                print(f"  built {i+1}/{len(order)} {time.time()-t0:.0f}s", flush=True)
        data = {"X32": np.vstack(X32), "X11": np.vstack(X11), "owner": np.asarray(owner), "tid": np.asarray(tid),
                "y": np.asarray(y, dtype=np.int8)}
        np.savez(cache_path, **data)
    t_build = time.time() - t0
    owner, tid, y = data["owner"], data["tid"], data["y"]
    sel_set, fit15_set, fit_set = set(select_ids), set(fit15), set(fit_ids)
    m_sel = np.fromiter((o in sel_set for o in owner), bool, len(owner))
    m_f15 = np.fromiter((o in fit15_set for o in owner), bool, len(owner))
    m_fall = np.fromiter((o in fit_set for o in owner), bool, len(owner))
    sel_gt = {s: gt[s] for s in select_ids}
    print(f"features ready {t_build:.0f}s rows={len(y)} select_rows={m_sel.sum()}", flush=True)

    import lightgbm as lgb

    def train(X, yy):
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                                 subsample=0.9, subsample_freq=1, colsample_bytree=0.9, random_state=args.seed,
                                 verbose=-1, n_jobs=4)
        clf.fit(X, yy)
        return clf

    sel_owner, sel_tid = owner[m_sel], tid[m_sel]
    grid = np.round(np.concatenate([np.linspace(0.5, 0.9, 9), np.linspace(0.91, 0.995, 18)]), 4)

    def decide(probs, threshold, gate=0.0, rel=0.0):
        thr = threshold
        pred = {s: set() for s in select_ids}
        best = {}
        for s, q in zip(sel_owner, probs):
            if q > best.get(s, -1):
                best[s] = q
        for s, t, q in zip(sel_owner, sel_tid, probs):
            mx = best[s]
            if q >= thr and mx >= gate and q >= rel * mx:
                pred[s].add(t)
        return pred

    def tune(probs, gates=(0.0,), rels=(0.0,)):
        best = None
        for thr in grid:
            for g in gates:
                for r in rels:
                    m = score_predictions(sel_gt, decide(probs, float(thr), g, r))
                    if best is None or m["macro_F0.5"] > best[1]["macro_F0.5"]:
                        best = ({"threshold": float(thr), "gate": g, "rel": r}, m)
        return best

    def per_s1(pred):
        return np.asarray([entity_f05(sel_gt[s], pred[s]) for s in select_ids])

    results = {}
    X32, X43 = data["X32"], np.hstack([data["X32"], data["X11"]])

    t0 = time.time()
    inc = train(X32[m_f15], y[m_f15])
    p_inc = inc.booster_.predict(X32[m_sel], num_threads=4)
    cfg, m = tune(p_inc)
    inc_scores = per_s1(decide(p_inc, **cfg))
    results["M2_incumbent_repro"] = {"cfg": cfg, "select": m, "sec": time.time() - t0}
    print(f"M2 incumbent: {m['macro_F0.5']:.4f} {cfg}", flush=True)

    def record(name, cfg, m, scores, t0, extra=None):
        d = scores - inc_scores
        rng_np = np.random.default_rng(0)
        boots = [d[rng_np.integers(0, len(d), len(d))].mean() for _ in range(1000)]
        results[name] = {"cfg": cfg, "select": m, "delta_vs_incumbent": float(d.mean()),
                         "delta_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                         "s1_improved": int((d > 1e-12).sum()), "s1_worse": int((d < -1e-12).sum()),
                         "sec": time.time() - t0, **(extra or {})}
        print(f"{name}: {m['macro_F0.5']:.4f} delta={d.mean():+.4f} CI95=[{results[name]['delta_ci95'][0]:+.4f},"
              f"{results[name]['delta_ci95'][1]:+.4f}] better={results[name]['s1_improved']} worse={results[name]['s1_worse']} "
              f"sing={m['singleton_accuracy']:.4f} {cfg}", flush=True)

    t0 = time.time()
    cfg, m = tune(p_inc, gates=(0.0, 0.8, 0.85, 0.9, 0.95), rels=(0.0,))
    record("D1_empty_gate_on_max_prob", cfg, m, per_s1(decide(p_inc, **cfg)), t0)
    t0 = time.time()
    cfg, m = tune(p_inc, gates=(0.0,), rels=(0.0, 0.5, 0.7, 0.8, 0.9))
    record("D2_relative_threshold", cfg, m, per_s1(decide(p_inc, **cfg)), t0)

    t0 = time.time()
    f1 = train(X43[m_f15], y[m_f15])
    p_f1 = f1.booster_.predict(X43[m_sel], num_threads=4)
    cfg, m = tune(p_f1)
    record("F1_extra_pairwise_features_fit15k", cfg, m, per_s1(decide(p_f1, **cfg)), t0, {"features": 43})

    t0 = time.time()
    n1 = train(X32[m_fall], y[m_fall])
    p_n1 = n1.booster_.predict(X32[m_sel], num_threads=4)
    cfg, m = tune(p_n1)
    record(f"N1_M2_features_fit{len(fit_ids)//1000}k", cfg, m, per_s1(decide(p_n1, **cfg)), t0, {"fit_rows": int(m_fall.sum())})

    t0 = time.time()
    fn = train(X43[m_fall], y[m_fall])
    p_fn = fn.booster_.predict(X43[m_sel], num_threads=4)
    cfg, m = tune(p_fn)
    record(f"F1N1_extra_features_fit{len(fit_ids)//1000}k", cfg, m, per_s1(decide(p_fn, **cfg)), t0, {"features": 43})

    import pickle
    with (WD / f"matcher_{args.tag}.pkl").open("wb") as f:
        pickle.dump({"incumbent": inc, "F1": f1, "N1": n1, "F1N1": fn, "results": results,
                     "base_features": P.FEATURE_NAMES, "extra2_features": EXTRA2_NAMES, "policy": P.POLICY_ID,
                     "fit_ids_n": len(fit_ids)}, f)
    report = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provenance": f"official_challenge_dataset:{prov.manifest_version}",
        "role": "selection-fold development; paired deltas vs incumbent on identical S1s",
        "candidate_policy": P.POLICY_ID, "n_select": len(select_ids), "n_fit_max": len(fit_ids),
        "feature_cache": str(cache_path), "results": results,
        "runtime_sec": {"build": t_build, "total": time.time() - t_all},
    }
    (out_dir / f"matcher_{args.tag}.json").write_text(json.dumps(report, indent=2) + "\n")
    print("DONE")


if __name__ == "__main__":
    main()
