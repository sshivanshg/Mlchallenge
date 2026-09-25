"""Matcher comparison under a retrieval policy (official data only).

Fit-fold candidates/negatives are regenerated with the chosen retrieval variant,
logistic (existing 9 features) is refit and one LightGBM model is compared on the
same pairs. Thresholds and the corroboration gate are tuned on the frozen
selection sample only. Optionally scores a frozen assessment ID file once.
"""

from __future__ import annotations

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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from data.provenance import load_manifest, require_official_dataset
from evaluation.metric_ext import score_predictions
from official_core import FEATURE_NAMES, SOURCE_H, accept_match, feat_vec, iter_tsv, load_records, norm_addr, norm_name
from run_blocking_diag import ROUTE_TABLES, Retriever, cap_metrics, rss_mb, s3_boundaries
from run_official_bounded import GT_H

REPO = SRC.parents[2]


def build_retriever(args, db, n_targets):
    bounds = s3_boundaries(db, ROUTE_TABLES) if args.variant != "v2" else {}
    pair_db = pair_bounds = None
    if args.variant == "v3b":
        pair_db = sqlite3.connect(f"file:{args.pair_index}?mode=ro", uri=True)
        pair_bounds = s3_boundaries(pair_db, ("by_pair", "by_toknum"))
    return Retriever(db, n_targets, args.variant, bounds, pair_db, pair_bounds)


def featurize(ids, cands, s1_map, db, cache):
    rows = []
    for sid in ids:
        a = s1_map[sid]
        aa = {**a, "name_norm": norm_name(a["business_name"]), "addr_norm": norm_addr(a["business_address"])}
        miss = [c for c in cands[sid] if c not in cache]
        if miss:
            cache.update(load_records(db, miss))
        for tid in cands[sid]:
            b = cache.get(tid)
            if b is not None:
                rows.append((sid, tid, feat_vec(aa, b)))
    return rows


def decide(rows, probs, ids, thr, gate):
    pred = {i: set() for i in ids}
    for (sid, tid, f), pr in zip(rows, probs):
        ok = accept_match(float(pr), f, thr) if gate else pr >= thr
        if ok:
            pred[sid].add(tid)
    return pred


def tune(rows, probs, ids, gt, grid):
    best = None
    for gate in (False, True):
        for thr in grid:
            m = score_predictions(gt, decide(rows, probs, ids, float(thr), gate))
            if best is None or m["macro_F0.5"] > best[2]["macro_F0.5"]:
                best = (float(thr), gate, m)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=REPO / "student_resource" / "dataset")
    p.add_argument("--work-dir", type=Path, default=REPO / "artifacts" / "official_index")
    p.add_argument("--variant", choices=["v2", "v3a", "v3b"], default="v3b")
    p.add_argument("--pair-index", type=Path, default=REPO / "artifacts" / "official_index" / "train_pair_index_v3.sqlite")
    p.add_argument("--cap", type=int, default=300)
    p.add_argument("--eval-s1", type=int, default=5000)
    p.add_argument("--fit-s1", type=int, default=8000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--assess-ids", type=Path, default=None, help="Frozen assessment JSON; scored once with tuned config")
    p.add_argument("--tag", default=None)
    args = p.parse_args()
    tag = args.tag or f"{args.variant}_cap{args.cap}"

    t_all = time.time()
    prov = require_official_dataset(args.dataset_root, require=("train",), check_hashes=True)
    gt_hash = load_manifest()["files"]["train/train_ground_truth.tsv"]["sha256"]
    split_version = f"official_{gt_hash[:12]}"
    out_dir = REPO / "reports" / "official" / split_version / "retrieval_exp1"
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_db = sqlite3.connect(args.work_dir / f"{split_version}_folds.sqlite")
    rng = random.Random(args.seed)

    def sample(fold, n):
        rows = list(fold_db.execute("SELECT id FROM fold WHERE fold=? ORDER BY id", (fold,)))
        rng.shuffle(rows)
        return [r[0] for r in rows[:n]]

    select_ids = sample("select", args.eval_s1)
    fit_ids = sample("fit", args.fit_s1)
    fold_db.close()
    assess_ids = json.loads(args.assess_ids.read_text())["ids"] if args.assess_ids else []
    needed = set(select_ids) | set(fit_ids) | set(assess_ids)

    s1_map = {}
    for row in iter_tsv(args.dataset_root / "train" / "train_source1.tsv", SOURCE_H):
        if row["entity_id"] in needed:
            s1_map[row["entity_id"]] = row

    db = sqlite3.connect(f"file:{args.work_dir / 'targets_index_v2.sqlite'}?mode=ro", uri=True)
    n_targets = db.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
    ret = build_retriever(args, db, n_targets)

    def gen(ids):
        t0 = time.time()
        out = {}
        for sid in ids:
            r = s1_map[sid]
            out[sid] = ret.retrieve(r["business_name"], r["business_address"], r["country"])[0][: args.cap]
        return out, time.time() - t0

    fit_c, t_fit_ret = gen(fit_ids)
    sel_c, t_sel_ret = gen(select_ids)
    print(f"retrieval fit {t_fit_ret:.0f}s select {t_sel_ret:.0f}s rss={rss_mb():.0f}MB", flush=True)

    gt = {i: set() for i in needed}
    for row in iter_tsv(args.dataset_root / "train" / "train_ground_truth.tsv", GT_H):
        sid = row["source1_entity_id"]
        if sid in needed:
            gt[sid] = {x for x in row["matched_entity_ids"].split(",") if x} if row["matched_entity_ids"] else set()
    sel_gt = {i: gt[i] for i in select_ids}

    cache: dict[str, dict] = {}
    t0 = time.time()
    fit_rows = []
    for sid in fit_ids:
        pos = [c for c in fit_c[sid] if c in gt[sid]]
        neg = [c for c in fit_c[sid] if c not in gt[sid]]
        rng.shuffle(neg)
        neg = neg[: max(8, 3 * max(len(pos), 1))]
        fit_rows.extend(featurize([sid], {sid: pos + neg}, s1_map, db, cache))
    X = np.stack([f for _, _, f in fit_rows])
    y = np.asarray([int(t in gt[s]) for s, t, _ in fit_rows], dtype=np.int32)
    t_fit_feat = time.time() - t0
    print(f"fit pairs n={len(y)} pos={int(y.sum())} feat {t_fit_feat:.0f}s", flush=True)

    t0 = time.time()
    sel_rows = featurize(select_ids, sel_c, s1_map, db, cache)
    Xs_sel = np.stack([f for _, _, f in sel_rows])
    t_sel_feat = time.time() - t0
    print(f"select pairs n={len(sel_rows)} feat {t_sel_feat:.0f}s rss={rss_mb():.0f}MB", flush=True)

    models = {}
    scaler = StandardScaler()
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed, solver="lbfgs")
    lr.fit(scaler.fit_transform(X), y)
    models["logistic"] = (lambda Z: lr.predict_proba(scaler.transform(Z))[:, 1])
    import lightgbm as lgb

    gbm = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                             subsample=0.9, subsample_freq=1, colsample_bytree=0.9, random_state=args.seed, verbose=-1)
    gbm.fit(X, y)
    models["lightgbm"] = (lambda Z: gbm.predict_proba(Z)[:, 1])

    grid = np.round(np.concatenate([np.linspace(0.5, 0.9, 9), np.linspace(0.91, 0.995, 18)]), 4)
    blk = cap_metrics(sel_c, sel_gt)
    results = {}
    for name, fn in models.items():
        t0 = time.time()
        probs = fn(Xs_sel)
        thr, gate, m = tune(sel_rows, probs, select_ids, sel_gt, grid)
        results[name] = {"threshold": thr, "gate": gate, "select": m, "tune_sec": time.time() - t0}
        print(f"{name}: select F0.5={m['macro_F0.5']:.4f} thr={thr} gate={gate} P={m['precision']:.4f} "
              f"R={m['recall']:.4f} sing={m['singleton_accuracy']:.4f}", flush=True)

    leader = max(results, key=lambda k: results[k]["select"]["macro_F0.5"])
    assess = None
    if assess_ids:
        ass_c, t_ass_ret = gen(assess_ids)
        ass_gt = {i: gt[i] for i in assess_ids}
        ass_rows = featurize(assess_ids, ass_c, s1_map, db, cache)
        probs = models[leader](np.stack([f for _, _, f in ass_rows]))
        cfg = results[leader]
        m = score_predictions(ass_gt, decide(ass_rows, probs, assess_ids, cfg["threshold"], cfg["gate"]))
        assess = {"model": leader, "metrics": m, "blocking": cap_metrics(ass_c, ass_gt),
                  "ids_file": str(args.assess_ids), "retrieval_sec": t_ass_ret}
        print(f"ASSESS {leader}: F0.5={m['macro_F0.5']:.4f} P={m['precision']} R={m['recall']} sing={m['singleton_accuracy']}", flush=True)

    model_path = args.work_dir / f"matcher_{tag}.pkl"
    with model_path.open("wb") as f:
        pickle.dump({"logistic": (scaler, lr), "lightgbm": gbm, "results": results, "leader": leader,
                     "feature_names": FEATURE_NAMES, "variant": args.variant, "cap": args.cap,
                     "split_version": split_version}, f)

    report = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provenance": f"official_challenge_dataset:{prov.manifest_version}",
        "split_version": split_version,
        "tag": tag,
        "variant": args.variant,
        "cap": args.cap,
        "n_fit_s1": len(fit_ids),
        "n_select_s1": len(select_ids),
        "fit_pairs": int(len(y)),
        "fit_pos": int(y.sum()),
        "negative_policy": "random max(8, 3*pos) non-truth candidates per fit S1 (unchanged from Stage 4)",
        "select_pairs": len(sel_rows),
        "select_blocking": blk,
        "results": results,
        "leader_on_select": leader,
        "assess": assess,
        "runtime_sec": {"fit_retrieval": t_fit_ret, "select_retrieval": t_sel_ret, "fit_features": t_fit_feat,
                        "select_features": t_sel_feat, "total": time.time() - t_all},
        "select_feature_ms_per_pair": 1000 * t_sel_feat / max(len(sel_rows), 1),
        "peak_rss_mb": rss_mb(),
        "model_path": str(model_path),
    }
    out = out_dir / f"matcher_{tag}{'_assess' if assess_ids else ''}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
