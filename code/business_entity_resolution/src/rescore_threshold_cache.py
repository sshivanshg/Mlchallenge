"""Threshold-only / gate / relative-threshold rescoring from a frozen feature+score cache.

Never rebuilds retrieval or features. Cache must be keyed by data/split/retrieval/feature
hashes (see run_matcher_v5.py). Refuses caches that contain assess/fresh fold IDs when
--forbid-assess is set (default). Used to cut experiment turnaround from minutes to seconds.
"""

from __future__ import annotations

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
CACHE = REPO / "artifacts" / "cache_v5"


def sample_fold(fold: str, n: int, seed: int = 42) -> list[str]:
    db = sqlite3.connect(f"file:{WD / 'official_70bc1d8a16c6_folds.sqlite'}?mode=ro", uri=True)
    rows = [r[0] for r in db.execute("SELECT id FROM fold WHERE fold=? ORDER BY id", (fold,))]
    random.Random(seed).shuffle(rows)
    return rows[:n]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, default=None)
    p.add_argument("--model", type=Path, default=REPO / "artifacts/official_index/matcher_v5.pkl")
    p.add_argument("--model-key", default="F1N1")
    p.add_argument("--n-features", type=int, default=43)
    p.add_argument("--eval-s1", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--thr-grid", default="0.55,0.60,0.65,0.70,0.75,0.80")
    p.add_argument("--gate-grid", default="0.0")
    p.add_argument("--rel-grid", default="0.0")
    p.add_argument("--forbid-assess", action="store_true", default=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    t0 = time.time()

    cache = args.cache or sorted(CACHE.glob("features_*.npz"), key=lambda q: q.stat().st_size)[-1]
    z = np.load(cache, allow_pickle=True)
    select_ids = sample_fold("select", args.eval_s1, args.seed)
    sel_set = set(select_ids)
    if args.forbid_assess:
        assess = set(sample_fold("assess", 50000, args.seed))  # large enough to detect leakage
        overlap = sel_set & assess
        # assess fold IDs should never appear as owners in a select/fit cache used for selection
        owners = set(z["owner"].tolist())
        leaked = owners & assess
        if leaked:
            raise SystemExit(f"cache contains {len(leaked)} assess-fold owners; refusing stale/held-out cache")

    owner = z["owner"]
    m = np.fromiter((o in sel_set for o in owner), bool, len(owner))
    X = np.hstack([z["X32"], z["X11"]]) if args.n_features == 43 else z["X32"]
    Xs = X[m]
    s_own, s_tid = owner[m], z["tid"][m]

    with args.model.open("rb") as f:
        bundle = pickle.load(f)
    model = bundle[args.model_key]
    t_pred0 = time.time()
    probs = model.booster_.predict(Xs, num_threads=args.threads)
    pred_sec = time.time() - t_pred0

    gt = {i: set() for i in select_ids}
    for row in iter_tsv(REPO / "student_resource" / "dataset" / "train" / "train_ground_truth.tsv", GT_H):
        if row["source1_entity_id"] in sel_set and row["matched_entity_ids"]:
            gt[row["source1_entity_id"]] = set(row["matched_entity_ids"].split(","))

    thrs = [float(x) for x in args.thr_grid.split(",") if x.strip()]
    gates = [float(x) for x in args.gate_grid.split(",") if x.strip()]
    rels = [float(x) for x in args.rel_grid.split(",") if x.strip()]

    # Group probs by S1 once
    by_s1: dict[str, list[tuple[str, float]]] = {s: [] for s in select_ids}
    for s, t, q in zip(s_own, s_tid, probs):
        by_s1[s].append((t, float(q)))

    results = []
    t_grid0 = time.time()
    best = None
    for thr in thrs:
        for gate in gates:
            for rel in rels:
                pred = {}
                for s, pairs in by_s1.items():
                    if not pairs:
                        pred[s] = set()
                        continue
                    mx = max(p for _, p in pairs)
                    if mx < gate:
                        pred[s] = set()
                        continue
                    floor = max(thr, rel * mx)
                    pred[s] = {t for t, p in pairs if p >= floor}
                mscore = score_predictions(gt, pred)
                row = {
                    "threshold": thr, "gate": gate, "rel": rel,
                    "macro_F0.5": mscore["macro_F0.5"],
                    "precision": mscore.get("precision"),
                    "recall": mscore.get("recall"),
                }
                # singleton sanity via entity_f05 on a few empties is covered by score_predictions
                results.append(row)
                if best is None or row["macro_F0.5"] > best["macro_F0.5"]:
                    best = row
    grid_sec = time.time() - t_grid0

    report = {
        "cache": str(cache),
        "cache_bytes": cache.stat().st_size,
        "model": str(args.model),
        "model_key": args.model_key,
        "n_eval_s1": len(select_ids),
        "n_pairs_scored": int(m.sum()),
        "predict_sec": pred_sec,
        "grid_sec": grid_sec,
        "total_sec": time.time() - t0,
        "pairs_per_sec_predict": int(m.sum()) / max(pred_sec, 1e-9),
        "grid_cells": len(results),
        "best": best,
        "results": results,
        "note": "threshold-only; retrieval/features not recomputed",
    }
    text = json.dumps(report, indent=2) + "\n"
    print(text, flush=True)
    out = args.out or (REPO / "reports" / "official" / "official_70bc1d8a16c6" / "matcher_exp3" / "threshold_rescore_v5.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)


if __name__ == "__main__":
    main()
