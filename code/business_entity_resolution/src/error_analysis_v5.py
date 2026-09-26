"""Selection-fold loss decomposition for the frozen v5 policy (uses the v5 feature cache).

Splits 1 - macro F0.5 into retrieval loss (1 - candidate oracle) and matcher loss,
then matcher loss into false-positive and false-negative parts, with singleton and
country slices and feature profiles of false-positive pairs.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import pickle
import random
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from evaluation.skill_metric import entity_f05
from frozen_v5 import policy as P5
from official_core import SOURCE_H, iter_tsv
from run_official_bounded import GT_H

REPO = SRC.parents[2]
WD = REPO / "artifacts" / "official_index"


def main():
    sel = json.loads((SRC / "frozen_v5" / "selected.json").read_text())
    cache = sorted((REPO / "artifacts" / "cache_v5").glob("features_*.npz"), key=lambda p: p.stat().st_size)[-1]
    z = np.load(cache, allow_pickle=True)
    fold_db = sqlite3.connect(WD / "official_70bc1d8a16c6_folds.sqlite")
    rows = list(fold_db.execute("SELECT id FROM fold WHERE fold='select' ORDER BY id"))
    random.Random(42).shuffle(rows)
    ids = [r[0] for r in rows[:5000]]
    need = set(ids)
    owner = z["owner"]
    m = np.fromiter((o in need for o in owner), bool, len(owner))
    X = np.hstack([z["X32"][m], z["X11"][m]])
    own, tid = owner[m], z["tid"][m]
    with (REPO / sel["model_path"]).open("rb") as f:
        model = pickle.load(f)[sel["model_key"]]
    prob = model.booster_.predict(X, num_threads=1)
    ds = REPO / "student_resource" / "dataset"
    gt = {i: set() for i in ids}
    for row in iter_tsv(ds / "train" / "train_ground_truth.tsv", GT_H):
        if row["source1_entity_id"] in need and row["matched_entity_ids"]:
            gt[row["source1_entity_id"]] = set(row["matched_entity_ids"].split(","))
    country = {r["entity_id"]: r["country"] for r in iter_tsv(ds / "train" / "train_source1.tsv", SOURCE_H) if r["entity_id"] in need}
    cand, pred = defaultdict(set), defaultdict(set)
    fp_rows, fn_rej_probs = [], []
    for i, (s, t, q) in enumerate(zip(own, tid, prob)):
        cand[s].add(t)
        if q >= sel["threshold"]:
            pred[s].add(t)
            if t not in gt[s]:
                fp_rows.append(i)
        elif t in gt[s]:
            fn_rej_probs.append(q)
    F = {s: entity_f05(gt[s], pred[s]) for s in ids}
    O = {s: entity_f05(gt[s], gt[s] & cand[s]) for s in ids}
    noFP = {s: entity_f05(gt[s], pred[s] & gt[s]) for s in ids}
    allTP = {s: entity_f05(gt[s], pred[s] | (gt[s] & cand[s])) for s in ids}
    mean = lambda d, ss=ids: float(np.mean([d[s] for s in ss]))
    report = {
        "n": len(ids), "macro_F0.5": mean(F), "candidate_oracle": mean(O),
        "loss_total": 1 - mean(F), "loss_retrieval(1-oracle)": 1 - mean(O), "loss_matcher(oracle-F)": mean(O) - mean(F),
        "gain_if_no_false_positives": mean(noFP) - mean(F),
        "gain_if_all_retrieved_truths_accepted": mean(allTP) - mean(F),
        "retrieved_truths_rejected": len(fn_rej_probs),
        "rejected_truth_prob_quantiles": {q: float(np.quantile(fn_rej_probs, q)) for q in (0.25, 0.5, 0.75, 0.9)} if fn_rej_probs else None,
        "false_positive_links": len(fp_rows),
        "singletons": {"n": sum(1 for s in ids if not gt[s]), "false_merge_s1": sum(1 for s in ids if not gt[s] and pred[s]),
                       "macro_loss_share": float(sum(1 - F[s] for s in ids if not gt[s]) / len(ids))},
        "by_country": {c: {"n": len(ss), "F": mean(F, ss), "oracle": mean(O, ss)}
                       for c, ss in ((c, [s for s in ids if country[s] == c]) for c in sorted(set(country.values())))},
    }
    names = P5.FEATURE_NAMES
    if fp_rows:
        Xfp = X[fp_rows]
        prof = {}
        for fname, cond in (("name_exact", lambda v: v >= 1), ("name_token_sort>=0.9", lambda v: v >= 0.9),
                            ("addr_token_sort<0.5", lambda v: v < 0.5), ("num_conflict", lambda v: v >= 1),
                            ("long_num_conflict", lambda v: v >= 1), ("country_equal", lambda v: v >= 1)):
            col = names.index(fname.split("<")[0].split(">")[0])
            prof[fname] = float(np.mean([cond(v) for v in Xfp[:, col]]))
        report["false_positive_feature_profile(fraction of FP links)"] = prof
    out = REPO / "reports" / "official" / "official_70bc1d8a16c6" / "matcher_exp3" / "error_analysis_v5_select.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
