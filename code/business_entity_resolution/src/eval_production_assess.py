"""Score the current production policy (v2 cap 120 + logistic_corroboration_v1 @ its
frozen threshold + corroboration gate) on a frozen assessment ID file, once."""

from __future__ import annotations

import argparse
import json
import pickle
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from data.provenance import load_manifest, require_official_dataset
from evaluation.metric_ext import score_predictions
from official_core import SOURCE_H, accept_match, feat_vec, generate_candidates_v2, iter_tsv, load_records, norm_addr, norm_name
from run_blocking_diag import cap_metrics
from run_official_bounded import GT_H

REPO = SRC.parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=REPO / "student_resource" / "dataset")
    p.add_argument("--work-dir", type=Path, default=REPO / "artifacts" / "official_index")
    p.add_argument("--assess-ids", type=Path, required=True)
    args = p.parse_args()
    t_all = time.time()
    prov = require_official_dataset(args.dataset_root, require=("train",), check_hashes=True)
    split_version = f"official_{load_manifest()['files']['train/train_ground_truth.tsv']['sha256'][:12]}"
    spec = json.loads(args.assess_ids.read_text())
    ids = spec["ids"]
    needed = set(ids)
    with (args.work_dir / "logistic_corroboration_v1.pkl").open("rb") as f:
        bundle = pickle.load(f)
    scaler, clf, thr, cap = bundle["scaler"], bundle["clf"], float(bundle["threshold"]), int(bundle["max_candidates"])

    s1_map = {}
    for row in iter_tsv(args.dataset_root / "train" / "train_source1.tsv", SOURCE_H):
        if row["entity_id"] in needed:
            s1_map[row["entity_id"]] = row
    db = sqlite3.connect(f"file:{args.work_dir / 'targets_index_v2.sqlite'}?mode=ro", uri=True)
    n_targets = db.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
    df_cache: dict[str, int] = {}
    cands, pred = {}, {}
    for sid in ids:
        r = s1_map[sid]
        c = generate_candidates_v2(db, r["business_name"], r["business_address"], r["country"],
                                   n_targets=n_targets, df_cache=df_cache, max_candidates=cap)
        cands[sid] = c
        pred[sid] = set()
        if not c:
            continue
        recs = load_records(db, c)
        aa = {**r, "name_norm": norm_name(r["business_name"]), "addr_norm": norm_addr(r["business_address"])}
        tids = [t for t in c if t in recs]
        F = np.stack([feat_vec(aa, recs[t]) for t in tids])
        pr = clf.predict_proba(scaler.transform(F))[:, 1]
        pred[sid] = {t for t, f, p_ in zip(tids, F, pr) if accept_match(float(p_), f, thr)}

    gt = {i: set() for i in ids}
    for row in iter_tsv(args.dataset_root / "train" / "train_ground_truth.tsv", GT_H):
        sid = row["source1_entity_id"]
        if sid in needed:
            gt[sid] = {x for x in row["matched_entity_ids"].split(",") if x} if row["matched_entity_ids"] else set()
    m = score_predictions(gt, pred)
    out = REPO / "reports" / "official" / split_version / "retrieval_exp1" / "production_v2_assess_fresh_v1.json"
    out.write_text(json.dumps({
        "provenance": f"official_challenge_dataset:{prov.manifest_version}",
        "policy": "v2 cap120 + logistic_corroboration_v1 + corroboration gate",
        "threshold": thr,
        "assess_ids": str(args.assess_ids),
        "assess_ids_sha256": spec["sha256_of_sorted_ids"],
        "metrics": m,
        "blocking": cap_metrics(cands, gt),
        "runtime_sec": time.time() - t_all,
    }, indent=2) + "\n")
    print(f"PRODUCTION on fresh assess: F0.5={m['macro_F0.5']:.4f} P={m['precision']} R={m['recall']} sing={m['singleton_accuracy']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
