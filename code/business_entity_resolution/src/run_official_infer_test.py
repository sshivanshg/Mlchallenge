"""Produce official test submission TSVs (challenge data only).

Builds test S2/S3 index, streams test S1, writes:
  student_resource/output/candidate_pairs.tsv
  student_resource/output/matching_results.tsv

Uses logistic+corroboration model trained on train fit fold.
"""

from __future__ import annotations

import argparse
import pickle
import random
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from data.provenance import load_manifest, require_official_dataset
from official_core import (
    FEATURE_NAMES,
    SOURCE_H,
    accept_match,
    build_target_index_v2,
    feat_vec,
    generate_candidates_v2,
    iter_tsv,
    load_records,
    norm_addr,
    norm_name,
)
from run_official_bounded import GT_H

REPO = SRC.parents[2]


def train_model(dataset_root: Path, work_dir: Path, split_version: str, seed: int, fit_s1: int, max_candidates: int):
    """Train logistic on bounded fit S1s using train target index."""
    index_path = work_dir / "targets_index_v2.sqlite"
    split_db = work_dir / f"{split_version}_folds.sqlite"
    if not index_path.exists() or not split_db.exists():
        raise SystemExit("Run Stage4 first to build train index + frozen split")

    fold_db = sqlite3.connect(split_db)
    rng = random.Random(seed)
    rows = list(fold_db.execute("SELECT id FROM fold WHERE fold='fit' ORDER BY id"))
    rng.shuffle(rows)
    fit_ids = [r[0] for r in rows[:fit_s1]]
    needed = set(fit_ids)

    gt = {i: set() for i in fit_ids}
    for row in iter_tsv(dataset_root / "train" / "train_ground_truth.tsv", GT_H):
        sid = row["source1_entity_id"]
        if sid in needed:
            gt[sid] = {x for x in row["matched_entity_ids"].split(",") if x} if row["matched_entity_ids"] else set()

    s1_map = {}
    for row in iter_tsv(dataset_root / "train" / "train_source1.tsv", SOURCE_H):
        if row["entity_id"] in needed:
            s1_map[row["entity_id"]] = row

    idb = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    n_targets = idb.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
    df_cache: dict[str, int] = {}
    X_rows, y_rows = [], []
    for i, sid in enumerate(fit_ids):
        a = s1_map[sid]
        aa = {**a, "name_norm": norm_name(a["business_name"]), "addr_norm": norm_addr(a["business_address"])}
        cands = generate_candidates_v2(
            idb, a["business_name"], a["business_address"], a["country"],
            n_targets=n_targets, df_cache=df_cache, max_candidates=max_candidates,
        )
        trecs = load_records(idb, cands)
        truth = gt[sid]
        pos = [c for c in cands if c in truth]
        neg = [c for c in cands if c not in truth]
        rng.shuffle(neg)
        neg = neg[: max(8, 3 * max(len(pos), 1))]
        for tid in pos:
            if tid in trecs:
                X_rows.append(feat_vec(aa, trecs[tid]))
                y_rows.append(1)
        for tid in neg:
            if tid in trecs:
                X_rows.append(feat_vec(aa, trecs[tid]))
                y_rows.append(0)
        if (i + 1) % 1000 == 0:
            print(f"  fit pairs {i+1}/{len(fit_ids)}", flush=True)
    idb.close()
    fold_db.close()
    X = np.stack(X_rows)
    y = np.asarray(y_rows, dtype=np.int32)
    scaler = StandardScaler()
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed, solver="lbfgs")
    clf.fit(scaler.fit_transform(X), y)
    print(f"trained n={len(y)} pos={int(y.sum())}", flush=True)
    return scaler, clf


def main() -> None:
    p = argparse.ArgumentParser(description="Official test inference → submission TSVs")
    p.add_argument("--dataset-root", type=Path, default=REPO / "student_resource" / "dataset")
    p.add_argument("--work-dir", type=Path, default=REPO / "artifacts" / "official_index")
    p.add_argument("--output-dir", type=Path, default=REPO / "student_resource" / "output")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fit-s1", type=int, default=8000)
    p.add_argument("--max-candidates", type=int, default=120)
    p.add_argument("--threshold", type=float, default=0.85, help="Corroboration-gate model threshold")
    p.add_argument("--model-path", type=Path, default=None)
    p.add_argument("--skip-test-index", action="store_true")
    p.add_argument("--limit-s1", type=int, default=0, help="Debug: only first N test S1s (0=all)")
    args = p.parse_args()

    t_all = time.time()
    print("== provenance ==", flush=True)
    prov = require_official_dataset(args.dataset_root, require=("train", "test"), check_hashes=True)
    manifest = load_manifest()
    gt_hash = manifest["files"]["train/train_ground_truth.tsv"]["sha256"]
    split_version = f"official_{gt_hash[:12]}"
    provenance = f"official_challenge_dataset:{prov.manifest_version}"
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.model_path or (args.work_dir / "logistic_corroboration_v1.pkl")
    if model_path.exists():
        print(f"== loading model {model_path} ==", flush=True)
        with model_path.open("rb") as f:
            bundle = pickle.load(f)
        scaler, clf = bundle["scaler"], bundle["clf"]
        threshold = float(bundle.get("threshold", args.threshold))
        max_candidates = int(bundle.get("max_candidates", args.max_candidates))
        print(f"  threshold={threshold} max_candidates={max_candidates} "
              f"select_f05={bundle.get('select_macro_f05')} assess_f05={bundle.get('assess_macro_f05')}", flush=True)
    else:
        print("== training model on fit fold ==", flush=True)
        scaler, clf = train_model(
            args.dataset_root, args.work_dir, split_version, args.seed, args.fit_s1, args.max_candidates
        )
        threshold = args.threshold
        max_candidates = args.max_candidates
        with model_path.open("wb") as f:
            pickle.dump(
                {
                    "scaler": scaler,
                    "clf": clf,
                    "threshold": threshold,
                    "feature_names": FEATURE_NAMES,
                    "max_candidates": max_candidates,
                    "split_version": split_version,
                    "provenance": provenance,
                },
                f,
            )

    test_index = args.work_dir / "test_targets_index_v2.sqlite"
    if not args.skip_test_index and not test_index.exists():
        print("== building TEST S2/S3 index ==", flush=True)
        stats = build_target_index_v2(args.dataset_root, test_index, split="test")
        (args.work_dir / "test_index_stats.json").write_text(
            __import__("json").dumps(stats, indent=2) + "\n"
        )
        print(stats, flush=True)

    idb = sqlite3.connect(f"file:{test_index}?mode=ro", uri=True)
    n_targets = idb.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
    print(f"test target universe: {n_targets:,}", flush=True)
    df_cache: dict[str, int] = {}

    match_path = args.output_dir / "matching_results.tsv"
    cand_path = args.output_dir / "candidate_pairs.tsv"
    # Write atomically via temp then replace
    match_tmp = args.output_dir / "matching_results.tsv.partial"
    cand_tmp = args.output_dir / "candidate_pairs.tsv.partial"

    n_s1 = 0
    n_links = 0
    n_cands = 0
    t0 = time.time()
    with match_tmp.open("w", encoding="utf-8", newline="") as mf, cand_tmp.open(
        "w", encoding="utf-8", newline=""
    ) as cf:
        mf.write("source1_entity_id\tmatched_entity_ids\n")
        cf.write("source1_entity_id\tcandidate_entity_ids\n")

        for row in iter_tsv(args.dataset_root / "test" / "test_source1.tsv", SOURCE_H):
            sid = row["entity_id"]
            cands = generate_candidates_v2(
                idb,
                row["business_name"],
                row["business_address"],
                row["country"],
                n_targets=n_targets,
                df_cache=df_cache,
                max_candidates=max_candidates,
            )
            # dedupe preserve order
            seen = set()
            cands_u = []
            for c in cands:
                if c not in seen:
                    seen.add(c)
                    cands_u.append(c)
            cands = cands_u
            matches: list[str] = []
            if cands:
                trecs = load_records(idb, cands)
                aa = {
                    **row,
                    "name_norm": norm_name(row["business_name"]),
                    "addr_norm": norm_addr(row["business_address"]),
                }
                feats, tids = [], []
                for tid in cands:
                    b = trecs.get(tid)
                    if not b:
                        continue
                    feats.append(feat_vec(aa, b))
                    tids.append(tid)
                if feats:
                    pr = clf.predict_proba(scaler.transform(np.stack(feats)))[:, 1]
                    for tid, f, p in zip(tids, feats, pr):
                        if accept_match(float(p), f, threshold):
                            matches.append(tid)
            mf.write(f"{sid}\t{','.join(matches)}\n")
            cf.write(f"{sid}\t{','.join(cands)}\n")
            n_s1 += 1
            n_links += len(matches)
            n_cands += len(cands)
            if n_s1 % 2000 == 0:
                elapsed = time.time() - t0
                rate = n_s1 / max(elapsed, 1e-6)
                eta = (1732544 - n_s1) / max(rate, 1e-6)
                print(
                    f"  inferred {n_s1:,}  links={n_links:,}  "
                    f"avg_cands={n_cands/n_s1:.1f}  rate={rate:.1f}/s  eta={eta/3600:.2f}h",
                    flush=True,
                )
            if args.limit_s1 and n_s1 >= args.limit_s1:
                print(f"  limit-s1 reached ({args.limit_s1})", flush=True)
                break

    match_tmp.replace(match_path)
    cand_tmp.replace(cand_path)
    idb.close()
    meta = {
        "provenance": provenance,
        "split_version": split_version,
        "n_s1": n_s1,
        "n_links": n_links,
        "avg_candidates": n_cands / max(n_s1, 1),
        "threshold": threshold,
        "max_candidates": max_candidates,
        "runtime_sec": time.time() - t_all,
        "matching_path": str(match_path),
        "candidate_path": str(cand_path),
        "model_path": str(model_path),
        "limited": bool(args.limit_s1),
    }
    (args.output_dir / "inference_meta.json").write_text(
        __import__("json").dumps(meta, indent=2) + "\n"
    )
    print("== INFER DONE ==", meta, flush=True)


if __name__ == "__main__":
    main()
