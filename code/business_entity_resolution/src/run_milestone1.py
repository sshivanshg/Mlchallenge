"""First research milestone orchestrator (Phases 0–4)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from blocking.candidates import generate_candidates
from data.eda import run_eda
from data.load import read_ground_truth, read_sources, write_json
from evaluation.metric_ext import candidate_diagnostics, score_predictions
from evaluation.splits import freeze_split
from features.pairwise import FEATURE_NAMES, build_matrix
from models.matchers import (
    apply_threshold,
    empty_predict,
    exact_normalized_predict,
    train_lightgbm,
    train_logistic,
    tune_threshold,
    weighted_similarity_scores,
)

REPO = ROOT.parents[1]  # /workspace when installed at code/business_entity_resolution


def _records(df: pd.DataFrame) -> dict[str, dict]:
    return {r["entity_id"]: r for r in df.to_dict(orient="records")}


def _append_experiment(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment_id",
        "timestamp",
        "hypothesis",
        "validation_version",
        "blocking_config",
        "feature_config",
        "model",
        "model_params",
        "decision_rule",
        "threshold",
        "candidate_recall",
        "precision",
        "recall",
        "macro_f0.5",
        "singleton_accuracy",
        "avg_candidates",
        "runtime",
        "notes",
        "data_provenance",
    ]
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


def _pair_rows(
    s1_ids: list[str],
    candidates: dict[str, list[str]],
    truth: dict[str, set[str]],
    s1_map: dict[str, dict],
    tgt_map: dict[str, dict],
    *,
    include_missed_positives_for_train: bool = False,
) -> tuple[list[dict], list[int], list[str], list[str]]:
    pairs, labels, groups, cids = [], [], [], []
    for s1 in s1_ids:
        cands = list(candidates.get(s1, []))
        if include_missed_positives_for_train:
            for t in truth.get(s1, set()):
                if t not in cands and t in tgt_map:
                    cands.append(t)
        for rank, cid in enumerate(cands):
            if cid not in tgt_map or s1 not in s1_map:
                continue
            a, b = s1_map[s1], tgt_map[cid]
            # retrieval rank feature: 1/(1+rank)
            pairs.append(
                {
                    "name_a": a["business_name"],
                    "addr_a": a["business_address"],
                    "country_a": a["country"],
                    "name_b": b["business_name"],
                    "addr_b": b["business_address"],
                    "country_b": b["country"],
                    "retrieval_score": 1.0 / (1.0 + rank),
                    "retrieval_rank": float(rank),
                }
            )
            labels.append(1 if cid in truth.get(s1, set()) else 0)
            groups.append(s1)
            cids.append(cid)
    return pairs, labels, groups, cids


def _error_analysis(
    truth: dict[str, set[str]],
    pred: dict[str, set[str]],
    candidates: dict[str, list[str]],
    s1_map: dict[str, dict],
    tgt_map: dict[str, dict],
    out_path: Path,
) -> dict:
    cats = {
        "singleton_fp": [],
        "false_positive": [],
        "false_negative_blocking": [],
        "false_negative_threshold": [],
        "partial_miss": [],
    }
    for s1, tset in truth.items():
        pset = pred.get(s1, set())
        cset = set(candidates.get(s1, []))
        if not tset and pset:
            cats["singleton_fp"].append({"s1": s1, "pred": sorted(pset)})
        fp = pset - tset
        fn = tset - pset
        if fp and tset:
            cats["false_positive"].append({"s1": s1, "fp": sorted(fp), "truth": sorted(tset)})
        for m in fn:
            if m not in cset:
                cats["false_negative_blocking"].append({"s1": s1, "miss": m})
            else:
                cats["false_negative_threshold"].append({"s1": s1, "miss": m})
        if tset and pset and fn and not (not cset & tset):
            cats["partial_miss"].append({"s1": s1, "pred": sorted(pset), "truth": sorted(tset)})

    # Attach a few raw examples
    examples = []
    for bucket in ("singleton_fp", "false_positive", "false_negative_blocking", "false_negative_threshold"):
        for item in cats[bucket][:5]:
            s1 = item["s1"]
            a = s1_map.get(s1, {})
            examples.append(
                {
                    "category": bucket,
                    "s1": s1,
                    "s1_name": a.get("business_name"),
                    "s1_addr": a.get("business_address"),
                    "detail": item,
                }
            )
    summary = {k: len(v) for k, v in cats.items()}
    report = {"counts": summary, "examples": examples}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md = ["# Error analysis", "", "## Counts", ""]
    for k, v in summary.items():
        md.append(f"- **{k}**: {v}")
    md += ["", "## Examples", ""]
    for ex in examples[:12]:
        md.append(f"- `{ex['category']}` `{ex['s1']}` name=`{ex['s1_name']}` detail=`{ex['detail']}`")
    md.append("")
    out_path.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    return report


def run(dataset_root: Path, reports_root: Path, seed: int = 42) -> None:
    from data.provenance import require_official_dataset

    prov = require_official_dataset(dataset_root, require=("train", "test"), check_hashes=True)
    provenance = f"official_challenge_dataset:{prov.manifest_version}"
    print(
        f"DATA PROVENANCE: {provenance} "
        f"(verified {len(prov.checked_files)} files via schema+sha256 manifest)"
    )
    train_dir = dataset_root / "train"
    from data.provenance import sha256_file

    live = {}
    for rel in prov.checked_files:
        digest, nbytes = sha256_file(dataset_root / rel)
        live[rel] = {"sha256": digest, "nbytes": nbytes}
    write_json(
        {"provenance": provenance, "verified_files": live},
        reports_root / "eda" / "data_hashes.json",
    )

    # --- EDA ---
    eda = run_eda(train_dir, reports_root / "eda", provenance)
    print("EDA complete", eda["ground_truth"]["total_s1"], "S1")

    s1, s2, s3 = read_sources(train_dir, "train")
    gt = read_ground_truth(train_dir / "train_ground_truth.tsv")
    for sid in s1["entity_id"]:
        gt.setdefault(sid, set())
    countries = dict(zip(s1["entity_id"], s1["country"]))

    # --- Freeze split ---
    split_dir = REPO / "experiments" / "splits"
    split = freeze_split(gt, countries, split_dir, seed=seed, val_frac=0.25, version="v1")
    train_ids, val_ids = split["train_ids"], split["val_ids"]
    val_gt = {i: gt[i] for i in val_ids}
    print(f"Split v1: train={len(train_ids)} val={len(val_ids)}")

    # --- Candidates on full train pool (realistic target universe) ---
    t0 = time.time()
    candidates = generate_candidates(s1, s2, s3, max_candidates=60, use_tfidf=True)
    blocking_time = time.time() - t0
    # Val-only blocking diagnostics
    val_cands = {i: candidates[i] for i in val_ids}
    n_targets = len(s2) + len(s3)
    block_diag = candidate_diagnostics(val_cands, val_gt, n_targets)
    block_diag["runtime_sec_full_train_index"] = blocking_time
    write_json(block_diag, reports_root / "experiments" / "blocking_stats.json")
    (reports_root / "experiments" / "blocking_baseline.md").write_text(
        "\n".join(
            [
                "# Blocking baseline",
                "",
                f"**Provenance:** `{provenance}`",
                f"**Validation:** split_v1 ({len(val_ids)} S1)",
                "",
                f"- micro candidate recall: {block_diag['micro_candidate_recall']}",
                f"- S2 recall: {block_diag['s2_recall']}; S3 recall: {block_diag['s3_recall']}",
                f"- full-set coverage: {block_diag['full_set_coverage']}",
                f"- avg/median/p95/p99/max candidates: "
                f"{block_diag['avg_candidates']:.2f} / {block_diag['median_candidates']} / "
                f"{block_diag['p95_candidates']} / {block_diag['p99_candidates']} / {block_diag['max_candidates']}",
                f"- zero-candidate fraction: {block_diag['zero_candidate_frac']:.3f}",
                f"- reduction ratio: {block_diag['reduction_ratio']:.6f}",
                f"- oracle macro F0.5 (perfect matcher on candidates): {block_diag['oracle_macro_F0.5']:.4f}",
                f"- true matches missed by blocking: {block_diag['true_matches_missed']}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    s1_map, tgt_map = _records(s1), _records(pd.concat([s2, s3], ignore_index=True))
    exp_csv = reports_root / "experiments" / "experiments.csv"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def log(exp_id, hypothesis, model, metrics, **kwargs):
        _append_experiment(
            exp_csv,
            {
                "experiment_id": exp_id,
                "timestamp": ts,
                "hypothesis": hypothesis,
                "validation_version": "split_v1",
                "blocking_config": kwargs.get("blocking_config", "union:exact+prefix+token+num+tfidf_char"),
                "feature_config": kwargs.get("feature_config", ""),
                "model": model,
                "model_params": kwargs.get("model_params", ""),
                "decision_rule": kwargs.get("decision_rule", "global_threshold"),
                "threshold": kwargs.get("threshold", ""),
                "candidate_recall": block_diag["micro_candidate_recall"],
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "macro_f0.5": metrics.get("macro_F0.5"),
                "singleton_accuracy": metrics.get("singleton_accuracy"),
                "avg_candidates": block_diag["avg_candidates"],
                "runtime": kwargs.get("runtime", ""),
                "notes": kwargs.get("notes", ""),
                "data_provenance": provenance,
            },
        )
        print(
            f"{exp_id}: macro_F0.5={metrics.get('macro_F0.5'):.4f} "
            f"P={metrics.get('precision')} R={metrics.get('recall')} "
            f"singleton={metrics.get('singleton_accuracy')}"
        )

    # --- Baseline A: empty ---
    t0 = time.time()
    pred_a = empty_predict(val_ids)
    met_a = score_predictions(val_gt, pred_a)
    log("B0_empty", "All-empty abstention floor", "empty", met_a, runtime=time.time() - t0, threshold="")

    # --- Baseline B: exact normalized ---
    t0 = time.time()
    pred_b = exact_normalized_predict(s1_map, tgt_map, candidates=None)
    pred_b = {i: pred_b.get(i, set()) for i in val_ids}
    met_b = score_predictions(val_gt, pred_b)
    log(
        "B1_exact_norm",
        "Normalized exact name+address/country",
        "exact_normalized",
        met_b,
        runtime=time.time() - t0,
        blocking_config="none_full_scan_exact_index",
        feature_config="name_legal+addr_norm",
    )

    # --- Baseline C: retrieval oracle upper bound already in blocking; also exact on candidates ---
    t0 = time.time()
    pred_c = exact_normalized_predict(s1_map, tgt_map, candidates=candidates)
    pred_c = {i: pred_c.get(i, set()) for i in val_ids}
    met_c = score_predictions(val_gt, pred_c)
    log(
        "B2_exact_on_candidates",
        "Exact normalized restricted to retrieval candidates",
        "exact_on_candidates",
        met_c,
        runtime=time.time() - t0,
        feature_config="name_legal+addr_norm",
    )

    # --- Build hard-negative training pairs from train fold candidates ---
    train_pairs, train_y, _, _ = _pair_rows(
        train_ids, candidates, gt, s1_map, tgt_map, include_missed_positives_for_train=True
    )
    val_pairs, val_y, val_groups, val_cids = _pair_rows(
        val_ids, candidates, gt, s1_map, tgt_map, include_missed_positives_for_train=False
    )
    Xtr, ytr = build_matrix(train_pairs), np.asarray(train_y, dtype=np.int32)
    Xva = build_matrix(val_pairs)
    print(f"Train pairs={len(ytr)} positives={int(ytr.sum())} | Val pairs={len(val_y)} pos={sum(val_y)}")

    # --- Weighted similarity ---
    t0 = time.time()
    scores_w = weighted_similarity_scores(Xva)
    thr_w, met_w = tune_threshold(val_groups, val_cids, scores_w, val_gt)
    pred_w = apply_threshold(val_groups, val_cids, scores_w, val_ids, thr_w)
    log(
        "M0_weighted",
        "Deterministic weighted string similarities on candidates",
        "weighted_similarity",
        met_w,
        runtime=time.time() - t0,
        threshold=thr_w,
        feature_config=",".join(FEATURE_NAMES),
        decision_rule="global_threshold_on_weighted_score",
    )

    # --- Logistic regression ---
    t0 = time.time()
    if len(set(ytr.tolist())) < 2:
        print("WARNING: train labels single-class; skipping logistic/lgbm fit quality")
        met_lr = met_w
        thr_lr = thr_w
        pred_lr = pred_w
        bundle_lr = None
    else:
        bundle_lr = train_logistic(Xtr, ytr, seed=seed)
        scores_lr = bundle_lr.predict_proba(Xva)
        thr_lr, met_lr = tune_threshold(val_groups, val_cids, scores_lr, val_gt)
        pred_lr = apply_threshold(val_groups, val_cids, scores_lr, val_ids, thr_lr)
    log(
        "M1_logistic",
        "Logistic regression on interpretable pair features; hard negatives from blockers",
        "logistic_regression",
        met_lr,
        runtime=time.time() - t0,
        threshold=thr_lr,
        feature_config=",".join(FEATURE_NAMES),
        model_params="sklearn LogisticRegression class_weight=balanced",
    )

    # --- LightGBM ---
    t0 = time.time()
    if len(set(ytr.tolist())) < 2:
        met_lgb, thr_lgb, pred_lgb = met_lr, thr_lr, pred_lr
        bundle_lgb = None
    else:
        bundle_lgb = train_lightgbm(Xtr, ytr, seed=seed)
        scores_lgb = bundle_lgb.predict_proba(Xva)
        thr_lgb, met_lgb = tune_threshold(val_groups, val_cids, scores_lgb, val_gt)
        pred_lgb = apply_threshold(val_groups, val_cids, scores_lgb, val_ids, thr_lgb)
    log(
        "M2_lightgbm",
        "LightGBM on same features/hard negatives; threshold max macro F0.5",
        "lightgbm",
        met_lgb,
        runtime=time.time() - t0,
        threshold=thr_lgb,
        feature_config=",".join(FEATURE_NAMES),
        model_params="LGBMClassifier n_estimators=120 lr=0.08",
    )

    # Pick best by macro F0.5
    ranked = [
        ("B0_empty", met_a, pred_a, None),
        ("B1_exact_norm", met_b, pred_b, None),
        ("B2_exact_on_candidates", met_c, pred_c, None),
        ("M0_weighted", met_w, pred_w, thr_w),
        ("M1_logistic", met_lr, pred_lr, thr_lr),
        ("M2_lightgbm", met_lgb, pred_lgb, thr_lgb),
    ]
    best = max(ranked, key=lambda x: x[1]["macro_F0.5"])
    print(f"BEST so far: {best[0]} macro_F0.5={best[1]['macro_F0.5']:.4f}")

    err = _error_analysis(
        val_gt, best[2], val_cands, s1_map, tgt_map, reports_root / "error_analysis" / "milestone1_errors.json"
    )

    # Threshold curve for best learned model (lgb if available)
    curve_scores = None
    curve_model = "M2_lightgbm"
    if bundle_lgb is not None:
        curve_scores = bundle_lgb.predict_proba(Xva)
    elif bundle_lr is not None:
        curve_scores = bundle_lr.predict_proba(Xva)
        curve_model = "M1_logistic"
    if curve_scores is not None:
        rows = []
        for t in np.linspace(0.1, 0.95, 18):
            preds = apply_threshold(val_groups, val_cids, curve_scores, val_ids, float(t))
            m = score_predictions(val_gt, preds)
            rows.append(
                {
                    "threshold": float(t),
                    "macro_F0.5": m["macro_F0.5"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "singleton_accuracy": m["singleton_accuracy"],
                    "predicted_links": int(sum(len(v) for v in preds.values())),
                }
            )
        write_json({"model": curve_model, "curve": rows}, reports_root / "experiments" / "threshold_curve.json")

    summary = {
        "provenance": provenance,
        "hashes": hashes,
        "split": {"version": "v1", "n_train": len(train_ids), "n_val": len(val_ids), "seed": seed},
        "blocking": block_diag,
        "baselines": {k: v for k, v, _, _ in ranked},
        "best_experiment": best[0],
        "best_metrics": best[1],
        "error_counts": err["counts"],
        "eda_singleton_rate": eda["ground_truth"]["singleton_rate"],
    }
    write_json(summary, reports_root / "experiments" / "milestone1_summary.json")
    print("Wrote reports under", reports_root)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=REPO / "student_resource" / "dataset")
    p.add_argument("--reports-root", type=Path, default=REPO / "reports")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    # Competition entrypoint: never generate synthetic data here.
    run(args.dataset_root, args.reports_root, seed=args.seed)


if __name__ == "__main__":
    main()
