"""E4: precision corroboration gate on Stage-4 logistic (official data only).

Reuses targets_index_v2 and frozen folds. Reports select + untouched assess.
"""

from __future__ import annotations

import argparse
import json
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
from evaluation.metric_ext import candidate_diagnostics, score_predictions
from official_core import (
    FEATURE_NAMES,
    SOURCE_H,
    accept_match,
    feat_vec,
    generate_candidates_v2,
    iter_tsv,
    load_records,
    norm_addr,
    norm_name,
)
from run_official_bounded import GT_H, append_exp

REPO = SRC.parents[2]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=REPO / "student_resource" / "dataset")
    p.add_argument("--work-dir", type=Path, default=REPO / "artifacts" / "official_index")
    p.add_argument("--reports-root", type=Path, default=REPO / "reports" / "official")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-s1", type=int, default=5000)
    p.add_argument("--fit-s1", type=int, default=8000)
    p.add_argument("--assess-s1", type=int, default=3000)
    p.add_argument("--max-candidates", type=int, default=120)
    p.add_argument("--threshold", type=float, default=0.95)
    args = p.parse_args()

    t_all = time.time()
    prov = require_official_dataset(args.dataset_root, require=("train", "test"), check_hashes=True)
    manifest = load_manifest()
    gt_hash = manifest["files"]["train/train_ground_truth.tsv"]["sha256"]
    split_version = f"official_{gt_hash[:12]}"
    provenance = f"official_challenge_dataset:{prov.manifest_version}"
    reports = args.reports_root / split_version

    split_db = args.work_dir / f"{split_version}_folds.sqlite"
    index_path = args.work_dir / "targets_index_v2.sqlite"
    if not split_db.exists() or not index_path.exists():
        raise SystemExit("Need frozen split + targets_index_v2 from Stage4")

    fold_db = sqlite3.connect(split_db)
    rng = random.Random(args.seed)

    def sample_fold(fold: str, n: int) -> list[str]:
        rows = list(fold_db.execute("SELECT id FROM fold WHERE fold=? ORDER BY id", (fold,)))
        rng.shuffle(rows)
        return [r[0] for r in rows[: min(n, len(rows))]]

    select_ids = sample_fold("select", args.eval_s1)
    fit_ids = sample_fold("fit", args.fit_s1)
    assess_ids = sample_fold("assess", args.assess_s1)
    needed = set(select_ids) | set(fit_ids) | set(assess_ids)

    gt = {i: set() for i in needed}
    for row in iter_tsv(args.dataset_root / "train" / "train_ground_truth.tsv", GT_H):
        sid = row["source1_entity_id"]
        if sid in needed:
            gt[sid] = {x for x in row["matched_entity_ids"].split(",") if x} if row["matched_entity_ids"] else set()

    s1_map = {}
    for row in iter_tsv(args.dataset_root / "train" / "train_source1.tsv", SOURCE_H):
        if row["entity_id"] in needed:
            s1_map[row["entity_id"]] = row

    idb = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    n_targets = idb.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
    df_cache: dict[str, int] = {}
    trec_cache: dict[str, dict] = {}

    def gen(ids):
        out = {}
        for i, sid in enumerate(ids):
            r = s1_map[sid]
            out[sid] = generate_candidates_v2(
                idb, r["business_name"], r["business_address"], r["country"],
                n_targets=n_targets, df_cache=df_cache, max_candidates=args.max_candidates,
            )
            if (i + 1) % 1000 == 0:
                print(f"  cands {i+1}/{len(ids)}", flush=True)
        return out

    print("== generate fit/select/assess candidates ==", flush=True)
    fit_c = gen(fit_ids)
    sel_c = gen(select_ids)
    ass_c = gen(assess_ids)

    # train logistic
    X_rows, y_rows = [], []
    for sid in fit_ids:
        a = s1_map[sid]
        aa = {**a, "name_norm": norm_name(a["business_name"]), "addr_norm": norm_addr(a["business_address"])}
        truth = gt[sid]
        miss = [c for c in fit_c[sid] if c not in trec_cache]
        if miss:
            trec_cache.update(load_records(idb, miss))
        pos = [c for c in fit_c[sid] if c in truth]
        neg = [c for c in fit_c[sid] if c not in truth]
        rng.shuffle(neg)
        neg = neg[: max(8, 3 * max(len(pos), 1))]
        for tid in pos + neg:
            b = trec_cache.get(tid)
            if not b:
                continue
            X_rows.append(feat_vec(aa, b))
            y_rows.append(1 if tid in truth else 0)
    X = np.stack(X_rows)
    y = np.asarray(y_rows, dtype=np.int32)
    scaler = StandardScaler()
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed, solver="lbfgs")
    clf.fit(scaler.fit_transform(X), y)
    print(f"trained logistic n={len(y)} pos={int(y.sum())}", flush=True)

    def predict(ids, cands, thr, use_gate: bool):
        preds = {i: set() for i in ids}
        for sid in ids:
            a = s1_map[sid]
            aa = {**a, "name_norm": norm_name(a["business_name"]), "addr_norm": norm_addr(a["business_address"])}
            miss = [c for c in cands[sid] if c not in trec_cache]
            if miss:
                trec_cache.update(load_records(idb, miss))
            feats, tids = [], []
            for tid in cands[sid]:
                b = trec_cache.get(tid)
                if not b:
                    continue
                feats.append(feat_vec(aa, b))
                tids.append(tid)
            if not feats:
                continue
            pr = clf.predict_proba(scaler.transform(np.stack(feats)))[:, 1]
            for tid, f, p in zip(tids, feats, pr):
                if use_gate:
                    if accept_match(float(p), f, thr):
                        preds[sid].add(tid)
                elif p >= thr:
                    preds[sid].add(tid)
        return preds

    # tune thr on select with gate
    print("== tune corroboration-gated threshold on select ==", flush=True)
    select_gt = {i: gt[i] for i in select_ids}
    best_t, best_m, best_pred = args.threshold, None, None
    for thr in np.linspace(0.70, 0.99, 30):
        pred = predict(select_ids, sel_c, float(thr), use_gate=True)
        m = score_predictions(select_gt, pred)
        if best_m is None or m["macro_F0.5"] > best_m["macro_F0.5"]:
            best_t, best_m, best_pred = float(thr), m, pred
    print(f"E4 select gated best thr={best_t} F={best_m['macro_F0.5']:.4f} sing={best_m['singleton_accuracy']:.4f}", flush=True)

    # baseline ungated at 0.95 for comparison on same ids
    base = predict(select_ids, sel_c, 0.95, use_gate=False)
    m_base = score_predictions(select_gt, base)

    assess_gt = {i: gt[i] for i in assess_ids}
    pred_a = predict(assess_ids, ass_c, best_t, use_gate=True)
    m_assess = score_predictions(assess_gt, pred_a)
    a_block = candidate_diagnostics(ass_c, assess_gt, n_targets)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    exp_csv = reports / "experiments.csv"
    for eid, hyp, metrics, fold, thr, notes in [
        ("E4_logistic_base_select", "Stage4 logistic thr=0.95 ungated (rerun)", m_base, "select_bounded", 0.95, "selection"),
        ("E4_logistic_corroboration_select", "Logistic + name/addr corroboration gate; thr on select", best_m, "select_bounded", best_t, "selection-fold"),
        ("E4_logistic_corroboration_assess", "Untouched assess with frozen corroboration gate+thr", m_assess, "assess_bounded", best_t, "UNTOUCHED assessment"),
    ]:
        append_exp(exp_csv, {
            "experiment_id": eid,
            "timestamp": ts,
            "hypothesis": hyp,
            "validation_version": split_version,
            "blocking_config": "v2_idf_sig_accent",
            "feature_config": ",".join(FEATURE_NAMES) + "+corroboration_gate",
            "model": "logistic+corroboration",
            "decision_rule": "frozen_threshold_from_select" if "assess" in eid else "global_threshold_on_select",
            "threshold": thr,
            "candidate_recall": a_block["micro_candidate_recall"] if "assess" in eid else "",
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "macro_f0.5": metrics.get("macro_F0.5"),
            "singleton_accuracy": metrics.get("singleton_accuracy"),
            "avg_candidates": "",
            "runtime": "",
            "notes": notes,
            "data_provenance": provenance,
            "fold": fold,
        })
        print(f"{eid}: F0.5={metrics['macro_F0.5']:.4f} P={metrics.get('precision')} R={metrics.get('recall')} sing={metrics.get('singleton_accuracy')}", flush=True)

    summary = {
        "provenance": provenance,
        "split_version": split_version,
        "stage": "4e",
        "E4_base_select": m_base,
        "E4_corroboration_select": {**best_m, "threshold": best_t},
        "E4_corroboration_assess": {**m_assess, "threshold": best_t},
        "runtime_total_sec": time.time() - t_all,
    }
    (reports / "stage4e_corroboration.json").write_text(json.dumps(summary, indent=2) + "\n")

    # persist model for test inference
    import pickle
    model_path = args.work_dir / "logistic_corroboration_v1.pkl"
    with model_path.open("wb") as f:
        pickle.dump(
            {
                "scaler": scaler,
                "clf": clf,
                "threshold": best_t,
                "feature_names": FEATURE_NAMES,
                "max_candidates": args.max_candidates,
                "split_version": split_version,
                "provenance": provenance,
                "select_macro_f05": best_m["macro_F0.5"],
                "assess_macro_f05": m_assess["macro_F0.5"],
            },
            f,
        )
    print(f"saved model {model_path}", flush=True)
    print("== E4 DONE ==", json.dumps(summary, indent=2), flush=True)
    idb.close()
    fold_db.close()


if __name__ == "__main__":
    main()
