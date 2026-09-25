"""End-to-end business entity resolution pipeline (JAX matcher)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from jax import random

from blocking import blocking_recall, generate_candidates
from features import FEATURE_NAMES, build_feature_matrix
from io_utils import read_ground_truth, read_sources, write_id_list_tsv
from metrics import macro_f05
from model import params_from_numpy, params_to_numpy, score_pairs, train_logistic, tune_threshold

ROOT = Path(__file__).resolve().parent


def _records_by_id(df: pd.DataFrame) -> dict[str, dict]:
    return {r["entity_id"]: r for r in df.to_dict(orient="records")}


def _group_split_s1(s1_ids: list[str], val_frac: float, seed: int) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed)
    ids = list(s1_ids)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_frac))
    return ids[n_val:], ids[:n_val]


def _pair_dataset(
    s1_ids: list[str],
    candidates: dict[str, list[str]],
    ground_truth: dict[str, set[str]],
    s1_map: dict[str, dict],
    tgt_map: dict[str, dict],
    neg_per_pos: int = 4,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    rng = np.random.default_rng(seed)
    pairs = []
    labels = []
    groups = []
    cand_ids = []
    for s1 in s1_ids:
        truth = ground_truth.get(s1, set())
        cands = candidates.get(s1, [])
        pos = [c for c in cands if c in truth]
        neg = [c for c in cands if c not in truth]
        if not pos and truth:
            # still include true matches even if blocking missed (train only signal)
            pos = [c for c in truth if c in tgt_map]
        chosen_neg = list(neg)
        if len(chosen_neg) > max(neg_per_pos * max(1, len(pos)), neg_per_pos):
            chosen_neg = list(rng.choice(chosen_neg, size=max(neg_per_pos * max(1, len(pos)), neg_per_pos), replace=False))
        for cid in pos + chosen_neg:
            if s1 not in s1_map or cid not in tgt_map:
                continue
            pairs.append((s1_map[s1], tgt_map[cid]))
            labels.append(1 if cid in truth else 0)
            groups.append(s1)
            cand_ids.append(cid)
    x = build_feature_matrix(pairs)
    y = np.asarray(labels, dtype=np.float32)
    return x, y, groups, cand_ids


def train_and_tune(
    train_dir: Path,
    model_dir: Path,
    val_frac: float = 0.2,
    seed: int = 42,
    max_candidates: int = 80,
) -> dict:
    s1, s2, s3 = read_sources(train_dir, "train")
    gt = read_ground_truth(train_dir / "train_ground_truth.tsv")
    # Ensure every S1 appears in GT map (singletons)
    for eid in s1["entity_id"]:
        gt.setdefault(eid, set())

    candidates = generate_candidates(s1, s2, s3, max_candidates=max_candidates)
    br = blocking_recall(candidates, gt)
    print(f"blocking recall ceiling (train): {br:.4f}")

    train_ids, val_ids = _group_split_s1(list(s1["entity_id"]), val_frac, seed)
    s1_map = _records_by_id(s1)
    tgt_map = _records_by_id(pd.concat([s2, s3], ignore_index=True))

    x_tr, y_tr, _, _ = _pair_dataset(train_ids, candidates, gt, s1_map, tgt_map, seed=seed)
    x_va, y_va, g_va, c_va = _pair_dataset(val_ids, candidates, gt, s1_map, tgt_map, seed=seed + 1)

    key = random.PRNGKey(seed)
    key, train_key = random.split(key)
    params = train_logistic(x_tr, y_tr, train_key, epochs=50, batch_size=128, lr=0.08)
    scores = score_pairs(params, x_va).tolist()
    val_gt = {s: gt[s] for s in val_ids}
    threshold, val_f = tune_threshold(scores, y_va.tolist(), g_va, val_gt, c_va, macro_f05)
    print(f"validation macro F0.5: {val_f:.4f} @ threshold={threshold:.3f}")

    model_dir.mkdir(parents=True, exist_ok=True)
    blob = params_to_numpy(params)
    np.savez(model_dir / "jax_logistic.npz", **blob)
    meta = {
        "threshold": threshold,
        "val_macro_f05": val_f,
        "blocking_recall_train": br,
        "feature_names": FEATURE_NAMES,
        "max_candidates": max_candidates,
        "seed": seed,
        "n_train_pairs": int(len(y_tr)),
        "n_val_pairs": int(len(y_va)),
        "model_license": "Original code MIT; JAX Apache-2.0; no pretrained >8B models used",
    }
    (model_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def predict_split(
    data_dir: Path,
    split: str,
    model_dir: Path,
    output_dir: Path,
    max_candidates: int | None = None,
) -> None:
    s1, s2, s3 = read_sources(data_dir, split)
    meta = json.loads((model_dir / "meta.json").read_text())
    threshold = float(meta["threshold"])
    max_cand = int(max_candidates or meta.get("max_candidates", 80))
    blob = np.load(model_dir / "jax_logistic.npz")
    params = params_from_numpy({k: blob[k] for k in blob.files})

    candidates = generate_candidates(s1, s2, s3, max_candidates=max_cand)
    s1_map = _records_by_id(s1)
    tgt_map = _records_by_id(pd.concat([s2, s3], ignore_index=True))

    matches: dict[str, list[str]] = {eid: [] for eid in s1["entity_id"]}
    for s1_id, cands in candidates.items():
        if not cands:
            continue
        pairs = [(s1_map[s1_id], tgt_map[c]) for c in cands if c in tgt_map]
        kept = [c for c in cands if c in tgt_map]
        x = build_feature_matrix(pairs)
        probs = score_pairs(params, x)
        for cid, p in zip(kept, probs):
            if p >= threshold:
                matches[s1_id].append(cid)

    s1_ids = list(s1["entity_id"])
    write_id_list_tsv(
        output_dir / "matching_results.tsv",
        s1_ids,
        matches,
        "source1_entity_id",
        "matched_entity_ids",
    )
    write_id_list_tsv(
        output_dir / "candidate_pairs.tsv",
        s1_ids,
        candidates,
        "source1_entity_id",
        "candidate_entity_ids",
    )
    print(f"Wrote outputs to {output_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="JAX business entity resolution pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train JAX matcher + tune F0.5 threshold")
    p_train.add_argument("--train-dir", type=Path, required=True)
    p_train.add_argument("--model-dir", type=Path, required=True)
    p_train.add_argument("--val-frac", type=float, default=0.2)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--max-candidates", type=int, default=80)

    p_pred = sub.add_parser("predict", help="Run blocking + matching on a split")
    p_pred.add_argument("--data-dir", type=Path, required=True)
    p_pred.add_argument("--split", choices=["train", "test"], default="test")
    p_pred.add_argument("--model-dir", type=Path, required=True)
    p_pred.add_argument("--output-dir", type=Path, required=True)
    p_pred.add_argument("--max-candidates", type=int, default=None)

    p_all = sub.add_parser("run", help="Train on train/ then predict test/")
    p_all.add_argument("--dataset-root", type=Path, required=True)
    p_all.add_argument("--model-dir", type=Path, required=True)
    p_all.add_argument("--output-dir", type=Path, required=True)
    p_all.add_argument("--seed", type=int, default=42)
    p_all.add_argument("--max-candidates", type=int, default=80)

    args = parser.parse_args(argv)
    if args.cmd == "train":
        train_and_tune(args.train_dir, args.model_dir, args.val_frac, args.seed, args.max_candidates)
    elif args.cmd == "predict":
        predict_split(args.data_dir, args.split, args.model_dir, args.output_dir, args.max_candidates)
    elif args.cmd == "run":
        train_and_tune(
            args.dataset_root / "train",
            args.model_dir,
            seed=args.seed,
            max_candidates=args.max_candidates,
        )
        predict_split(
            args.dataset_root / "test",
            "test",
            args.model_dir,
            args.output_dir,
            args.max_candidates,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
