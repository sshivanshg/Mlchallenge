"""Matchers: empty, exact, weighted, logistic, LightGBM + threshold search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from evaluation.metric_ext import score_predictions
from features.pairwise import FEATURE_NAMES, build_matrix, pair_features
from normalization.views import view_address, view_name_legal


def empty_predict(s1_ids: list[str]) -> dict[str, set[str]]:
    return {s: set() for s in s1_ids}


def exact_normalized_predict(
    s1_records: dict[str, dict],
    tgt_records: dict[str, dict],
    candidates: dict[str, list[str]] | None = None,
) -> dict[str, set[str]]:
    """Match when name_legal equal and (addr equal OR both addr empty not rewarded)."""
    # Index targets by name_legal
    by_name: dict[str, list[str]] = {}
    for tid, rec in tgt_records.items():
        key = view_name_legal(rec["business_name"])
        if not key:
            continue
        by_name.setdefault(key, []).append(tid)

    preds: dict[str, set[str]] = {s: set() for s in s1_records}
    for sid, rec in s1_records.items():
        key = view_name_legal(rec["business_name"])
        if not key:
            continue
        pool = by_name.get(key, [])
        if candidates is not None:
            cset = set(candidates.get(sid, []))
            pool = [t for t in pool if t in cset]
        sa = view_address(rec["business_address"])
        for tid in pool:
            ta = view_address(tgt_records[tid]["business_address"])
            if sa and ta and sa == ta:
                preds[sid].add(tid)
            elif sa and ta and sa != ta:
                continue
            else:
                # name exact only — conservative: require also same country if both present
                ca, cb = rec.get("country", ""), tgt_records[tid].get("country", "")
                if ca and cb and ca.strip().casefold() == cb.strip().casefold():
                    preds[sid].add(tid)
    return preds


def weighted_similarity_scores(X: np.ndarray) -> np.ndarray:
    """Deterministic weighted blend of high-signal features (no training)."""
    name = FEATURE_NAMES.index("name_token_sort")
    addr = FEATURE_NAMES.index("addr_token_sort")
    jaro = FEATURE_NAMES.index("name_jaro")
    num = FEATURE_NAMES.index("num_overlap")
    country = FEATURE_NAMES.index("country_equal")
    exact = FEATURE_NAMES.index("name_exact")
    w = np.zeros(X.shape[1], dtype=np.float32)
    w[name] = 0.35
    w[jaro] = 0.20
    w[addr] = 0.20
    w[num] = 0.10
    w[country] = 0.05
    w[exact] = 0.10
    return np.clip(X @ w, 0, 1)


@dataclass
class SklearnBundle:
    name: str
    scaler: StandardScaler | None
    model: object

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X) if self.scaler is not None else X
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(Xs)[:, 1]
        return self.model.predict(Xs)


def train_logistic(X: np.ndarray, y: np.ndarray, seed: int = 42) -> SklearnBundle:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=seed,
        solver="lbfgs",
    )
    clf.fit(Xs, y)
    return SklearnBundle("logistic", scaler, clf)


def train_lightgbm(X: np.ndarray, y: np.ndarray, seed: int = 42) -> SklearnBundle:
    import lightgbm as lgb

    clf = lgb.LGBMClassifier(
        n_estimators=120,
        learning_rate=0.08,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        verbose=-1,
    )
    clf.fit(X, y)
    # Ensure numpy matrix inference does not warn about feature names
    if hasattr(clf, "feature_names_in_"):
        delattr(clf, "feature_names_in_")
    return SklearnBundle("lightgbm", None, clf)


def tune_threshold(
    group_ids: list[str],
    cand_ids: list[str],
    scores: np.ndarray,
    truth: dict[str, set[str]],
    grid: np.ndarray | None = None,
) -> tuple[float, dict]:
    if grid is None:
        grid = np.unique(
            np.concatenate(
                [
                    np.linspace(0.2, 0.95, 31),
                    np.quantile(scores, np.linspace(0.05, 0.95, 19)) if len(scores) else [0.5],
                ]
            )
        )
    best_t, best = 0.5, None
    for t in grid:
        preds = {s: set() for s in truth}
        for g, c, s in zip(group_ids, cand_ids, scores):
            if s >= t:
                preds[g].add(c)
        metrics = score_predictions(truth, preds)
        if best is None or metrics["macro_F0.5"] > best["macro_F0.5"]:
            best_t, best = float(t), metrics
    assert best is not None
    best = dict(best)
    best["threshold"] = best_t
    return best_t, best


def apply_threshold(
    group_ids: list[str],
    cand_ids: list[str],
    scores: np.ndarray,
    s1_ids: list[str],
    threshold: float,
) -> dict[str, set[str]]:
    preds = {s: set() for s in s1_ids}
    for g, c, s in zip(group_ids, cand_ids, scores):
        if s >= threshold:
            preds[g].add(c)
    return preds
