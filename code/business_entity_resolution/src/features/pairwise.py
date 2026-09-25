"""Interpretable pairwise features."""

from __future__ import annotations

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from normalization.views import (
    country_key,
    numeric_tokens,
    significant_tokens,
    view_address,
    view_name_legal,
)

FEATURE_NAMES = [
    "name_exact",
    "addr_exact",
    "name_jaro",
    "name_ratio",
    "name_token_sort",
    "name_token_set",
    "name_partial",
    "name_jaccard",
    "name_contain_ab",
    "name_contain_ba",
    "addr_jaro",
    "addr_ratio",
    "addr_token_sort",
    "addr_jaccard",
    "num_overlap",
    "num_conflict",
    "country_equal",
    "country_missing",
    "name_missing",
    "addr_missing",
    "name_len_diff",
    "addr_len_diff",
    "retrieval_score",
    "retrieval_rank",
    "name_addr_agree",
]


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0  # both empty ≠ evidence
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _contain(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa)


def pair_features(
    name_a: str,
    addr_a: str,
    country_a: str,
    name_b: str,
    addr_b: str,
    country_b: str,
    retrieval_score: float = 0.0,
    retrieval_rank: float = 0.0,
) -> np.ndarray:
    na, nb = view_name_legal(name_a), view_name_legal(name_b)
    aa, ab = view_address(addr_a), view_address(addr_b)
    ta, tb = significant_tokens(na), significant_tokens(nb)
    ra, rb = significant_tokens(aa), significant_tokens(ab)
    nums_a, nums_b = set(numeric_tokens(aa)), set(numeric_tokens(ab))
    if nums_a and nums_b:
        num_ov = len(nums_a & nums_b) / len(nums_a | nums_b)
        num_conflict = float(len(nums_a & nums_b) == 0)
    else:
        num_ov, num_conflict = 0.0, 0.0
    ca, cb = country_key(country_a), country_key(country_b)
    country_eq = 1.0 if ca and cb and ca == cb else 0.0
    country_miss = 1.0 if (not ca or not cb) else 0.0
    name_miss = 1.0 if (not na or not nb) else 0.0
    addr_miss = 1.0 if (not aa or not ab) else 0.0
    name_sim = fuzz.token_sort_ratio(na, nb) / 100.0 if (na or nb) else 0.0
    addr_sim = fuzz.token_sort_ratio(aa, ab) / 100.0 if (aa or ab) else 0.0
    agree = 1.0 if (name_sim >= 0.85 and addr_sim >= 0.85) else (
        -1.0 if (name_sim >= 0.85 and addr_sim > 0 and addr_sim < 0.5) else 0.0
    )
    feats = [
        1.0 if na and nb and na == nb else 0.0,
        1.0 if aa and ab and aa == ab else 0.0,
        JaroWinkler.normalized_similarity(na, nb) if (na or nb) else 0.0,
        fuzz.ratio(na, nb) / 100.0,
        name_sim,
        fuzz.token_set_ratio(na, nb) / 100.0,
        fuzz.partial_ratio(na, nb) / 100.0,
        _jaccard(ta, tb),
        _contain(ta, tb),
        _contain(tb, ta),
        JaroWinkler.normalized_similarity(aa, ab) if (aa or ab) else 0.0,
        fuzz.ratio(aa, ab) / 100.0,
        addr_sim,
        _jaccard(ra, rb),
        num_ov,
        num_conflict,
        country_eq,
        country_miss,
        name_miss,
        addr_miss,
        abs(len(na) - len(nb)) / max(len(na), len(nb), 1),
        abs(len(aa) - len(ab)) / max(len(aa), len(ab), 1),
        float(retrieval_score),
        float(retrieval_rank),
        agree,
    ]
    return np.asarray(feats, dtype=np.float32)


def build_matrix(pairs: list[dict]) -> np.ndarray:
    if not pairs:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    rows = [
        pair_features(
            p["name_a"],
            p["addr_a"],
            p["country_a"],
            p["name_b"],
            p["addr_b"],
            p["country_b"],
            p.get("retrieval_score", 0.0),
            p.get("retrieval_rank", 0.0),
        )
        for p in pairs
    ]
    return np.stack(rows, axis=0)
