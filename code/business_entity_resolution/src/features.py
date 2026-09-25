"""Pairwise similarity features — pure functions over strings."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from normalize import country_key, normalize_text, numeric_tokens, tokenize

FEATURE_NAMES = [
    "name_jaccard",
    "name_jaro",
    "name_token_sort",
    "name_partial",
    "addr_jaccard",
    "addr_jaro",
    "addr_token_sort",
    "num_overlap",
    "country_equal",
    "name_len_ratio",
    "addr_len_ratio",
]


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _len_ratio(a: str, b: str) -> float:
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0
    return min(la, lb) / max(la, lb)


def pair_features(
    name_a: str,
    addr_a: str,
    country_a: str,
    name_b: str,
    addr_b: str,
    country_b: str,
) -> np.ndarray:
    na = normalize_text(name_a, "name")
    nb = normalize_text(name_b, "name")
    aa = normalize_text(addr_a, "address")
    ab = normalize_text(addr_b, "address")
    ta, tb = tokenize(na), tokenize(nb)
    ra, rb = tokenize(aa), tokenize(ab)
    nums_a, nums_b = set(numeric_tokens(aa)), set(numeric_tokens(ab))
    if nums_a or nums_b:
        num_ov = len(nums_a & nums_b) / max(1, len(nums_a | nums_b))
    else:
        num_ov = 0.0
    ca, cb = country_key(country_a), country_key(country_b)
    country_eq = 1.0 if ca and cb and ca == cb else 0.0

    feats = [
        _jaccard(ta, tb),
        JaroWinkler.normalized_similarity(na, nb) if na or nb else 1.0,
        fuzz.token_sort_ratio(na, nb) / 100.0,
        fuzz.partial_ratio(na, nb) / 100.0,
        _jaccard(ra, rb),
        JaroWinkler.normalized_similarity(aa, ab) if aa or ab else 1.0,
        fuzz.token_sort_ratio(aa, ab) / 100.0,
        num_ov,
        country_eq,
        _len_ratio(na, nb),
        _len_ratio(aa, ab),
    ]
    return np.asarray(feats, dtype=np.float32)


def build_feature_matrix(
    pairs: list[tuple[dict, dict]],
) -> np.ndarray:
    """pairs: list of (s1_record_dict, cand_record_dict)."""
    if not pairs:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    rows = [
        pair_features(
            a["business_name"],
            a["business_address"],
            a["country"],
            b["business_name"],
            b["business_address"],
            b["country"],
        )
        for a, b in pairs
    ]
    return np.stack(rows, axis=0)
