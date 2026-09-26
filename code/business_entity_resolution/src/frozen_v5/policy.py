"""Frozen inference policy v5 (candidate: v4all retrieval + 43-feature LightGBM).

Retrieval and the first 32 features are the frozen v4 policy unchanged; 11 extra
pairwise agreement/contradiction features are appended (copied verbatim from
run_matcher_v5.extra2). Do not edit; version a new package.
"""

from __future__ import annotations

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from frozen_v4.policy import (  # frozen, immutable
    AUX_TABLES,
    CAP,
    PAIR_TABLES,
    ROUTE_TABLES,
    ROUTES,
    Retriever,
    block_features,
    load_records,
    nums,
    s1_view,
    s3_boundaries,
    sig_tokens,
)
from frozen_v4.policy import FEATURE_NAMES as V4_FEATURES

POLICY_ID = "v5_v4all_cap300_lgbm_43f"
EXTRA2_NAMES = ["name_jw", "name_token_set", "addr_token_set", "addr_partial", "first_num_equal",
                "long_num_shared", "long_num_conflict", "name_len_ratio", "addr_len_ratio", "name_tok_contain_s1", "name_tok_contain_t"]
FEATURE_NAMES = V4_FEATURES + EXTRA2_NAMES


def extra2(a, recs, cands) -> np.ndarray:
    na = a["name_norm"] or ""
    aa = a["addr_norm"] or ""
    a_nums = nums(aa)
    a_long = {n for n in a_nums if len(n) >= 5}
    ta = set(sig_tokens(na))
    rows = []
    for t in cands:
        b = recs[t]
        nb, ab = b["name_norm"] or "", b["addr_norm"] or ""
        b_nums = nums(ab)
        b_long = {n for n in b_nums if len(n) >= 5}
        tb = set(sig_tokens(nb))
        rows.append([
            JaroWinkler.normalized_similarity(na, nb) if (na or nb) else 0.0,
            fuzz.token_set_ratio(na, nb) / 100.0 if (na or nb) else 0.0,
            fuzz.token_set_ratio(aa, ab) / 100.0 if (aa or ab) else 0.0,
            fuzz.partial_ratio(aa, ab) / 100.0 if (aa or ab) else 0.0,
            float(bool(a_nums) and bool(b_nums) and a_nums[0] == b_nums[0]),
            float(bool(a_long & b_long)),
            float(bool(a_long) and bool(b_long) and not (a_long & b_long)),
            min(len(na), len(nb)) / max(len(na), len(nb), 1),
            min(len(aa), len(ab)) / max(len(aa), len(ab), 1),
            len(ta & tb) / len(ta) if ta else 0.0,
            len(ta & tb) / len(tb) if tb else 0.0,
        ])
    return np.asarray(rows, dtype=np.float32)


def features(a, cands, scores, routes, recs) -> np.ndarray:
    return np.hstack([block_features(a, cands, scores, routes, recs), extra2(a, recs, cands)])
