"""Multi-route candidate generation (local data only)."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from normalization.views import attach_views


def _add(scores: dict[str, float], ids: Iterable[str], weight: float) -> None:
    for eid in ids:
        scores[eid] += weight


def generate_candidates(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    *,
    max_per_route: int = 40,
    max_candidates: int = 60,
    use_tfidf: bool = True,
    tfidf_k: int = 20,
) -> dict[str, list[str]]:
    s1p = attach_views(s1)
    tgt = attach_views(pd.concat([s2, s3], ignore_index=True))
    by_name: dict[str, list[str]] = defaultdict(list)
    by_prefix: dict[str, list[str]] = defaultdict(list)
    by_token: dict[str, list[str]] = defaultdict(list)
    by_num: dict[str, list[str]] = defaultdict(list)
    id_country = {}

    for row in tgt.itertuples(index=False):
        eid = row.entity_id
        id_country[eid] = row.country_norm
        if row.name_legal:
            by_name[row.name_legal].append(eid)
        if row.name_prefix3:
            by_prefix[row.name_prefix3].append(eid)
        for tok in row.name_tokens:
            if len(tok) >= 3:
                by_token[tok].append(eid)
        for num in row.addr_nums:
            if len(num) >= 3:
                by_num[num].append(eid)

    nn_idx = None
    nn_ids = None
    if use_tfidf and len(tgt) > 0:
        corpus = (tgt["name_legal"].fillna("") + " " + tgt["addr_norm"].fillna("")).tolist()
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        X = vec.fit_transform(corpus)
        nn = NearestNeighbors(metric="cosine", algorithm="brute")
        nn.fit(X)
        q = vec.transform(
            (s1p["name_legal"].fillna("") + " " + s1p["addr_norm"].fillna("")).tolist()
        )
        k = min(tfidf_k, len(tgt))
        dists, idxs = nn.kneighbors(q, n_neighbors=k)
        nn_idx = (dists, idxs)
        nn_ids = tgt["entity_id"].tolist()

    out: dict[str, list[str]] = {}
    for i, row in enumerate(s1p.itertuples(index=False)):
        scores: dict[str, float] = defaultdict(float)
        if row.name_legal and row.name_legal in by_name:
            _add(scores, by_name[row.name_legal][:max_per_route], 5.0)
        if row.name_prefix3 and row.name_prefix3 in by_prefix:
            _add(scores, by_prefix[row.name_prefix3][:max_per_route], 1.0)
        for tok in row.name_tokens:
            if len(tok) >= 3 and tok in by_token:
                bucket = by_token[tok]
                w = 2.0 / (1.0 + len(bucket) / 40.0)
                _add(scores, bucket[:max_per_route], w)
        for num in row.addr_nums:
            if len(num) >= 3 and num in by_num:
                _add(scores, by_num[num][:max_per_route], 1.8)
        if nn_idx is not None:
            dists, idxs = nn_idx
            for d, j in zip(dists[i], idxs[i]):
                scores[nn_ids[j]] += 2.0 * (1.0 - float(d))

        # Soft country preference (never hard-exclude)
        for eid in list(scores):
            if row.country_norm and id_country.get(eid) and id_country[eid] != row.country_norm:
                scores[eid] *= 0.5

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        out[row.entity_id] = [e for e, _ in ranked[:max_candidates]]
    return out
