"""Blocking / candidate generation — sub-quadratic inverted indexes."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import pandas as pd

from normalize import (
    country_key,
    name_prefix,
    normalize_text,
    numeric_tokens,
    significant_tokens,
    tokenize,
)


def prepare_records(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["name_norm"] = out["business_name"].map(lambda x: normalize_text(x, "name"))
    out["addr_norm"] = out["business_address"].map(lambda x: normalize_text(x, "address"))
    out["country_norm"] = out["country"].map(country_key)
    out["name_tokens"] = out["name_norm"].map(lambda x: significant_tokens(tokenize(x)))
    out["addr_nums"] = out["addr_norm"].map(numeric_tokens)
    out["name_pref"] = out["name_norm"].map(lambda x: name_prefix(x, 3))
    return out


def _build_indexes(records: pd.DataFrame) -> dict[str, dict[str, set[str]]]:
    by_token: dict[str, set[str]] = defaultdict(set)
    by_prefix: dict[str, set[str]] = defaultdict(set)
    by_num: dict[str, set[str]] = defaultdict(set)
    by_country: dict[str, set[str]] = defaultdict(set)

    for row in records.itertuples(index=False):
        eid = row.entity_id
        by_country[row.country_norm].add(eid)
        if row.name_pref:
            by_prefix[row.name_pref].add(eid)
        for tok in row.name_tokens:
            if len(tok) >= 3:
                by_token[tok].add(eid)
        for num in row.addr_nums:
            if len(num) >= 3:
                by_num[num].add(eid)
    return {
        "token": dict(by_token),
        "prefix": dict(by_prefix),
        "num": dict(by_num),
        "country": dict(by_country),
    }


def generate_candidates(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    max_per_key: int = 200,
    max_candidates: int = 80,
) -> dict[str, list[str]]:
    """Return S1 -> ordered candidate S2/S3 IDs (final set fed to the matcher)."""
    s1p = prepare_records(s1)
    targets = prepare_records(pd.concat([s2, s3], ignore_index=True))
    indexes = _build_indexes(targets)
    id_to_country = dict(zip(targets["entity_id"], targets["country_norm"]))

    candidates: dict[str, list[str]] = {}
    for row in s1p.itertuples(index=False):
        scores: dict[str, float] = defaultdict(float)
        country = row.country_norm

        def add_bucket(ids: Iterable[str], weight: float) -> None:
            for eid in ids:
                # Prefer same country label when both present; never hard-code labels.
                if country and id_to_country.get(eid) and id_to_country[eid] != country:
                    scores[eid] += weight * 0.15
                else:
                    scores[eid] += weight

        if row.name_pref and row.name_pref in indexes["prefix"]:
            bucket = list(indexes["prefix"][row.name_pref])[:max_per_key]
            add_bucket(bucket, 1.0)

        for tok in row.name_tokens:
            if len(tok) < 3 or tok not in indexes["token"]:
                continue
            bucket = list(indexes["token"][tok])
            # Rare tokens are more discriminative.
            weight = 2.0 / (1.0 + len(bucket) / 50.0)
            add_bucket(bucket[:max_per_key], weight)

        for num in row.addr_nums:
            if len(num) >= 3 and num in indexes["num"]:
                add_bucket(list(indexes["num"][num])[:max_per_key], 1.5)

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        candidates[row.entity_id] = [eid for eid, _ in ranked[:max_candidates]]

    return candidates


def blocking_recall(
    candidates: dict[str, list[str]],
    ground_truth: dict[str, set[str]],
) -> float:
    """Fraction of true matched IDs retained by blocking (recall ceiling)."""
    hit = total = 0
    for s1, truths in ground_truth.items():
        if not truths:
            continue
        total += len(truths)
        cset = set(candidates.get(s1, []))
        hit += sum(1 for t in truths if t in cset)
    return hit / total if total else 1.0
