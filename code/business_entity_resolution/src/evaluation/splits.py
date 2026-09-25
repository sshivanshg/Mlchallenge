"""Leakage-safe S1-grouped validation splits."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def build_positive_components(gt: dict[str, set[str]]) -> list[set[str]]:
    """Group S1 IDs that share any positive target ID (and singleton S1s alone)."""
    target_to_s1: dict[str, set[str]] = defaultdict(set)
    for s1, matches in gt.items():
        for m in matches:
            target_to_s1[m].add(s1)

    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for s1 in gt:
        parent.setdefault(s1, s1)
    for s1s in target_to_s1.values():
        s1s = list(s1s)
        for other in s1s[1:]:
            union(s1s[0], other)

    comps: dict[str, set[str]] = defaultdict(set)
    for s1 in gt:
        comps[find(s1)].add(s1)
    return list(comps.values())


def freeze_split(
    gt: dict[str, set[str]],
    countries: dict[str, str],
    out_dir: Path,
    seed: int = 42,
    val_frac: float = 0.25,
    version: str = "v1",
) -> dict[str, Any]:
    """Component-disjoint train/val split of S1 entities; write manifests."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frozen = out_dir / f"split_{version}.json"
    if frozen.exists():
        meta = json.loads(frozen.read_text(encoding="utf-8"))
        train_ids, val_ids = set(meta["train_ids"]), set(meta["val_ids"])
        if (meta["seed"] != seed or meta["val_frac"] != val_frac
                or meta["version"] != version or train_ids & val_ids
                or train_ids | val_ids != set(gt)):
            raise ValueError(f"Frozen split conflicts with current inputs: {frozen}")
        for component in build_positive_components(gt):
            if component & train_ids and component & val_ids:
                raise ValueError(f"Frozen split leaks a positive component: {frozen}")
        return meta

    rng = np.random.default_rng(seed)
    components = sorted(build_positive_components(gt), key=min)
    # Stratify-ish: shuffle components, fill val until frac
    order = np.arange(len(components))
    rng.shuffle(order)
    n_s1 = len(gt)
    target_val = max(1, int(round(n_s1 * val_frac)))
    val, train = set(), set()
    for idx in order:
        comp = components[idx]
        if len(val) < target_val:
            val |= comp
        else:
            train |= comp
    # Ensure train non-empty
    if not train:
        raise ValueError("Cannot create a component-disjoint split with these labels")

    def dist(ids: set[str]) -> dict[str, Any]:
        match_counts = [len(gt[i]) for i in ids]
        ccounts: dict[str, int] = defaultdict(int)
        for i in ids:
            ccounts[countries.get(i, "")] += 1
        return {
            "n_s1": len(ids),
            "singleton_rate": sum(1 for c in match_counts if c == 0) / max(len(ids), 1),
            "mean_matches": float(sum(match_counts) / max(len(match_counts), 1)),
            "country_counts": dict(ccounts),
            "match_count_hist": {
                str(k): sum(1 for c in match_counts if c == k)
                for k in sorted(set(match_counts))
            },
        }

    meta = {
        "version": version,
        "seed": seed,
        "val_frac": val_frac,
        "strategy": "positive-component-disjoint S1 groups; residual fill by shuffled components",
        "leakage_protections": [
            "S1 entities sharing a positive S2/S3 target stay in the same fold",
            "No random pair split",
            "TF-IDF / model fitting must use train fold only",
        ],
        "train": dist(train),
        "val": dist(val),
        "train_ids": sorted(train),
        "val_ids": sorted(val),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"split_{version}.json").write_text(json.dumps(meta, indent=2) + "\n")
    (out_dir / f"split_{version}_train_ids.txt").write_text("\n".join(sorted(train)) + "\n")
    (out_dir / f"split_{version}_val_ids.txt").write_text("\n".join(sorted(val)) + "\n")
    return meta
