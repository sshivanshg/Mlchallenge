"""Programmatic dataset audit — no external enrichment."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from data.load import read_ground_truth, read_sources, write_json

_NON_ASCII = re.compile(r"[^\x00-\x7f]")
_DIGIT = re.compile(r"\d+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def _field_stats(series: pd.Series) -> dict[str, Any]:
    s = series.fillna("").astype(str)
    empty = int((s.str.strip() == "").sum())
    lengths = s.str.len()
    tokens = s.map(lambda x: len(x.split()) if x.strip() else 0)
    return {
        "n": int(len(s)),
        "empty": empty,
        "empty_rate": empty / max(len(s), 1),
        "char_len_mean": float(lengths.mean()) if len(s) else 0.0,
        "char_len_p50": float(lengths.median()) if len(s) else 0.0,
        "char_len_p95": float(lengths.quantile(0.95)) if len(s) else 0.0,
        "token_len_mean": float(pd.Series(tokens).mean()) if len(s) else 0.0,
        "non_ascii_rate": float(s.map(lambda x: bool(_NON_ASCII.search(x))).mean()) if len(s) else 0.0,
        "has_digit_rate": float(s.map(lambda x: bool(_DIGIT.search(x))).mean()) if len(s) else 0.0,
        "has_punct_rate": float(s.map(lambda x: bool(_PUNCT.search(x))).mean()) if len(s) else 0.0,
        "numeric_token_mean": float(
            s.map(lambda x: len(_DIGIT.findall(x))).mean()
        )
        if len(s)
        else 0.0,
    }


def _source_report(df: pd.DataFrame, name: str) -> dict[str, Any]:
    ids = df["entity_id"].astype(str)
    return {
        "source": name,
        "n_records": int(len(df)),
        "n_unique_ids": int(ids.nunique()),
        "n_duplicate_ids": int(len(ids) - ids.nunique()),
        "country_counts": dict(Counter(df["country"].astype(str).tolist())),
        "business_name": _field_stats(df["business_name"]),
        "business_address": _field_stats(df["business_address"]),
        "country": _field_stats(df["country"]),
    }


def _gt_report(gt: dict[str, set[str]], s1_ids: list[str]) -> dict[str, Any]:
    for sid in s1_ids:
        gt.setdefault(sid, set())
    match_counts = [len(gt[s]) for s in s1_ids]
    n_s2 = n_s3 = both = 0
    for s in s1_ids:
        m = gt[s]
        has_s2 = any(x.startswith("S2-") for x in m)
        has_s3 = any(x.startswith("S3-") for x in m)
        n_s2 += sum(1 for x in m if x.startswith("S2-"))
        n_s3 += sum(1 for x in m if x.startswith("S3-"))
        both += int(has_s2 and has_s3)
    hist = Counter(match_counts)
    return {
        "total_s1": len(s1_ids),
        "singletons": sum(1 for c in match_counts if c == 0),
        "singleton_rate": sum(1 for c in match_counts if c == 0) / max(len(s1_ids), 1),
        "exactly_one_match": sum(1 for c in match_counts if c == 1),
        "multi_match": sum(1 for c in match_counts if c > 1),
        "n_s2_matches": n_s2,
        "n_s3_matches": n_s3,
        "entities_matched_both_s2_s3": both,
        "match_count_histogram": {str(k): int(v) for k, v in sorted(hist.items())},
        "max_matches_per_s1": max(match_counts) if match_counts else 0,
        "mean_matches_per_s1": float(sum(match_counts) / max(len(match_counts), 1)),
    }


def run_eda(train_dir: Path, out_dir: Path, data_provenance: str) -> dict[str, Any]:
    s1, s2, s3 = read_sources(train_dir, "train")
    gt = read_ground_truth(train_dir / "train_ground_truth.tsv")
    report = {
        "provenance": data_provenance,
        "train_dir": str(train_dir),
        "sources": {
            "s1": _source_report(s1, "S1"),
            "s2": _source_report(s2, "S2"),
            "s3": _source_report(s3, "S3"),
        },
        "ground_truth": _gt_report(gt, list(s1["entity_id"].astype(str))),
        "noise_examples": _noise_examples(s1, s2, s3, gt),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(report, out_dir / "dataset_stats.json")
    (out_dir / "dataset_report.md").write_text(_to_markdown(report), encoding="utf-8")
    return report


def _noise_examples(s1, s2, s3, gt, k: int = 5) -> list[dict[str, Any]]:
    tgt = pd.concat([s2, s3], ignore_index=True).set_index("entity_id")
    s1i = s1.set_index("entity_id")
    examples = []
    for sid, matches in gt.items():
        if not matches or sid not in s1i.index:
            continue
        mid = next(iter(matches))
        if mid not in tgt.index:
            continue
        a, b = s1i.loc[sid], tgt.loc[mid]
        examples.append(
            {
                "s1_id": sid,
                "match_id": mid,
                "s1_name": a["business_name"],
                "match_name": b["business_name"],
                "s1_address": a["business_address"],
                "match_address": b["business_address"],
                "country": a["country"],
            }
        )
        if len(examples) >= k:
            break
    return examples


def _to_markdown(report: dict[str, Any]) -> str:
    gt = report["ground_truth"]
    lines = [
        "# Dataset audit",
        "",
        f"**Provenance:** `{report['provenance']}`",
        "",
        "## Sources",
        "",
    ]
    for key in ("s1", "s2", "s3"):
        s = report["sources"][key]
        lines += [
            f"### {s['source']}",
            f"- records: {s['n_records']} (unique IDs {s['n_unique_ids']}, duplicate IDs {s['n_duplicate_ids']})",
            f"- countries: `{s['country_counts']}`",
            f"- name empty rate: {s['business_name']['empty_rate']:.3f}; addr empty: {s['business_address']['empty_rate']:.3f}",
            f"- name non-ASCII rate: {s['business_name']['non_ascii_rate']:.3f}",
            f"- addr digit-token mean: {s['business_address']['numeric_token_mean']:.2f}",
            "",
        ]
    lines += [
        "## Ground truth",
        f"- total S1: {gt['total_s1']}",
        f"- singletons: {gt['singletons']} ({100*gt['singleton_rate']:.1f}%)",
        f"- exactly-one: {gt['exactly_one_match']}; multi: {gt['multi_match']}",
        f"- S2 links: {gt['n_s2_matches']}; S3 links: {gt['n_s3_matches']}; both-sources entities: {gt['entities_matched_both_s2_s3']}",
        f"- match-count histogram: `{gt['match_count_histogram']}`",
        f"- max matches/S1: {gt['max_matches_per_s1']}",
        "",
        "## Noise examples (true pairs)",
        "",
    ]
    for ex in report["noise_examples"]:
        lines.append(
            f"- `{ex['s1_id']}` ↔ `{ex['match_id']}` ({ex['country']}): "
            f"name `{ex['s1_name']}` vs `{ex['match_name']}`"
        )
    lines.append("")
    return "\n".join(lines)
