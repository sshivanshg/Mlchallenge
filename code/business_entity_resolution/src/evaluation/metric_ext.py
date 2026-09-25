"""Competition metric wrappers + candidate diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

_SKILL = (
    Path(__file__).resolve().parents[4]
    / ".agents"
    / "skills"
    / "aws-entity-resolution"
    / "scripts"
)
if _SKILL.is_dir() and str(_SKILL) not in sys.path:
    sys.path.insert(0, str(_SKILL))

from metric import entity_f05, evaluate  # noqa: E402


def ensure_complete(pred: dict[str, set[str]], universe: Mapping[str, set[str]]) -> dict[str, set[str]]:
    return {k: set(pred.get(k, set())) for k in universe}


def score_predictions(truth: dict[str, set[str]], pred: dict[str, set[str]]) -> dict:
    return evaluate(truth, pred)


def candidate_diagnostics(
    candidates: dict[str, list[str]],
    truth: dict[str, set[str]],
    n_targets: int,
) -> dict:
    hit = total = 0
    s2_hit = s2_tot = s3_hit = s3_tot = 0
    full = non_sing = 0
    counts = []
    zero = 0
    for s1, tset in truth.items():
        cset = set(candidates.get(s1, []))
        counts.append(len(cset))
        zero += int(len(cset) == 0)
        if not tset:
            continue
        non_sing += 1
        total += len(tset)
        inter = tset & cset
        hit += len(inter)
        if tset <= cset:
            full += 1
        for x in tset:
            if x.startswith("S2-"):
                s2_tot += 1
                s2_hit += int(x in cset)
            elif x.startswith("S3-"):
                s3_tot += 1
                s3_hit += int(x in cset)
    counts_sorted = sorted(counts)
    n = max(len(counts_sorted), 1)

    def pct(p: float) -> float:
        if not counts_sorted:
            return 0.0
        idx = min(len(counts_sorted) - 1, int(round((p / 100) * (len(counts_sorted) - 1))))
        return float(counts_sorted[idx])

    denom = max(len(truth) * max(n_targets, 1), 1)
    oracle_pred = {s1: set(truth[s1]) & set(candidates.get(s1, [])) for s1 in truth}
    oracle = score_predictions(truth, oracle_pred)
    return {
        "micro_candidate_recall": hit / total if total else None,
        "s2_recall": s2_hit / s2_tot if s2_tot else None,
        "s3_recall": s3_hit / s3_tot if s3_tot else None,
        "full_set_coverage": full / non_sing if non_sing else None,
        "avg_candidates": sum(counts) / n,
        "median_candidates": pct(50),
        "p95_candidates": pct(95),
        "p99_candidates": pct(99),
        "max_candidates": float(max(counts) if counts else 0),
        "zero_candidate_frac": zero / n,
        "reduction_ratio": 1.0 - (sum(counts) / denom),
        "oracle_macro_F0.5": oracle["macro_F0.5"],
        "true_matches_total": total,
        "true_matches_retrieved": hit,
        "true_matches_missed": total - hit,
    }


__all__ = [
    "entity_f05",
    "evaluate",
    "score_predictions",
    "candidate_diagnostics",
    "ensure_complete",
]
