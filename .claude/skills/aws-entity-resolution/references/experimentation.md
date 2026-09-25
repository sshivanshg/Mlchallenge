# Experiments and error analysis

## Reproducible ledger

Maintain one immutable record per run, including failed/aborted runs. Record:

experiment_id; parent/baseline_id; hypothesis; code/config change; code revision; input hashes; environment versions; deterministic seeds; validation split/hash; features; blocking strategy; model/artifact/license; thresholds/decision policy; candidate recall; precision; recall; macro_F0.5; singleton accuracy; candidate count statistics; reduction ratio; runtime; peak memory; status; notes; public leaderboard score when available.

Use explicit null plus a reason for unavailable metrics. Store per-fold/per-slice counts and metrics, per-S1 predictions/scores, final candidate manifests, and failure logs as local artifacts. Keep sensitive records out of external logs.

Every hypothesis should name its targeted error cluster and a rejection condition. Change as few major variables as practical. Compare identical evaluation entities and resource accounting. Do not call a change better based only on pairwise accuracy, ROC-AUC, generic F1, or a favorable public score.

Set experiment count/time/memory budgets before long sweeps. Smoke-test on a small training subset, then run the frozen comparison. Cache expensive features with fold/data/code/config keys. Recover interrupted runs without mixing configurations. Preserve best-run artifacts and exact commands before starting the next branch.

## Error review protocol

After important runs, join truth, candidate membership, raw/normalized fields, scores, and decisions locally. Sample singleton false positives, ordinary false positives, false negatives, and improvements/regressions against the baseline. Include samples across sources/countries and score ranges, not just the most conspicuous mistakes.

Tag multiple causes when appropriate:

- Normalization failure; typo; abbreviation; legal suffix; transliteration.
- Name reordered; address reordered; missing address; landmark address.
- Ambiguous same/similar name; numerical mismatch.
- Retrieval failure; threshold failure; singleton false positive; false negative.
- Source-specific issue; country-specific issue.

Distinguish retrieval misses from scored-but-rejected positives, and feature/model failure from selection failure. For every cluster, record count, affected S1 entities, macro-score contribution, examples from provided data, proposed intervention, and expected cost. Prioritize recoverable score loss, not raw pair frequency.

Example hypotheses: add a digit-preserving route for missed numeric addresses; add name/address contradiction interactions for singleton false merges; retain original diacritics beside folded text when normalization creates collisions. Treat these as proposals, not predetermined improvements.

## Selection and leaderboard discipline

Use local evidence as the main selection mechanism. Record every public submission with the exact run/config, local metrics, hypothesis, and resulting score. Predefine a modest submission budget; avoid probing individual identities or inferring hidden labels from leaderboard changes.

A public improvement with local regression triggers investigation of variance, domain shift, or validation mismatch. It is not automatic promotion. Do not repeatedly redesign validation to agree with public scores. Keep a final untouched assessment and log how often it is inspected.

Before final inference, freeze features, retrieval budgets, model, thresholds, seeds, and serialization. Full-training refit is allowed after selection, but check score-scale changes: use a predetermined calibration strategy based on held-out or out-of-fold evidence, not guessed test labels. Stop when budget is exhausted or remaining hypotheses lack a plausible measurable gain.
