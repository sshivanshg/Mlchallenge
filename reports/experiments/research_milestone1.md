# Research Milestone 1 — SUPERSEDED (synthetic stub only)

> **This report was produced on `synthetic_dev_fixture_v2` before the official
> challenge dumps were available.** It is kept only as historical ledger context.
>
> **Do not use these scores for model selection or leaderboard claims.**
>
> Re-run on the official dataset:
>
> ```bash
> bash scripts/download_dataset.sh
> python3 code/business_entity_resolution/src/run_milestone1.py \
>   --dataset-root student_resource/dataset \
>   --reports-root reports \
>   --seed 42
> ```
>
> Release: https://github.com/sshivanshg/Mlchallenge/releases/tag/dataset-v1

The body below is the obsolete synthetic-run write-up.

---

**Date:** 2026-09-25 (synthetic era)  
**Skill:** `.agents/skills/aws-entity-resolution/`  
**Branch:** `main`

## Critical data caveat (resolved)

Official AWS challenge dumps are now installed under `student_resource/dataset/`
and packaged as GitHub release `dataset-v1`. Numbers in the remainder of this
file are **not** from that dump.

## Dataset (`synthetic_dev_fixture_v2`) — obsolete

| Source | Records |
| --- | --- |
| S1 | 400 |
| S2 | 658 |
| S3 | 553 |

See archived CSV/JSON under `reports/experiments/` from the same timestamp for
the stub run details. Treat all of them as non-authoritative once an
`official_challenge_dataset` milestone report is generated.
