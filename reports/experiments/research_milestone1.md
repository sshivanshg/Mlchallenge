# Research Milestone 1 — Measured Report

**Date:** 2026-09-25  
**Skill:** `.agents/skills/aws-entity-resolution/` (installed; SKILL.md followed)  
**Branch:** `main`

## Critical data caveat

**Official AWS challenge dumps (≈millions of rows) are NOT present in this repository.**

What was found under `student_resource/dataset/` was initially a tiny smoke fixture. For milestone measurement we generated a local development fixture:

- **Provenance:** `synthetic_dev_fixture_v2` (`student_resource/dataset/PROVENANCE.txt`)
- **Purpose:** exercise the experimentation engine only
- **Not** official challenge data; **do not** treat scores as leaderboard estimates

All metrics below are on that synthetic fixture + frozen `split_v1`.

---

## Phase 0 — Repository audit

| Item | Location | Status |
| --- | --- | --- |
| Skill | `.agents/skills/aws-entity-resolution/SKILL.md` | Present (also mirrored under `.claude/skills`, `.cursor/skills`) |
| Problem statement | `ProblemStatement.md`, `student_resource/README.md` | Present |
| Validator | `student_resource/utils/validate_submission.py` | Present |
| Docs template | `student_resource/Documentation_template.md` | Present |
| Official train/test dumps | — | **Missing** |
| Prior code | `code/business_entity_resolution/` | Preserved; extended into research layout |

---

## Dataset (`synthetic_dev_fixture_v2`)

| Source | Records |
| --- | --- |
| S1 | 400 |
| S2 | 658 |
| S3 | 553 |

**Ground truth**

- Singletons: **135 / 400 = 33.75%**
- Exactly one match: 136
- Multi-match: 129 (max 2 / S1)
- S2 links: 241; S3 links: 153; both-sources entities: 129
- Train countries: US 200 / India 200 (France only in test fixture)

**Observed noise (fixture)**

- Legal suffix / street abbreviation variants (`Street`↔`St`, `Ltd`↔deletion typos)
- Token reorderings and character deletions
- Generic shared names (`Global Trading`, `Premier Services`, …)
- Shared-address collisions across different businesses

Full audit: `reports/eda/dataset_report.md`, `reports/eda/dataset_stats.json`.

---

## Validation

| Field | Value |
| --- | --- |
| Version | `split_v1` |
| Strategy | Positive-component-disjoint S1 groups (S1s sharing a positive S2/S3 target stay together) |
| Seed | 42 |
| Val fraction | 0.25 |
| Sizes | train 300 / val 100 S1 |
| Leakage protections | No random pair split; TF-IDF/model fit on train fold pairs only; val candidates scored without injecting missed positives |
| Manifests | `experiments/splits/split_v1*.json/txt` |

---

## Blocking (val)

| Metric | Value |
| --- | --- |
| Micro candidate recall | **1.000** |
| S2 / S3 recall | 1.000 / 1.000 |
| Full-set coverage | 1.000 |
| Avg / p95 / max candidates | 60 / 60 / 60 (cap) |
| Reduction ratio | ~0.950 |
| Oracle macro F0.5 | 1.000 |
| True matches missed | 0 |

Config: union of exact-name, prefix3, rare tokens, address numerics, char_wb TF-IDF ANN; soft country down-weight (never hard exclude).  
Report: `reports/experiments/blocking_baseline.md`.

---

## Baselines (validation `split_v1`)

| Experiment | Candidate Recall | Precision | Recall | Singleton Acc | Macro F0.5 |
| --- | --- | --- | --- | --- | --- |
| B0_empty | 1.00* | — | 0.00 | 1.00 | **0.3500** |
| B1_exact_norm | n/a (full index) | 1.00 | 0.439 | 1.00 | **0.6850** |
| B2_exact_on_candidates | 1.00 | 1.00 | 0.439 | 1.00 | **0.6850** |
| M0_weighted (thr=0.875) | 1.00 | 0.974 | 0.755 | 1.00 | **0.8706** |
| M1_logistic (thr=0.50) | 1.00 | 0.925 | 1.00 | 0.971 | **0.9767** |
| M2_lightgbm (thr=0.50) | 1.00 | 1.00 | 0.990 | 1.00 | **0.9983** |

\*Empty baseline does not use candidates; candidate recall column shows current blocker ceiling for context.

Ledger: `reports/experiments/experiments.csv`.

---

## Current best local result

- **Experiment:** `M2_lightgbm`
- **Macro F0.5:** **0.9983** (synthetic val only)
- **Config:** LightGBM (`n_estimators=120`, `lr=0.08`) on 25 interpretable pair features; hard negatives = non-match blocker candidates; global threshold 0.50 tuned for macro F0.5
- **Features:** exact/Jaro/RapidFuzz name+address, Jaccard/containment, numeric overlap/conflict, open-set country equality/missingness, retrieval rank/score, name–address agreement
- **License:** LightGBM MIT; sklearn BSD; RapidFuzz MIT; no pretrained >8B models

---

## Failure analysis (best model)

Counts on val (`reports/error_analysis/milestone1_errors.*`):

| Category | Count |
| --- | --- |
| singleton_fp | 0 |
| false_positive | 0 |
| false_negative_blocking | 0 |
| false_negative_threshold | 1 |
| partial_miss | 1 |

Example FN: `S1-00250` (`Global Trading LLC`) missed `S3-00346` — generic-name ambiguity / score below threshold.

On this fixture, failures are scarce; **do not extrapolate** to official data where blocking recall << 1 and generic-name collisions dominate.

---

## Next 3 experiments (evidence-based)

1. **Stress blocking under cap reduction** — avg candidates are pinned at the 60-cap with recall 1.0; measure recall–cost curve at caps {5,10,20,40} before trusting the blocker on large data.
2. **Generic-name / shared-address features** — sole FN is a generic trade name; add rarity-weighted token IDF (fit on train only) and stronger name↔address contradiction penalties; re-check singleton FP rate.
3. **Leave-country-out stress** — train on US-only / India-only partitions, evaluate on the held-out country within the fixture; verify open-set country handling before France-bearing official test.

**Blocked for real leaderboard progress:** obtain and hash the official train/test TSV dumps into `student_resource/dataset/`, re-freeze splits, re-run milestone1 — synthetic scores are not private-LB proxies.

---

## Reproduce

```bash
python3 -m unittest discover -s .agents/skills/aws-entity-resolution/scripts -p 'test_*.py'
python3 -m unittest discover -s code/business_entity_resolution/tests -p 'test_*.py'
python3 code/business_entity_resolution/src/run_milestone1.py \
  --dataset-root student_resource/dataset \
  --reports-root reports \
  --make-hard-synthetic \
  --seed 42
```
