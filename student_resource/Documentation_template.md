# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Cursor Cloud Agent  
**Team Members:** Autonomous agent run  
**Submission Date:** 2026-09-25

---

## 1. Executive Summary
Blocking + interpretable pairwise features + classical matchers (logistic / LightGBM), with thresholds tuned for **macro F0.5** on an S1-grouped validation split. Uses the **official** challenge dataset under `student_resource/dataset/` (not synthetic stubs).

---

## 2. Methodology

### 2.1 Problem Analysis
Names/addresses show legal-suffix and street-abbreviation noise, punctuation (`&`/`and`), light typos, and partial addresses. Train countries are US/India; test also includes **France** (unseen). Country must be treated as an open string label.

### 2.2 Solution Strategy
**Approach Type:** Blocking + pairwise classifier  
**Core Innovation:** Multi-route local retrieval + hard negatives from blocker candidates; open-set country equality without hard-coded country vocab; F0.5-tuned thresholds.

---

## 3. Candidate Generation (Blocking)
- **Blocking keys used:** normalized name prefix, significant name tokens, address numeric tokens, char n-gram TF-IDF neighbors; soft country down-weight (never hard-exclude).
- **Candidate pairs generated:** capped per S1 (see experiment configs).
- **How you ensured true matches were not lost:** measure blocking recall ceiling on held-out S1 before matching.

---

## 4. Matching Model

**Features used:**
- Name: exact, Jaccard, Jaro-Winkler, RapidFuzz ratios, containment
- Address: exact, token/char similarities, numeric-token overlap/conflict
- Other: open-set country equality/missingness, retrieval rank/score, name–address agreement

**Model type:** logistic regression and/or LightGBM (MIT/Apache-compatible stack; ≪8B params)  
**Threshold selection method:** grid search maximizing macro F0.5 on held-out S1 entities

**Model license record:** scikit-learn / LightGBM / RapidFuzz / JAX (if used) — MIT or Apache-2.0; no external pretrained entity-resolution or geocoding models.

---

## 5. Results & Error Analysis

- **F0.5 Score (macro):** re-measure on the **official** validation split (`reports/experiments/` after `run_milestone1.py`). Do not cite synthetic stub scores.
- **Common false positives / negatives:** fill from official-data error analysis.

---

## 6. Conclusion
Pipeline targets precision-weighted macro F0.5 on the official challenge dumps installed via `scripts/download_dataset.sh` / release `dataset-v1`.

---

## Appendix

### A. Code Artefacts
Runnable code: `code/business_entity_resolution/` (`src/`, `README.md`, `requirements.txt`).

```bash
# Ensure official data is present
bash scripts/download_dataset.sh

cd code/business_entity_resolution/src
python3 run_milestone1.py \
  --dataset-root ../../../student_resource/dataset \
  --reports-root ../../../reports \
  --seed 42
```

Submission validation (after inference writes `output/`):

```bash
cd student_resource
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test --check-ids
```

### B. Additional Results
See `reports/experiments/` and `reports/error_analysis/` produced on the official dataset.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
