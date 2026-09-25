# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Cursor Cloud Agent  
**Team Members:** Autonomous agent run  
**Submission Date:** 2026-09-25

---

## 1. Executive Summary
Blocking uses token/prefix/numeric inverted indexes; pairs are scored by a from-scratch **JAX logistic regression** trained with `jit`/`grad` and functional PRNG splits. Decision threshold is tuned for **macro F0.5** on an S1-grouped validation split (precision-weighted, singletons included).

---

## 2. Methodology

### 2.1 Problem Analysis
Names/addresses show legal-suffix and street-abbreviation noise, punctuation (`&`/`and`), light typos, and partial addresses. Train countries are US/India; test also includes **France** (unseen). Country must be treated as an open string label.

### 2.2 Solution Strategy
**Approach Type:** Blocking + pairwise classifier (JAX)  
**Core Innovation:** Pure-functional JAX matcher (pytree params, no pretrained LLM); open-set country equality feature without hard-coded country vocab.

---

## 3. Candidate Generation (Blocking)
- **Blocking keys used:** normalized name prefix (3 chars), significant name tokens (len≥3), address numeric tokens (ZIP/PIN-like, len≥3); soft down-weight when country labels differ (labels never filtered).
- **Candidate pairs generated:** up to 80 candidates per S1 (configurable).
- **How you ensured true matches were not lost:** multi-key union + rare-token upweighting; measure blocking recall ceiling on train GT before matching.

---

## 4. Matching Model

**Features used:**
- Name: Jaccard, Jaro-Winkler, token-sort ratio, partial ratio, length ratio
- Address: Jaccard, Jaro-Winkler, token-sort ratio, numeric-token overlap, length ratio
- Other: country label equality (open-set strings)

**Model type:** JAX logistic regression (custom, MIT-licensed code; JAX itself is Apache-2.0; ≪8B params)  
**Threshold selection method:** grid search maximizing macro F0.5 on held-out S1 entities

**Model license record:** JAX/jaxlib Apache-2.0; pipeline code intended MIT; no external pretrained entity-resolution or geocoding models.

---

## 5. Results & Error Analysis

- **F0.5 Score (macro):** on synthetic smoke data, validation macro F0.5 = 1.0000; blocking recall ceiling = 1.0000 (see `artifacts/model/meta.json` after `run`). Replace with official-data numbers when the challenge dump is present.
- **Common false positives (wrong merges):** similar trade names sharing tokens without address/PIN agreement — raise threshold under F0.5.
- **Common false negatives (missed matches):** heavy name transpositions or landmark-only addresses that miss all blocking keys.

---

## 6. Conclusion
A JAX-native blocking+matcher pipeline produces validator-PASS `matching_results.tsv` / `candidate_pairs.tsv`, keeps country open-set (France flows through), and tunes for precision-weighted macro F0.5. Swap in the official dataset under `student_resource/dataset/` and re-run to produce leaderboard outputs.

---

## Appendix

### A. Code Artefacts
Runnable code: `code/business_entity_resolution/` (`src/`, `README.md`, `requirements.txt`). Entrypoint:

```bash
cd code/business_entity_resolution/src
python3 run_pipeline.py run \
  --dataset-root ../../../student_resource/dataset \
  --model-dir ../artifacts/model \
  --output-dir ../../../student_resource/output
```

Then:
```bash
cd ../../../student_resource
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test --check-ids
```

### B. Additional Results
Synthetic mini-set used when official train/test dumps were not in the repo (`generate_sample_data.py`). Official-scale timings/metrics should be re-measured after placing the real TSV dumps.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
