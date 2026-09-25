# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** _to be filled by the team_  
**Team Members:** _to be filled by the team_  
**Submission Date:** 2026-09-25

---

## 1. Executive Summary
Local multi-route blocking over SQLite inverted indexes, nine interpretable name/address/number/country
similarity features, and a LightGBM classifier with a single probability threshold chosen to maximize
macro F0.5 (singletons included) on a held-out, component-disjoint selection fold. Only the official
challenge TSVs are used (verified by SHA-256 manifest); no external lookups, APIs, or pretrained models.

Frozen policy: `v3b_cap300_lgbm_thr0.98_nogate` (`code/business_entity_resolution/src/frozen_v3b/policy.py`).

---

## 2. Methodology

### 2.1 Problem Analysis
- Train: 2,206,821 S1 and 10,320,219 S2+S3 records (US, India). Test: 1,732,544 S1 (US 663,106;
  India 809,986; France 259,452 — France is absent from training) and 9,969,589 S2+S3 records.
- 5.58% of training S1 entities are singletons; the mean is 3.46 matches per S1 (up to 11), so the
  task is zero-to-many, not one-to-one.
- Noise: legal-suffix variants, abbreviations, word reordering, typos, joined words and domain-style
  names (`kritavilabscom`), accents, non-Latin script names, partial or reordered addresses.

### 2.2 Solution Strategy
**Approach Type:** Blocking + pairwise classifier + thresholded set selection  
**Core Innovation:** Selective composite retrieval keys (pairs of informative name tokens, and name
token × address number) applied *before* per-key truncation, with explicit S2/S3 allocation inside
each key lookup. Diagnostics showed that most retrieval loss came from arbitrary truncation of common
keys, not from missing keys.

**Validation:** S1 entities were split 70/15/15 into fit / select / assess folds, with connected positive
components kept together (seed 42). The model is fit on fit-fold candidates and the threshold is tuned on
5,000 select S1s. Final confirmation used 5,000 fresh assessment S1s that share no component with
earlier-inspected assessment IDs (`experiments/splits/official_70bc1d8a16c6_assess_fresh_v1.json`).

---

## 3. Candidate Generation (Blocking)
- **Normalization:** NFKC, case-fold, accent fold, `&`→`and`, punctuation removal, legal/street
  abbreviation expansion; informative name tokens exclude stopwords and generic legal words.
- **Blocking keys used:** exact normalized name; sorted informative-token signature; 4-char compact
  name prefix; informative name tokens (rare tokens fetch full postings, common tokens a bounded
  sample); multi-token intersection; address numbers (≥3 digits); **token pairs**; **token × address
  number**. Per-key lookups are split between S2 and S3 rows so neither source is starved.
- **Ranking:** additive route scores with an IDF weight on tokens and a soft ×0.55 country mismatch
  down-weight (never a hard filter, so unseen countries such as France keep candidates).
- **Candidate pairs generated:** top 300 per S1 (select mean 269 candidates).
- **How you ensured true matches were not lost:** measured candidate recall and candidate-oracle
  macro F0.5 on held-out S1s, attributed every miss to a stage (missing key, per-key truncation, final
  cap), and changed one variable at a time.

| Retrieval (select, 5,000 S1) | Cap | Recall | Oracle macro F0.5 |
|---|---|---|---|
| v2 (previous) | 120 | 0.615 | 0.785 |
| v3b (final) | 120 | 0.741 | 0.867 |
| v3b (final) | 300 | 0.799 | 0.904 |

---

## 4. Matching Model

**Features used (9):** name token-sort ratio, informative-name-token Jaccard, exact normalized name,
address token-sort ratio, address-number Jaccard, country equality, a name-high/address-low conflict
flag, name partial ratio, address-token Jaccard. IDs, row order, and split membership are never features.

**Model type:** LightGBM binary classifier (300 trees, 31 leaves, learning rate 0.05; MIT license;
≈ tens of thousands of parameters). Trained on 103,884 fit-fold candidate pairs (22,165 positives) with
up to max(8, 3×positives) random non-matching candidates per S1 as negatives.  
**Threshold selection method:** grid search of a single probability threshold (and an optional
corroboration gate, which was not selected) maximizing exact macro F0.5 on the selection fold → 0.98.

**Model license record:** LightGBM (MIT), scikit-learn (BSD-3), RapidFuzz (MIT), NumPy (BSD).
No pretrained or external models.

---

## 5. Results & Error Analysis

| Policy | Evaluation set | Macro F0.5 | Precision | Recall | Singleton acc. |
|---|---|---|---|---|---|
| Previous (v2 cap 120, logistic + gate) | select 5,000 | 0.664 | 0.872 | 0.523 | 0.652 |
| Final (v3b cap 300, LightGBM) | select 5,000 | 0.796 | 0.944 | 0.659 | 0.889 |
| Previous | fresh assess 5,000 | 0.664 | 0.863 | 0.532 | 0.612 |
| **Final** | **fresh assess 5,000** | **0.793** | 0.943 | 0.664 | 0.794 |

- The fresh-assessment candidate oracle is 0.908, so both retrieval (≈ 19% of true links not
  retrieved) and matching (≈ 0.115 below the oracle) still limit the score.
- **Common false negatives:** cross-script names (Devanagari/Kannada transliterations) sharing no key;
  typos in rare tokens; true matches ranked beyond the cap for very common names.
- **Common false positives:** same or near-identical names at different addresses (chains,
  generic names), which hurt singletons most.
- France has no labels, so its accuracy is unmeasured. Unlabeled check: France has more predicted links
  per S1 (4.7) than US (2.8) or India (2.1).

---

## 6. Conclusion
Retrieval diagnostics showed that most losses came from truncation, and selective composite keys fixed
much of it. That, plus a small tree model, raised measured macro F0.5 from 0.664 to 0.793 on untouched
held-out entities. Remaining work: typo-tolerant and cross-script retrieval, and hard-negative mining
for same-name/different-address pairs.

---

## Appendix

### A. Code Artefacts
Runnable code: `code/business_entity_resolution/` (`src/`, `README.md`, `requirements.txt`).
Exact reproduction commands are in `code/business_entity_resolution/README.md`
("Reproduce the submission"). Final inference:

```bash
python3 -u code/business_entity_resolution/src/run_infer_v3b.py --split test --workers 4 \
    --shard-size 20000 --out-dir artifacts/submissions/v3b_cap300_lgbm_v1
```

Submission validation:

```bash
cd student_resource
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test --check-ids
```

### B. Additional Results
`reports/official/official_70bc1d8a16c6/retrieval_exp1/` (blocking diagnostics, cap sweeps, miss
attribution, matcher comparisons, fresh-assessment scores) and `.../official_results_table.json`.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
