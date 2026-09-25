# Official Stage 1 — streaming audit (verified dump)

**Provenance:** `official_challenge_dataset:dataset-v1`  
**Source:** `reports/eda/official_audit.json`  
**Machine:** after provenance sha256 pass on `student_resource/dataset/`

## Counts

| Source | Records (approx from audit) |
| --- | ---: |
| Train S1 | 2,206,821 |
| Truth rows | 2,206,821 |
| Truth links | 7,638,365 |
| S2 links / S3 links | 3,693,619 / 3,944,746 |

## Ground truth cardinality

| Matches/S1 | Count |
| ---: | ---: |
| 0 (singleton) | 123,247 (**5.58%**) |
| 1 | 119,157 |
| 2–11 | remainder (max 11) |

## All-empty baseline (full train S1 universe)

| Metric | Value |
| --- | ---: |
| **macro F0.5** | **0.055848** |
| singleton_accuracy | 1.0 |
| precision | null |
| recall | 0.0 |

Abstention alone is weak on this dump (few singletons). Matching is required for competitive scores.

## Limitations (Stage 1 only)

- No blocking / matcher / threshold / test metrics yet.
- Target-ID existence across S2/S3 not audited in this pass.
