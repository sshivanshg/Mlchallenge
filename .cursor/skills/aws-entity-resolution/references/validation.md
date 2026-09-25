# Validation and selection

## Exact target

For each S1 entity i, let T_i be its true target-ID set and P_i its predicted set. Let TP = |T_i ∩ P_i|, FP = |P_i − T_i|, FN = |T_i − P_i|.

- If T_i and P_i are both empty: F_i = 1.
- Otherwise: F_i = 5 TP / (5 TP + 4 FP + FN).
- macro_F0.5 = sum(F_i) / number_of_S1_entities.

This is F_beta with beta = 0.5. Do not average pairwise F scores or compute F from globally averaged precision/recall. S1 entities receive equal weight regardless of match count. Verify this reference against the official evaluator, especially source aggregation and empty sets.

Keep all S1 IDs explicitly, including zero-candidate entities. Missing prediction rows are errors, not implicit singletons. Deduplicate pair edges before set scoring but reject duplicate IDs/rows in submission validation so serialization bugs remain visible.

## Split by relationships

Inspect the S1–S2/S3 label graph before choosing splits. Group connected positive components and known alias/duplicate families so related records cannot straddle folds. Use only established relationships; do not group every weak text neighbor into a giant component. Document unobservable relationships and incomplete-label risk.

Never split random pairs: repeated S1 records, aliases, shared targets, or duplicates can leak identity. Check intersections of record IDs, positive components, and known duplicate families across fold manifests. Stratify groups where feasible by country, source coverage, singleton status, and match cardinality without breaking groups.

Distinguish model-fit, calibration/selection, and untouched assessment roles. Use grouped nested splits or out-of-fold predictions if data is limited. Repeated threshold/hypothesis searches can overfit a validation set; reserve an assessment set or outer folds and restrict how often they are inspected.

Fit token statistics, IDF, frequency features, vocabularies, learned normalizers, embeddings, hard-negative selection, and calibrators only within the allowed fold. Building an inference index over held-out target records is necessary retrieval access; fitting corpus statistics on those records is transductive learning and requires explicit permission. Use the same policy for validation and test.

Keep the realistic inference candidate universe, including unmatched distractors. Do not restrict validation retrieval to known positive targets or remove all difficult neighbors. Define target-pool access explicitly. Default to component-disjoint pools that mimic deployment; if deployment exposes a shared pool, model it without training on held-out labels and document the tradeoff.

## Required diagnostics

Let C_i be final matcher candidates; let U be all eligible S2/S3 target records under the task definition.

- Micro candidate recall = sum |T_i ∩ C_i| / sum |T_i|.
- Macro candidate recall = mean(|T_i ∩ C_i| / |T_i|) over non-singletons.
- Full-set coverage = fraction of non-singletons with T_i contained in C_i.
- Candidate oracle score = target metric using P_i = T_i ∩ C_i. This is an upper bound for that candidate set, not achieved model performance.
- Candidate counts: mean, p50, p95, p99, maximum, zero fraction; report per source/country.
- Reduction ratio = 1 − sum |C_i| / (|S1| × |U|). Use the full task-eligible universe, not a denominator already shrunk by your blocking rules. State any official eligibility restrictions.
- Micro precision = total TP / (total TP + total FP); micro recall = total TP / (total TP + total FN). Use null when a denominator is zero. Label these as diagnostics.
- Singleton accuracy = fraction of true singletons receiving an empty prediction. Also report singleton false-positive rate, non-singleton empty-prediction rate, singleton prevalence, and predicted-empty rate.
- Report macro_F0.5 overall and by singleton/non-singleton, source, country, missingness, and match cardinality. Source diagnostics must not replace official combined-set scoring.

Measure candidate recall after every truncation/filter and at the final matcher boundary. Keep blocking-only misses visible in end-to-end recall. If there are no positives/singletons in a slice, report undefined diagnostics as null with counts.

## Thresholds, singletons, and uncertainty

Tune set decisions against complete S1 validation sets. Start with one global score threshold and deterministic tie behavior. Search observed score breakpoints or a documented grid; freeze the comparison operator. Never assume probability 0.5 is optimal.

Compare source-specific thresholds, calibrated probabilities, a singleton gate, and ambiguity rejection only as controlled ablations. Calibrate on inference-like candidates, not artificially balanced random pairs. Hard-negative sampling changes score priors; preserve deployment prevalence in calibration.

A gap between two candidates for the same S1 is not necessarily ambiguity: both may be correct. Use margins only after checking relevant cardinality, such as two competing S1 owners for a target when exclusivity is established. Top-1/top-k must preserve genuine multiple matches; quantify recall loss. Country-specific thresholds fitted to US/India need a validated unknown-country fallback and strong evidence to justify use.

Use leave-US-out and leave-India-out experiments as imperfect domain-shift stress tests; fit every learned transformation inside each training country partition. Evaluate unseen categories, Unicode/diacritics, missing country, and unusual address shapes with local fixtures. These tests do not prove performance on France; do not manufacture French labels or enrichment.

Compare paired per-S1 scores on identical folds; bootstrap relationship groups, not pairs, for uncertainty. Use additional seeds/folds when needed to resolve an actual decision. Prefer stable gains over tiny uncertain improvements, especially after many trials. Recheck threshold stability and singleton tradeoffs under plausible prevalence shifts without tuning to test guesses.
