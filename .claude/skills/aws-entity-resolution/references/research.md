# Candidate and matcher research

## Baseline ladder

Measure an all-empty baseline to expose singleton prevalence, then a conservative normalized-exact baseline, then multi-rule retrieval plus interpretable string features and a simple learned matcher. Exact equality is evidence, not proof: common names, shared addresses, and missing fields can collide.

Compare logistic regression and gradient-boosted trees, including LightGBM, XGBoost, or CatBoost after eligibility checks. Do not run every library without a hypothesis. Use pandas/polars, scipy sparse matrices, scikit-learn, and rapidfuzz where suitable. Consider compact embeddings or licensed transformer encoders only after classical error analysis identifies a plausible benefit and the challenge/model eligibility gates pass.

## Preserve multiple representations

Retain immutable raw fields, source identity, and transformation versions. Add views for Unicode normalization, case folding, punctuation/whitespace, ampersand/and, token order, character n-grams, legal suffixes, abbreviations, and street suffixes. Keep digit-bearing, diacritic-preserving, and optional accent-folded/transliteration-safe views separately. Never overwrite raw text.

Treat mappings as hypotheses: broad suffix stripping can erase identity; ASCII-only processing loses names; number removal confuses branches. Derive task-specific abbreviation mappings from allowed training data. Avoid downloaded country dictionaries, postal databases, or address enrichment. Local generic Unicode transforms are not external entity evidence.

Distinguish absent, empty, malformed, and literal strings such as 'nan'. Do not reward equality of two missing fields. DBA/trade-name splitting should preserve both whole-name and component views and be validated.

Use generic number/token features without assuming fixed US ZIP, Indian PIN, or French postal patterns. Country is an opaque open-set value with explicit missing/unknown handling. Avoid ordinal numeric country codes. Fit categorical encoders with safe unseen-category behavior; assess models that memorize country.

## Retrieve, union, then measure

Build source-aware retrieval using a union of complementary routes:

- Normalized exact names or name/address combinations.
- Rare token inverted indexes and token containment.
- Character n-gram TF-IDF and word TF-IDF nearest neighbors.
- Numeric/address-component overlap routes.
- Optional approximate nearest neighbors or learned embeddings when justified.
- Country-aware routes plus validated broad fallback for missing, unseen, or noisy country values.

Do not impose hard country exclusion without rule/data evidence that it is safe. Always support France and unseen strings. Never create a Cartesian all-pairs matrix unless measured sizes justify it; use sparse top-k products, indexes, batching, or bounded retrieval.

Union and deduplicate source-qualified candidates. Preserve route membership, raw retrieval score, rank, and source. Tune route budgets and per-source coverage; a single global cap can starve one source. Evaluate recall/runtime/count distributions, not classifier accuracy alone. Compare approximate retrieval with an exact subset before accepting index recall loss.

Measure the recall–cost curve before and after caps, prefilters, or cascades. Do not inject known positives into validation candidates. If training-only positive injection is necessary to learn rare matches, label it, quantify the distribution change, and retain honest retrieval misses in evaluation.

Define which stage is the final matching model before implementation. In a cascade, export exactly the candidates entering the designated final match scorer after all pruning, and preserve upstream stage manifests. For an ensemble of final scorers, define and document the actual scored union. Never fabricate the export by appending predictions afterward.

## Feature families

Ablate families rather than adding everything at once:

- Exact equality on nonmissing normalized views; edit/Levenshtein and optionally Jaro/Jaro-Winkler similarity.
- Token Jaccard, containment in both directions, rare-token overlap, character n-gram and word TF-IDF cosine.
- Prefix/suffix, acronym, reordered tokens, and legal-suffix normalized similarities.
- Number overlap/conflicts, postal-code-like token agreement, and address-component/token overlap.
- Missingness, text lengths, token counts, source, and open-set-safe country agreement/missingness.
- Name/address interactions and contradiction indicators; distinguish 'no address evidence' from 'conflicting address evidence'.
- Retrieval route, score, rank, and neighborhood density, fitted without label leakage.

Avoid ID-derived features, target encodings fitted outside folds, test-learned normalization, and statistics silently fitted on the full corpus. Numeric disagreement can be strong evidence but is not an unconditional veto when addresses are noisy or incomplete.

## Hard-negative cycle

1. Generate candidates using the same routes and budgets as inference within training folds.
2. Join complete training labels; if labels are incomplete, keep uncertain pairs unlabeled rather than declaring them negative.
3. Include confusing neighbors: similar names/different addresses; shared addresses/different names; same tokens/different entities; high-scoring false positives from a prior fit or grouped cross-fit.
4. Mix hard, ordinary retrieved, and a small diagnostic random-negative sample as useful. Control per-S1/source sampling so prolific groups do not dominate.
5. Record selection model/version, fold, labels, sample weights, and provenance. Never mine validation/test labels for training or move held-out errors into the same run's training pool.
6. Retrain and recalibrate on separate inference-like held-out candidates. Compare against the previous run using complete S1 sets.

Review suspicious 'false positives' against provided labels locally. Treat suspected label errors as unresolved unless official evidence authorizes correction; do not rewrite labels to favor a model.

## Escalation and ensembles

Use neural approaches only for residual failures they could plausibly address, such as name variants unrecoverable by current representations. Check compute, offline execution, license, parameter count, and permitted pretrained use first. A compatible license alone is insufficient permission.

Ensemble only models with measured complementary held-out errors. Learn weights/calibration from out-of-fold predictions; account for candidate differences and avoid treating absent scores as calibrated zeros. Evaluate the full union, latency, singleton false positives, and total model eligibility. Preserve the simpler model unless the improvement is robust enough to justify deployment cost.
