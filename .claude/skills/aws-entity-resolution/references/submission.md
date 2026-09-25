# Final inference and submission contract

## Resolve serialization first

Read the official sample and parser. The task specifies column names and TSV format but does not specify the in-cell list encoding or empty-set encoding. Do not assume JSON, comma-separated IDs, or blank cells. Resolve ID namespaces/collisions, quoting, escaping, encoding, and ordering before implementing the codec. Block final export if unresolved.

Required files and exact ordered headers:

| File | First column | Second column |
| --- | --- | --- |
| matching_results.tsv | source1_entity_id | matched_entity_ids |
| candidate_pairs.tsv | source1_entity_id | candidate_entity_ids |

Preserve IDs as opaque strings, including leading zeros. Use source-qualified internal keys; convert only through the official external-ID mapping. Never silently collapse S2/S3 collisions.

## Frozen inference sequence

1. Load the selected configuration, data hashes, eligible model artifacts, normalization versions, and predetermined thresholds/calibration.
2. Construct permitted target indexes without fitting on test data unless explicitly allowed. Transform unknown countries safely.
3. Generate candidates, apply all selected pruning/caps, and persist the exact final matcher-input pairs with source-qualified IDs and configuration hash.
4. Score every persisted final pair, including bounded batches. Fail on missing/nonfinite scores or incomplete batches; do not silently discard candidates. For a scorer ensemble, persist each input and its actual scored union.
5. Apply the frozen set decision policy. Assert every predicted pair exists in the final matcher-input manifest. Never repair violations by adding predictions to that manifest.
6. Aggregate candidates and matches against the full S1 test ID list, retaining zero-candidate and predicted-empty entities.
7. Encode with the verified codec and write both TSVs atomically. Read them back with the official parser or a fixture-validated equivalent.

The candidate export must not come from an earlier blocking pool or only accepted matches. In a cascade, establish the final-scorer boundary from official rules and document each preceding reduction.

## Mandatory checks

- Exact filenames, header order, tab separators, valid encoding, and exactly two columns after parsing.
- Each test S1 ID appears exactly once in each file; no missing, extra, duplicate, or training-only S1 IDs.
- All target IDs belong to allowed S2/S3 test records; IDs survive a round trip unchanged.
- No duplicate target IDs within cells; empty lists use the official representation.
- For every S1, predicted IDs are a subset of exported candidate IDs.
- Exported candidate pairs equal the final matcher-input manifest, not merely a superset of predictions.
- Deterministic target ordering and tie handling; reruns produce identical files or document genuine nondeterminism before acceptance.
- Hash model/config/data/manifests/output files; record row counts, candidate counts, match cardinalities, empty-prediction rate, runtime, and peak memory.
- Flag distribution shifts for review without adjusting frozen decisions merely to resemble training prevalence.

## Required codec tests when implemented

Use the official serialization scheme to test empty predictions, zero candidates, multiple matches across sources, leading-zero IDs, Unicode IDs if legal, delimiter/quote characters if legal, duplicate rows/IDs, missing/extra S1 IDs, invalid target IDs, cross-source collisions, and predictions outside candidates. If a character is illegal, assert rejection rather than inventing an escape rule.

Test candidate-manifest equality: an export copied from an earlier larger pool must fail even if it contains all predictions. Test chunk-boundary aggregation and a partial-inference failure so incomplete files cannot pass. Include byte-level TSV verification and parser round trips.

Finalize only after metric tests, codec tests, eligibility review, and all submission assertions pass. Generating files does not itself authorize an external upload unless the active task includes submission.
