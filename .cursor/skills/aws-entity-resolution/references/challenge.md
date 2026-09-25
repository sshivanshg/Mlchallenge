# Challenge contract and provenance

## Supplied facts

- S1 is deduplicated; each S1 entity has zero, one, or multiple S2/S3 matches.
- Fields: entity_id, business_name, business_address, country.
- Noise includes spelling, abbreviations, suffixes, punctuation, reordering, DBA/trade names, transliteration, missing address components, and landmarks.
- Training countries are US/India; test additionally includes France.
- Optimize macro F_0.5 across S1 entities. True singleton plus empty prediction scores 1; true singleton plus any prediction scores 0.
- Final files are matching_results.tsv and candidate_pairs.tsv, each with exactly one row per test S1 entity. Predicted matches must be included in final model candidates.
- Final learned model limit: 8B parameters; compatible license must be MIT or Apache 2.0.

## Inspect before assuming

Create a local contract note from supplied official rules, sample submission, evaluator, labels, and data dictionary. Record:

- File schemas, label completeness, unmatched-label semantics, source membership, ID uniqueness and cross-source collisions.
- Whether S2/S3 may map to multiple S1 entities; whether labels identify all matches or only selected pairs. Absence of a positive label is not automatically a negative.
- Whether sets combine both target sources or official scoring treats them separately; entity weighting and empty-set behavior.
- Exact list delimiter, quoting, escaping, empty representation, ID namespaces, encoding, and required order in output cells.
- Pretrained weights, offline augmentation, transductive fitting, ensemble parameter accounting, and permissible resource use.

Do not invent unknown rules. Use the supplied combined-set interpretation provisionally; block affected training/evaluation/submission decisions if official semantics are missing or contradictory. Continue unrelated inspection.

Profile counts by source/country, missingness, ID integrity, label cardinality, singleton prevalence, duplicate text, script/character distribution, lengths, numbers, and name/address disagreement. Inspect representative supplied records locally. Separate training EDA from permitted unlabeled test schema/drift checks; never infer test labels or tune on guessed test identities.

## External-data firewall

Only challenge data may supply entity evidence. Prohibit business search, registries, ER services, geocoders, maps, external address enrichment, online translation/transliteration services, internet augmentation, and external corpora/dictionaries used as entity enrichment. Do not query even a single difficult record online.

Default ER pipelines to network-disabled execution. Allowlist input files and hash them. Audit new dependencies for downloads, telemetry, hosted inference, pretrained assets, gazetteers, and hidden address datasets. Keep record content out of network requests and remote logs. Use local transformations; any offline augmentation must derive only from training-fold challenge records and be explicitly permitted.

Generic software documentation or license metadata is not entity evidence. Access it only if challenge rules permit such access, with no record text/IDs in queries. Separate approved dependency/artifact acquisition from offline ER runs. If rules are unclear, continue with local verified resources and ask a narrowly scoped question before the affected action.

## Model eligibility gate

Before recommending or acquiring any learned artifact, record exact name/revision, parameter count, license text/location/hash, weight license, base-model license, adapter/derivative terms, and whether official rules permit pretrained use. A framework license alone does not establish a checkpoint's license. A model card assertion without compatible artifact terms is insufficient.

Do not list speculative encoder recommendations. Classical estimators are the default experimental path, subject to verified package and resulting artifact eligibility. License permissibility does not establish permission to use external training data. Do not train/fine-tune on external corpora.

For ensembles, conservatively account for all deployed learned components, including encoders and rerankers, until organizers clarify the parameter ceiling. Quantization does not reduce parameter count; mixture-of-experts totals include inactive experts. Treat unclear parameter accounting or license terms as unresolved, not as an exemption.
