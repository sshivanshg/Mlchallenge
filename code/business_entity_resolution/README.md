# Business Entity Resolution — JAX Pipeline

MIT/Apache-2.0 stack. Matcher is a **from-scratch JAX logistic regression** (no >8B pretrained model).

## Layout

```
src/
  normalize.py              # NFKC/casefold, legal+address expansions (open-set country)
  blocking.py               # token / prefix / numeric inverted-index candidates
  features.py               # pairwise name/address similarities
  model.py                  # pure functional JAX train (jit + grad + PRNG splits)
  metrics.py                # macro F0.5 (singletons included)
  io_utils.py               # TSV read/write (sep=\\t, dtype=str, keep_default_na=False)
  generate_sample_data.py   # synthetic mini data when challenge dump is absent
  run_pipeline.py           # train / predict / run entrypoint
```

## Setup

```bash
cd code/business_entity_resolution
python3 -m pip install -r requirements.txt
```

## Data

Place official challenge files under `student_resource/dataset/`:

```
dataset/train/train_source{1,2,3}.tsv
dataset/train/train_ground_truth.tsv
dataset/test/test_source{1,2,3}.tsv
```

If those are missing, generate a schema-compatible synthetic set (includes France on test):

```bash
cd src
python3 generate_sample_data.py --out-dir ../../../student_resource/dataset
```

## End-to-end

From `code/business_entity_resolution/src`:

```bash
python3 run_pipeline.py run \
  --dataset-root ../../../student_resource/dataset \
  --model-dir ../artifacts/model \
  --output-dir ../../../student_resource/output \
  --seed 42
```

Validate:

```bash
cd ../../../student_resource
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test --check-ids
```

## Design notes

- Country is an open string label (never one-hot to {US, India}).
- `candidate_pairs.tsv` is the **final** candidate set scored by the JAX model.
- Threshold is tuned for **macro F0.5** on an S1-grouped validation split.
