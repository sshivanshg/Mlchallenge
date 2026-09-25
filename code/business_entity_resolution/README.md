# Business Entity Resolution — research engine

Primary methodology: vendored Agent Skill  
[`.agents/skills/aws-entity-resolution/SKILL.md`](../../.agents/skills/aws-entity-resolution/SKILL.md)

Optimize **macro F0.5** (set-level, singletons included). Fair play: competition data only.

## Data (official challenge TSVs)

Use the **real** dataset under `student_resource/dataset/`. Do **not** use synthetic generators for scored research.

```bash
# From repo root — downloads release assets and extracts TSVs
bash scripts/download_dataset.sh
```

Release: https://github.com/sshivanshg/Mlchallenge/releases/tag/dataset-v1  
Details: [`data/README.md`](../../data/README.md)

Expected layout:

```text
student_resource/dataset/train/train_source{1,2,3}.tsv
student_resource/dataset/train/train_ground_truth.tsv
student_resource/dataset/test/test_source{1,2,3}.tsv
```

## Layout

```
src/
  data/            # load, EDA (official data)
  normalization/   # multi-view text normalization
  blocking/        # multi-route candidate generation
  features/        # interpretable pairwise features
  models/          # empty / exact / weighted / logistic / LightGBM
  evaluation/      # metric wrappers, splits, diagnostics
  inference/       # (reserved) frozen export
  run_milestone1.py
configs/
tests/
```

Reports land in repo-root `reports/`.

## Setup

```bash
python3 -m pip install -r requirements.txt
bash scripts/download_dataset.sh   # if dataset/ not already present
```

## First research milestone (official data)

```bash
python3 -m unittest discover -s ../../.agents/skills/aws-entity-resolution/scripts -p 'test_*.py'
python3 -m unittest discover -s ../tests -p 'test_*.py'

python3 src/run_milestone1.py \
  --dataset-root ../../student_resource/dataset \
  --reports-root ../../reports \
  --seed 42
```

Do **not** pass any synthetic generator flags. Competition entrypoints call
`data.provenance.require_official_dataset` (schema + SHA-256 manifest) and will
fail closed on fixtures/stubs/non-official paths.
