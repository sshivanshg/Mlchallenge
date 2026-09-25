# Business Entity Resolution — research engine

Primary methodology: vendored Agent Skill  
[`.agents/skills/aws-entity-resolution/SKILL.md`](../../.agents/skills/aws-entity-resolution/SKILL.md)

Optimize **macro F0.5** (set-level, singletons included). Fair play: competition data only.

## Layout

```
src/
  data/            # load, EDA, synthetic fixtures (dev only)
  normalization/   # multi-view text normalization
  blocking/        # multi-route candidate generation
  features/        # interpretable pairwise features
  models/          # empty / exact / weighted / logistic / LightGBM
  evaluation/      # metric wrappers, splits, diagnostics
  inference/       # (reserved) frozen export
  run_milestone1.py
configs/
tests/
experiments/       # local run artifacts (optional)
```

Reports land in repo-root `reports/`.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## First research milestone

```bash
# Metric safeguards
python3 -m unittest discover -s ../../.agents/skills/aws-entity-resolution/scripts -p 'test_*.py'
python3 -m unittest discover -s ../tests -p 'test_*.py'

# If official dumps are absent, this builds synthetic_dev_fixture_v2 (NOT official):
python3 src/run_milestone1.py \
  --dataset-root ../../student_resource/dataset \
  --reports-root ../../reports \
  --make-hard-synthetic \
  --seed 42
```

When official TSVs are placed under `student_resource/dataset/{train,test}/`, omit `--make-hard-synthetic` and re-run.

## Notes

- Country is open-set (never hard-filter to US/India).
- Validation splits are component-disjoint S1 manifests under `experiments/splits/`.
- Experiment ledger: `reports/experiments/experiments.csv` (append-only).
