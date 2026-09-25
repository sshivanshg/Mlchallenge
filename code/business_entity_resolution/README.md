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

## Reproduce the submission (final frozen policy `v4all_cap300_lgbm_m2_thr0.70`)

Steps 1–3 below build the shared indexes. Then:

```bash
S=code/business_entity_resolution/src
python3 -u $S/build_aux_index_v4.py --split train     # address/compact-name keys, ~4 min
python3 -u $S/build_aux_index_v4.py --split test
python3 -u $S/run_matcher_v4.py --variant v4all --fit-s1 15000 --experiments M1b,M2 --tag v4all_m2
python3 -u $S/verify_infer_v4.py --n 1000             # frozen_v4 == research path
python3 -u $S/run_infer_v4.py --split test --workers 4 --shard-size 20000 \
    --out-dir artifacts/submissions/v4all_m2_v1        # ~1.9 h with 4 workers
python3 scripts/package_submission.py --output-dir artifacts/submissions/v4all_m2_v1/output --team-name <TEAM>
```

## Previous policy (fallback) `v3b_cap300_lgbm_thr0.98_nogate`

Run from the repository root. Every step reads only `student_resource/dataset/`
and verifies it against `configs/official_dataset_manifest.json`. Measured on
4 vCPU / 15 GiB RAM; indexes are SQLite files under `artifacts/official_index/`.

```bash
S=code/business_entity_resolution/src
# 1. Frozen component-disjoint fit/select/assess split + train S2/S3 index (~5 min each)
python3 -u $S/run_official_bounded.py                 # writes *_folds.sqlite, targets_index.sqlite
python3 -u $S/run_official_stage4.py --rebuild-index   # writes targets_index_v2.sqlite
# 2. Composite-key indexes (pairs of name tokens; name token x address number), ~4 min each
python3 -u $S/build_pair_index.py --split train
python3 -u $S/build_pair_index.py --split test
# 3. Test S2/S3 index (~6 min)
python3 -c "import sys; sys.path.insert(0, '$S'); from pathlib import Path; \
from official_core import build_target_index_v2 as b; \
print(b(Path('student_resource/dataset'), Path('artifacts/official_index/test_targets_index_v2.sqlite'), split='test'))"
# 4. Train LightGBM on fit-fold candidates, tune threshold on the selection fold
python3 -u $S/run_matcher_v3.py --variant v3b --cap 300   # writes matcher_v3b_cap300.pkl
# 5. Sharded, resumable test inference (~2.3 h with 4 workers) -> <out-dir>/output/*.tsv
python3 -u $S/run_infer_v3b.py --split test --workers 4 --shard-size 20000 \
    --out-dir artifacts/submissions/v3b_cap300_lgbm_v1
```

`run_infer_v3b.py` refuses a model whose SHA-256 differs from the frozen bundle,
reuses a shard only when its manifest `run_hash` (code, model, config, and input
hashes) matches, and fails on any gap, duplicate, reordering, or match outside the
exported candidate list. `verify_infer_v3b.py` checks the frozen policy against
the research reference path. Retrieval, features, and the decision rule used for
the submission live in `src/frozen_v3b/policy.py`.

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
