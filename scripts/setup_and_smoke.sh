#!/usr/bin/env bash
# End-to-end smoke for the installed aws-entity-resolution skill + pipeline.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== skill metric unit tests =="
python3 -m unittest discover -s .agents/skills/aws-entity-resolution/scripts -p 'test_*.py' -v

echo "== mirror integrity (agents == claude == cursor) =="
diff -rq .agents/skills/aws-entity-resolution .claude/skills/aws-entity-resolution \
  --exclude __pycache__
diff -rq .agents/skills/aws-entity-resolution .cursor/skills/aws-entity-resolution \
  --exclude __pycache__

echo "== synthetic data + train/predict (if dataset missing) =="
DATA="$ROOT/student_resource/dataset"
OUT="$ROOT/student_resource/output"
MODEL="$ROOT/code/business_entity_resolution/artifacts/model"
SRC="$ROOT/code/business_entity_resolution/src"
if [[ ! -f "$DATA/test/test_source1.tsv" ]]; then
  python3 "$SRC/generate_sample_data.py" --out-dir "$DATA" --n-train 120 --n-test 80
fi
python3 "$SRC/run_pipeline.py" run \
  --dataset-root "$DATA" \
  --model-dir "$MODEL" \
  --output-dir "$OUT" \
  --seed 42

echo "== submission validator =="
cd "$ROOT/student_resource"
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test \
  --check-ids

echo "OK — skill + pipeline smoke complete"
