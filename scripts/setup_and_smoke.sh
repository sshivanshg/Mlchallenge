#!/usr/bin/env bash
# End-to-end check: skill metric tests + official dataset presence + validator.
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

DATA="$ROOT/student_resource/dataset"
NEED=(
  "$DATA/train/train_source1.tsv"
  "$DATA/train/train_source2.tsv"
  "$DATA/train/train_source3.tsv"
  "$DATA/train/train_ground_truth.tsv"
  "$DATA/test/test_source1.tsv"
  "$DATA/test/test_source2.tsv"
  "$DATA/test/test_source3.tsv"
)

missing=0
for f in "${NEED[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "MISSING: $f"
    missing=1
  fi
done

if [[ "$missing" -ne 0 ]]; then
  echo "== official dataset missing; downloading from GitHub release dataset-v1 =="
  bash "$ROOT/scripts/download_dataset.sh"
fi

echo "== official dataset provenance (schema + sha256 manifest) =="
python3 "$ROOT/code/business_entity_resolution/src/data/provenance.py" \
  --dataset-root "$DATA" \
  --require train test

echo "== official dataset present =="
du -sh "$DATA" "$DATA/train" "$DATA/test"

echo "== submission validator (format only; needs existing output/ or skip) =="
OUT="$ROOT/student_resource/output"
if [[ -f "$OUT/matching_results.tsv" && -f "$OUT/candidate_pairs.tsv" ]]; then
  cd "$ROOT/student_resource"
  python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
else
  echo "No output/ yet — skipping validator (run inference first)."
fi

echo "OK — skill + official dataset ready"
