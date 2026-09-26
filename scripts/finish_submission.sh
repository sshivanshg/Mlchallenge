#!/usr/bin/env bash
# Wait for a sharded inference run to finish, then validate and package it.
# Usage: scripts/finish_submission.sh <run-dir> <package-dest-dir> <team-name>
# Writes <run-dir>/finish_status.json. Never touches other submission directories.
set -uo pipefail
RUN="$1"; DEST="$2"; TEAM="${3:-TEAMNAME}"
cd "$(dirname "$0")/.."
LOG="$RUN/run.log"; STATUS="$RUN/finish_status.json"
status() { printf '{"stage":"%s","ok":%s,"time_utc":"%s"}\n' "$1" "$2" "$(date -u +%FT%TZ)" > "$STATUS"; echo "[$(date -u +%T)] $1 ok=$2"; }

status waiting_for_inference true
until grep -q '"shards_new"' "$LOG" 2>/dev/null; do
  if grep -q Traceback "$LOG" 2>/dev/null || ! pgrep -f "out-dir $RUN" >/dev/null; then
    sleep 20
    grep -q '"shards_new"' "$LOG" 2>/dev/null && break
    status inference_failed_or_stopped false; exit 1
  fi
  sleep 30
done
OUT="$RUN/output"
EMPTY=$(mktemp -d)
( cd "$EMPTY" && python3 "$OLDPWD/student_resource/utils/validate_submission.py" --matching "$OLDPWD/$OUT/matching_results.tsv" \
    --test-dir "$OLDPWD/student_resource/dataset/test" --check-ids ) > "$RUN/validate_official.log" 2>&1
grep -q "^PASS" "$RUN/validate_official.log" || { status official_validator_failed false; exit 1; }
status official_validator_pass true
python3 scripts/check_candidate_ids.py "$OUT/candidate_pairs.tsv" > "$RUN/candidate_ids.log" 2>&1 || { status candidate_ids_failed false; exit 1; }
status candidate_ids_pass true
python3 -u scripts/package_submission.py --output-dir "$OUT" --team-name "$TEAM" --dest "$DEST" > "$RUN/package.log" 2>&1 \
  || { status packaging_failed false; exit 1; }
status packaged_ready_to_upload true
