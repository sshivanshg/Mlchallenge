# Reports note

Files in this tree dated from the early synthetic stub run
(`synthetic_dev_fixture_v2`, ~400 S1 rows) are **historical only**.

Authoritative experimentation must be regenerated on
`official_challenge_dataset` after:

```bash
bash scripts/download_dataset.sh
python3 code/business_entity_resolution/src/run_milestone1.py \
  --dataset-root student_resource/dataset \
  --reports-root reports \
  --seed 42
```
