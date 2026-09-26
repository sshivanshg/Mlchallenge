"""Stream candidate_pairs.tsv and confirm every candidate ID exists in test S2/S3."""

import json
import sys
import time

TEST = "student_resource/dataset/test"


def main():
    t0 = time.time()
    valid = set()
    for fn in ("test_source2.tsv", "test_source3.tsv"):
        with open(f"{TEST}/{fn}", encoding="utf-8") as f:
            next(f)
            for line in f:
                valid.add(line.split("\t", 1)[0])
    rows = n = bad = 0
    with open(sys.argv[1], encoding="utf-8") as f:
        next(f)
        for line in f:
            rows += 1
            ids = line.rstrip("\n").split("\t", 1)[1]
            for x in ids.split(",") if ids else ():
                n += 1
                bad += x not in valid
    res = {"rows": rows, "candidate_ids": n, "unknown_ids": bad, "valid_universe": len(valid), "sec": round(time.time() - t0)}
    print(json.dumps(res))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
