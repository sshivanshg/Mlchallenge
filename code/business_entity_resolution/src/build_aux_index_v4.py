"""Build auxiliary retrieval keys over S2+S3 (research; does not touch v2/v3 indexes).

Tables (rows inserted S2 file then S3 file):
  by_numtok(key='num#addrtok', id)  address number (>=3 digits) x informative address token
  by_numnum(key='numA#numB', id)    pairs of distinct address numbers (>=3 digits)
  by_cpre8(key=compact_name[:8], id) first 8 chars of the space-free normalized name
Address tokens: alphabetic, length >= 4, not generic address words.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from itertools import combinations
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from data.provenance import require_official_dataset
from frozen_v3b.policy import norm_addr, norm_name, nums
from official_core import SOURCE_H, iter_tsv

REPO = SRC.parents[2]
MAX_ATOK, MAX_NUMS = 6, 3
ADDR_STOP = {
    "road", "street", "avenue", "lane", "drive", "boulevard", "floor", "near", "opposite", "opp", "building",
    "block", "sector", "house", "office", "shop", "plot", "door", "main", "cross", "suite", "unit", "apartment",
    "nagar", "colony", "village", "district", "state", "city", "west", "east", "north", "south", "india", "united",
    "states", "america", "france", "complex", "tower", "phase", "stage", "ground", "first", "second", "third",
    "limited", "private", "company", "post", "area", "industrial", "estate", "park", "rue", "avenue", "boulevard",
}


def addr_tokens(aa: str) -> list[str]:
    return list(dict.fromkeys(t for t in aa.split() if len(t) >= 4 and t.isalpha() and t not in ADDR_STOP))[:MAX_ATOK]


def compact_prefix(nn: str) -> str:
    c = "".join(nn.split())
    return c[:8] if len(c) >= 8 else ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=REPO / "student_resource" / "dataset")
    p.add_argument("--split", default="train", choices=["train", "test"])
    args = p.parse_args()
    require_official_dataset(args.dataset_root, require=(args.split,), check_hashes=True)
    out = REPO / "artifacts" / "official_index" / f"{args.split}_aux_index_v4.sqlite"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    building = out.with_name(out.name + ".building")
    if building.exists():
        building.unlink()
    t0 = time.time()
    db = sqlite3.connect(building)
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    for t in ("by_numtok", "by_numnum", "by_cpre8"):
        db.execute(f"CREATE TABLE {t} (key TEXT, id TEXT)")
    counts = {"by_numtok": 0, "by_numnum": 0, "by_cpre8": 0}
    buf = {k: [] for k in counts}
    n = 0
    for source in (2, 3):
        for row in iter_tsv(args.dataset_root / args.split / f"{args.split}_source{source}.tsv", SOURCE_H):
            eid = row["entity_id"]
            aa = norm_addr(row["business_address"])
            an = list(dict.fromkeys(nums(aa)))[:MAX_NUMS]
            for num in an:
                for t in addr_tokens(aa):
                    buf["by_numtok"].append((f"{num}#{t}", eid))
            for a, b in combinations(sorted(an), 2):
                buf["by_numnum"].append((f"{a}#{b}", eid))
            cp = compact_prefix(norm_name(row["business_name"]))
            if cp:
                buf["by_cpre8"].append((cp, eid))
            n += 1
            if n % 200000 == 0:
                for k, v in buf.items():
                    db.executemany(f"INSERT INTO {k} VALUES (?,?)", v)
                    counts[k] += len(v)
                    v.clear()
                db.commit()
                print(f"  {n:,} {counts} {time.time()-t0:.0f}s", flush=True)
    for k, v in buf.items():
        db.executemany(f"INSERT INTO {k} VALUES (?,?)", v)
        counts[k] += len(v)
    db.commit()
    print("  creating indexes...", flush=True)
    for k in counts:
        db.execute(f"CREATE INDEX idx_{k} ON {k}(key)")
    db.commit()
    for k, expected in counts.items():
        got = db.execute(f"SELECT MAX(rowid) FROM {k}").fetchone()[0] or 0
        if got != expected:
            raise RuntimeError(f"{k}: {got} != {expected}")
    db.close()
    building.replace(out)
    stats = {"split": args.split, "targets": n, **counts, "runtime_sec": time.time() - t0,
             "bytes": out.stat().st_size, "complete": True, "max_addr_tokens": MAX_ATOK, "max_nums": MAX_NUMS}
    out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
