"""Build a separate SQLite index of selective composite keys over S2+S3.

Tables (rows inserted S2 file then S3 file, like targets_index_v2):
  by_pair(key='tokA|tokB', id)   unordered pairs of distinct informative name tokens
  by_toknum(key='tok#num', id)   informative name token x address number (>=3 digits)
Does not modify targets_index_v2.sqlite.
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
from official_core import SOURCE_H, iter_tsv, norm_addr, norm_name, nums, sig_tokens

REPO = SRC.parents[2]
MAX_TOKENS = 8
MAX_NUMS = 6


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=REPO / "student_resource" / "dataset")
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    require_official_dataset(args.dataset_root, require=(args.split,), check_hashes=True)
    out = args.out or (REPO / "artifacts" / "official_index" / f"{args.split}_pair_index_v3.sqlite")
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    t0 = time.time()
    db = sqlite3.connect(out)
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("CREATE TABLE by_pair (key TEXT, id TEXT)")
    db.execute("CREATE TABLE by_toknum (key TEXT, id TEXT)")
    bp, bt = [], []
    n = n_pair = n_tn = 0
    for source in (2, 3):
        for row in iter_tsv(args.dataset_root / args.split / f"{args.split}_source{source}.tsv", SOURCE_H):
            eid = row["entity_id"]
            toks = list(dict.fromkeys(sig_tokens(norm_name(row["business_name"]))))[:MAX_TOKENS]
            for a, b in combinations(toks, 2):
                bp.append(("|".join(sorted((a, b))), eid))
            anums = list(dict.fromkeys(nums(norm_addr(row["business_address"]))))[:MAX_NUMS]
            for t in toks:
                for num in anums:
                    bt.append((f"{t}#{num}", eid))
            n += 1
            if n % 200000 == 0:
                db.executemany("INSERT INTO by_pair VALUES (?,?)", bp)
                db.executemany("INSERT INTO by_toknum VALUES (?,?)", bt)
                n_pair += len(bp)
                n_tn += len(bt)
                bp.clear()
                bt.clear()
                db.commit()
                print(f"  {n:,} targets  pairs={n_pair:,} toknum={n_tn:,}  {time.time()-t0:.0f}s", flush=True)
    db.executemany("INSERT INTO by_pair VALUES (?,?)", bp)
    db.executemany("INSERT INTO by_toknum VALUES (?,?)", bt)
    n_pair += len(bp)
    n_tn += len(bt)
    db.commit()
    print("  creating indexes...", flush=True)
    db.execute("CREATE INDEX idx_pair ON by_pair(key)")
    db.execute("CREATE INDEX idx_toknum ON by_toknum(key)")
    db.commit()
    db.close()
    stats = {"split": args.split, "targets": n, "pair_rows": n_pair, "toknum_rows": n_tn,
             "max_tokens": MAX_TOKENS, "max_nums": MAX_NUMS, "runtime_sec": time.time() - t0,
             "bytes": out.stat().st_size}
    (out.with_suffix(".stats.json")).write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
