"""Build selective address-token + rare name-char-ngram aux index (v5b).

Tables (S2 rows inserted before S3, same as v4 aux):
  by_addrtok(key, id)       alphabetic address tokens with corpus DF <= max_df
  addr_token_df(key, df)    DF over the full target universe (unlabeled corpus stats)
  by_ngram4(key, id)        rare compact-name character 4-grams (DF <= ngram_max_df)

Hypothesis: most no-shared-key misses are Latin S1 -> non-Latin S2/S3 with shared
rare street/locality tokens; remaining Latin typos share character n-grams.
Does not modify v2/v3/v4 indexes.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from data.provenance import require_official_dataset
from frozen_v4.policy import ADDR_STOP, MAX_ATOK, norm_addr, norm_name
from official_core import SOURCE_H, iter_tsv

REPO = SRC.parents[2]
COMPACT_RE = re.compile(r"[^a-z0-9]+")


def addr_tokens(aa: str) -> list[str]:
    return list(
        dict.fromkeys(t for t in aa.split() if len(t) >= 4 and t.isalpha() and t not in ADDR_STOP)
    )[:MAX_ATOK]


def compact_alnum(nn: str) -> str:
    return COMPACT_RE.sub("", nn or "")


def char_ngrams(compact: str, n: int = 4) -> list[str]:
    if len(compact) < n:
        return []
    return list(dict.fromkeys(compact[i : i + n] for i in range(len(compact) - n + 1)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=REPO / "student_resource" / "dataset")
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--max-df", type=int, default=5000, help="Max DF to index an address token")
    p.add_argument("--ngram-max-df", type=int, default=2000, help="Max DF to index a name 4-gram")
    p.add_argument("--skip-ngram", action="store_true")
    args = p.parse_args()
    require_official_dataset(args.dataset_root, require=(args.split,), check_hashes=True)
    out = REPO / "artifacts" / "official_index" / f"{args.split}_aux_index_v5b.sqlite"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    building = out.with_name(out.name + ".building")
    if building.exists():
        building.unlink()

    t0 = time.time()
    print("pass1: counting address-token and ngram DFs...", flush=True)
    addr_df: Counter[str] = Counter()
    ngram_df: Counter[str] = Counter()
    n = 0
    for source in (2, 3):
        path = args.dataset_root / args.split / f"{args.split}_source{source}.tsv"
        for row in iter_tsv(path, SOURCE_H):
            aa = norm_addr(row["business_address"])
            for t in set(addr_tokens(aa)):
                addr_df[t] += 1
            if not args.skip_ngram:
                c = compact_alnum(norm_name(row["business_name"]))
                for g in char_ngrams(c, 4):
                    ngram_df[g] += 1
            n += 1
            if n % 500_000 == 0:
                print(f"  counted {n:,} addr_keys={len(addr_df):,} ngram_keys={len(ngram_df):,} "
                      f"{time.time()-t0:.0f}s", flush=True)

    keep_addr = {k for k, v in addr_df.items() if v <= args.max_df}
    keep_ng = {k for k, v in ngram_df.items() if v <= args.ngram_max_df} if not args.skip_ngram else set()
    print(f"pass1 done: targets={n:,} keep_addr={len(keep_addr):,}/{len(addr_df):,} "
          f"keep_ngram={len(keep_ng):,}/{len(ngram_df):,} {time.time()-t0:.0f}s", flush=True)

    db = sqlite3.connect(building)
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("CREATE TABLE by_addrtok (key TEXT, id TEXT)")
    db.execute("CREATE TABLE addr_token_df (key TEXT PRIMARY KEY, df INTEGER)")
    db.execute("CREATE TABLE by_ngram4 (key TEXT, id TEXT)")
    db.execute("CREATE TABLE ngram4_df (key TEXT PRIMARY KEY, df INTEGER)")
    db.executemany("INSERT INTO addr_token_df VALUES (?,?)", [(k, addr_df[k]) for k in keep_addr])
    if keep_ng:
        db.executemany("INSERT INTO ngram4_df VALUES (?,?)", [(k, ngram_df[k]) for k in keep_ng])
    db.commit()

    print("pass2: inserting postings...", flush=True)
    counts = {"by_addrtok": 0, "by_ngram4": 0}
    buf_a: list[tuple[str, str]] = []
    buf_g: list[tuple[str, str]] = []
    n2 = 0
    for source in (2, 3):
        path = args.dataset_root / args.split / f"{args.split}_source{source}.tsv"
        for row in iter_tsv(path, SOURCE_H):
            eid = row["entity_id"]
            aa = norm_addr(row["business_address"])
            for t in addr_tokens(aa):
                if t in keep_addr:
                    buf_a.append((t, eid))
            if keep_ng:
                c = compact_alnum(norm_name(row["business_name"]))
                for g in char_ngrams(c, 4):
                    if g in keep_ng:
                        buf_g.append((g, eid))
            n2 += 1
            if n2 % 200_000 == 0:
                if buf_a:
                    db.executemany("INSERT INTO by_addrtok VALUES (?,?)", buf_a)
                    counts["by_addrtok"] += len(buf_a)
                    buf_a.clear()
                if buf_g:
                    db.executemany("INSERT INTO by_ngram4 VALUES (?,?)", buf_g)
                    counts["by_ngram4"] += len(buf_g)
                    buf_g.clear()
                db.commit()
                print(f"  {n2:,} {counts} {time.time()-t0:.0f}s", flush=True)
    if buf_a:
        db.executemany("INSERT INTO by_addrtok VALUES (?,?)", buf_a)
        counts["by_addrtok"] += len(buf_a)
    if buf_g:
        db.executemany("INSERT INTO by_ngram4 VALUES (?,?)", buf_g)
        counts["by_ngram4"] += len(buf_g)
    db.commit()

    print("  creating indexes...", flush=True)
    db.execute("CREATE INDEX idx_by_addrtok ON by_addrtok(key)")
    db.execute("CREATE INDEX idx_by_ngram4 ON by_ngram4(key)")
    db.commit()
    for table, expected in counts.items():
        got = db.execute(f"SELECT MAX(rowid) FROM {table}").fetchone()[0] or 0
        if got != expected:
            raise RuntimeError(f"{table}: {got} != {expected}")
    db.close()
    building.replace(out)
    stats = {
        "split": args.split,
        "targets": n,
        "max_df": args.max_df,
        "ngram_max_df": args.ngram_max_df,
        "skip_ngram": args.skip_ngram,
        "keep_addr_keys": len(keep_addr),
        "keep_ngram_keys": len(keep_ng),
        **counts,
        "runtime_sec": time.time() - t0,
        "bytes": out.stat().st_size,
        "complete": True,
        "note": "addr_token_df / ngram4_df are unlabeled target-corpus statistics",
    }
    out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
