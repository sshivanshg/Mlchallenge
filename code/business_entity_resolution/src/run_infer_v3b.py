"""Deterministic, resumable, sharded inference for frozen policy v3b_cap300_lgbm.

S1 rows are streamed in file order and cut into fixed-size shards. Each worker
opens its own read-only SQLite connections, scores candidates in batches with a
single-threaded LightGBM booster, and writes shard files atomically followed by
a completion manifest. A shard is reused on resume only if its manifest run_hash
and file digests match. The merge step re-streams S1 IDs and fails on any gap,
reordering, duplicate, or match outside the exported candidate list.

  python3 run_infer_v3b.py --split test --workers 4 --out-dir artifacts/submissions/v3b_cap300_lgbm
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import pickle
import resource
import sqlite3
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from data.provenance import load_manifest, require_official_dataset
from frozen_v3b import policy as P
from official_core import SOURCE_H, iter_tsv

REPO = SRC.parents[2]
DEFAULT_MODEL = REPO / "artifacts" / "official_index" / "matcher_v3b_cap300.pkl"
MODEL_SHA256 = "bf9430b2c4a07b24a6c1c897cccc62e027774a5f9d85b03392b13886f94a3807"

_W: dict = {}


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def load_booster(model_path: Path):
    with model_path.open("rb") as f:
        bundle = pickle.load(f)
    if bundle.get("variant") != "v3b" or int(bundle.get("cap")) != P.CAP:
        raise RuntimeError(f"bundle policy mismatch: {bundle.get('variant')} cap={bundle.get('cap')}")
    if bundle.get("feature_names") != P.FEATURE_NAMES:
        raise RuntimeError("bundle feature order differs from frozen policy")
    res = bundle["results"]["lightgbm"]
    if abs(res["threshold"] - P.THRESHOLD) > 1e-12 or res["gate"]:
        raise RuntimeError(f"bundle decision rule differs: {res['threshold']} gate={res['gate']}")
    return bundle["lightgbm"].booster_


def _init_worker(index_path, pair_path, model_path, bounds, pair_bounds, n_targets):
    db = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True, check_same_thread=False)
    pdb = sqlite3.connect(f"file:{pair_path}?mode=ro", uri=True, check_same_thread=False)
    for c in (db, pdb):
        c.execute("PRAGMA cache_size=-262144")
    _W["db"] = db
    _W["ret"] = P.Retriever(db, pdb, n_targets, bounds, pair_bounds)
    _W["booster"] = load_booster(Path(model_path))


def score_rows(rows, batch: int = 256):
    """Return list of (sid, candidates, matches, probs) for S1 rows, in order."""
    ret, db, booster = _W["ret"], _W["db"], _W["booster"]
    out = []
    for i in range(0, len(rows), batch):
        chunk = rows[i : i + batch]
        cands = [ret.candidates(r["business_name"], r["business_address"], r["country"]) for r in chunk]
        recs = P.load_records(db, {t for c in cands for t in c})
        feats, owner = [], []
        for j, (r, c) in enumerate(zip(chunk, cands)):
            a = P.s1_view(r)
            for t in c:
                feats.append(P.feat_vec(a, recs[t]))
                owner.append(j)
        probs = booster.predict(np.stack(feats), num_threads=1) if feats else np.zeros(0)
        k = 0
        for j, (r, c) in enumerate(zip(chunk, cands)):
            pr = probs[k : k + len(c)]
            k += len(c)
            out.append((r["entity_id"], c, [t for t, p in zip(c, pr) if p >= P.THRESHOLD], pr))
    return out


def process_shard(shard_id: int, rows: list[dict], shard_dir: str, run_hash: str) -> dict:
    t0 = time.time()
    sd = Path(shard_dir)
    base = sd / f"shard_{shard_id:05d}"
    m_tmp, c_tmp = base.with_suffix(".match.tsv.tmp"), base.with_suffix(".cand.tsv.tmp")
    results = score_rows(rows)
    stats = Counter()
    by_country = {}
    ccount = []
    with m_tmp.open("w", encoding="utf-8", newline="\n") as mf, c_tmp.open("w", encoding="utf-8", newline="\n") as cf:
        for r, (sid, c, m, _) in zip(rows, results):
            if sid != r["entity_id"]:
                raise RuntimeError("row order mismatch inside shard")
            mf.write(f"{sid}\t{','.join(m)}\n")
            cf.write(f"{sid}\t{','.join(c)}\n")
            ctry = r["country"]
            d = by_country.setdefault(ctry, Counter())
            d["s1"] += 1
            d["links"] += len(m)
            d["empty_pred"] += int(not m)
            d["cands"] += len(c)
            d["zero_cand"] += int(not c)
            stats["links"] += len(m)
            ccount.append(len(c))
    m_path, c_path = base.with_suffix(".match.tsv"), base.with_suffix(".cand.tsv")
    os.replace(m_tmp, m_path)
    os.replace(c_tmp, c_path)
    manifest = {
        "shard_id": shard_id,
        "run_hash": run_hash,
        "n_s1": len(rows),
        "first_id": rows[0]["entity_id"],
        "last_id": rows[-1]["entity_id"],
        "links": stats["links"],
        "cand_mean": float(np.mean(ccount)),
        "cand_max": int(max(ccount)),
        "by_country": {k: dict(v) for k, v in by_country.items()},
        "match_sha256": sha256_file(m_path),
        "cand_sha256": sha256_file(c_path),
        "match_bytes": m_path.stat().st_size,
        "cand_bytes": c_path.stat().st_size,
        "sec": time.time() - t0,
        "worker_pid": os.getpid(),
        "worker_peak_rss_mb": peak_rss_mb(),
    }
    mtmp = base.with_suffix(".json.tmp")
    mtmp.write_text(json.dumps(manifest, indent=1) + "\n")
    os.replace(mtmp, base.with_suffix(".json"))
    return manifest


def shard_complete(shard_dir: Path, shard_id: int, run_hash: str, verify_digest: bool) -> dict | None:
    base = shard_dir / f"shard_{shard_id:05d}"
    mp = base.with_suffix(".json")
    if not mp.exists():
        return None
    try:
        m = json.loads(mp.read_text())
    except json.JSONDecodeError:
        return None
    if m.get("run_hash") != run_hash:
        return None
    for kind in ("match", "cand"):
        f = base.with_suffix(f".{kind}.tsv")
        if not f.exists() or f.stat().st_size != m[f"{kind}_bytes"]:
            return None
        if verify_digest and sha256_file(f) != m[f"{kind}_sha256"]:
            return None
    return m


def iter_s1(path: Path, ids_filter: set | None, limit: int):
    n = 0
    for row in iter_tsv(path, SOURCE_H):
        if ids_filter is not None and row["entity_id"] not in ids_filter:
            continue
        yield row
        n += 1
        if limit and n >= limit:
            return


def merge(shard_dir: Path, out_dir: Path, s1_path: Path, ids_filter, limit, n_shards, shard_size, run_hash):
    out_dir.mkdir(parents=True, exist_ok=True)
    m_final, c_final = out_dir / "matching_results.tsv", out_dir / "candidate_pairs.tsv"
    m_tmp, c_tmp = out_dir / "matching_results.tsv.tmp", out_dir / "candidate_pairs.tsv.tmp"
    expected = iter_s1(s1_path, ids_filter, limit)
    n = links = empty = 0
    with m_tmp.open("w", encoding="utf-8", newline="\n") as mf, c_tmp.open("w", encoding="utf-8", newline="\n") as cf:
        mf.write("source1_entity_id\tmatched_entity_ids\n")
        cf.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid_ in range(n_shards):
            m = shard_complete(shard_dir, sid_, run_hash, verify_digest=True)
            if m is None:
                raise RuntimeError(f"shard {sid_} incomplete or stale at merge")
            base = shard_dir / f"shard_{sid_:05d}"
            with base.with_suffix(".match.tsv").open(encoding="utf-8") as fm, base.with_suffix(".cand.tsv").open(encoding="utf-8") as fc:
                for lm, lc in zip(fm, fc):
                    row = next(expected, None)
                    s_m, ids_m = lm.rstrip("\n").split("\t")
                    s_c, ids_c = lc.rstrip("\n").split("\t")
                    if row is None or not (s_m == s_c == row["entity_id"]):
                        raise RuntimeError(f"coverage/order mismatch at {s_m}")
                    cl = ids_c.split(",") if ids_c else []
                    ml = ids_m.split(",") if ids_m else []
                    if len(set(cl)) != len(cl) or len(set(ml)) != len(ml):
                        raise RuntimeError(f"duplicate ids for {s_m}")
                    if not set(ml) <= set(cl):
                        raise RuntimeError(f"match outside candidates for {s_m}")
                    if any(not (x.startswith("S2-") or x.startswith("S3-")) for x in cl):
                        raise RuntimeError(f"non S2/S3 id for {s_m}")
                    mf.write(lm)
                    cf.write(lc)
                    n += 1
                    links += len(ml)
                    empty += int(not ml)
    if next(expected, None) is not None:
        raise RuntimeError("S1 rows remain after last shard: incomplete coverage")
    os.replace(m_tmp, m_final)
    os.replace(c_tmp, c_final)
    return {"n_s1": n, "links": links, "empty_predictions": empty,
            "matching_sha256": sha256_file(m_final), "candidate_sha256": sha256_file(c_final),
            "matching_bytes": m_final.stat().st_size, "candidate_bytes": c_final.stat().st_size}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=REPO / "student_resource" / "dataset")
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--index", type=Path, default=None)
    p.add_argument("--pair-index", type=Path, default=None)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--shard-size", type=int, default=20000)
    p.add_argument("--limit", type=int, default=0, help="first N S1 rows only (benchmark/testing)")
    p.add_argument("--ids-file", type=Path, default=None, help="JSON {ids:[...]} or text list; keeps file order")
    p.add_argument("--stop-after-shards", type=int, default=0, help="testing: exit after N new shards")
    args = p.parse_args()
    t_all = time.time()

    prov = require_official_dataset(args.dataset_root, require=(args.split,), check_hashes=True)
    mf = load_manifest()["files"]
    s1_rel = f"{args.split}/{args.split}_source1.tsv"
    s1_path = args.dataset_root / s1_rel
    wd = REPO / "artifacts" / "official_index"
    index = args.index or (wd / ("targets_index_v2.sqlite" if args.split == "train" else "test_targets_index_v2.sqlite"))
    pair = args.pair_index or (wd / f"{args.split}_pair_index_v3.sqlite")
    pair_stats = json.loads(pair.with_suffix(".stats.json").read_text())
    if pair_stats.get("split") != args.split:
        raise SystemExit(f"pair index split mismatch: {pair_stats.get('split')}")
    if args.split == "test" and not pair_stats.get("complete"):
        raise SystemExit("test pair index lacks completion metadata")
    model_sha = sha256_file(args.model)
    if model_sha != MODEL_SHA256:
        raise SystemExit(f"model sha256 {model_sha} != frozen {MODEL_SHA256}")
    load_booster(args.model)

    ids_filter = None
    if args.ids_file:
        txt = args.ids_file.read_text()
        ids_filter = set(json.loads(txt)["ids"]) if txt.lstrip().startswith("{") else set(txt.split())

    db = sqlite3.connect(f"file:{index}?mode=ro", uri=True)
    n_targets = db.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
    expected_targets = sum(
        sum(1 for _ in iter_tsv(args.dataset_root / args.split / f"{args.split}_source{s}.tsv", SOURCE_H)) for s in (2, 3)
    ) if args.split == "test" else n_targets
    if n_targets != expected_targets or pair_stats["targets"] != n_targets:
        raise SystemExit(f"target universe mismatch: index={n_targets} pair={pair_stats['targets']} expected={expected_targets}")
    bounds = P.s3_boundaries(db, P.ROUTE_TABLES)
    db.close()
    pdb = sqlite3.connect(f"file:{pair}?mode=ro", uri=True)
    pair_bounds = P.s3_boundaries(pdb, P.PAIR_TABLES)
    pdb.close()

    config = {
        "policy_id": P.POLICY_ID,
        "policy_code_sha256": sha256_file(Path(P.__file__)),
        "writer_code_sha256": sha256_file(Path(__file__)),
        "model_sha256": model_sha,
        "cap": P.CAP,
        "threshold": P.THRESHOLD,
        "feature_names": P.FEATURE_NAMES,
        "split": args.split,
        "s1_sha256": mf[s1_rel]["sha256"],
        "index": str(index),
        "index_bytes": index.stat().st_size,
        "pair_index": str(pair),
        "pair_stats": pair_stats,
        "n_targets": n_targets,
        "shard_size": args.shard_size,
        "limit": args.limit,
        "ids_file_sha256": sha256_file(args.ids_file) if args.ids_file else None,
        "provenance": f"official_challenge_dataset:{prov.manifest_version}",
    }
    run_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    shard_dir = args.out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = args.out_dir / "run_config.json"
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        if old.get("run_hash") != run_hash:
            print("WARNING: config changed since previous run; stale shards will be recomputed", flush=True)
    cfg_path.write_text(json.dumps({"run_hash": run_hash, **config}, indent=2) + "\n")
    print(f"run_hash={run_hash[:16]} targets={n_targets:,} workers={args.workers}", flush=True)

    ctx_args = (str(index), str(pair), str(args.model), bounds, pair_bounds, n_targets)
    n_shards = reused = new = 0
    n_rows = 0
    shard_times = []
    t_run = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=ctx_args) as ex:
        pending = set()
        buf = []

        def submit(sid_, rows):
            nonlocal reused
            if shard_complete(shard_dir, sid_, run_hash, verify_digest=False):
                reused += 1
                return
            pending.add(ex.submit(process_shard, sid_, rows, str(shard_dir), run_hash))

        def drain(block_until: int):
            nonlocal new
            while len(pending) > block_until:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for f in done:
                    pending.discard(f)
                    m = f.result()
                    new += 1
                    shard_times.append(m["sec"])
                    elapsed = time.time() - t_run
                    print(f"  shard {m['shard_id']:05d} n={m['n_s1']} {m['sec']:.0f}s links={m['links']} "
                          f"rss={m['worker_peak_rss_mb']:.0f}MB done_new={new} elapsed={elapsed:.0f}s", flush=True)
                    if args.stop_after_shards and new >= args.stop_after_shards:
                        print("stop-after-shards reached; terminating workers before merge", flush=True)
                        for proc in list(getattr(ex, "_processes", {}).values()):
                            proc.terminate()
                        for proc in list(getattr(ex, "_processes", {}).values()):
                            proc.join(10)
                        os._exit(3)

        for row in iter_s1(s1_path, ids_filter, args.limit):
            buf.append(row)
            n_rows += 1
            if len(buf) == args.shard_size:
                submit(n_shards, buf)
                n_shards += 1
                buf = []
                drain(2 * args.workers)
        if buf:
            submit(n_shards, buf)
            n_shards += 1
        drain(0)
    run_sec = time.time() - t_run

    t_merge = time.time()
    merged = merge(shard_dir, args.out_dir / "output", s1_path, ids_filter, args.limit, n_shards, args.shard_size, run_hash)
    merge_sec = time.time() - t_merge
    if merged["n_s1"] != n_rows:
        raise RuntimeError(f"merged {merged['n_s1']} != streamed {n_rows}")
    by_country = Counter()
    by_country_links = Counter()
    by_country_empty = Counter()
    for i in range(n_shards):
        m = json.loads((shard_dir / f"shard_{i:05d}.json").read_text())
        for c, d in m["by_country"].items():
            by_country[c] += d["s1"]
            by_country_links[c] += d["links"]
            by_country_empty[c] += d["empty_pred"]
    meta = {
        "run_hash": run_hash,
        **{k: config[k] for k in ("policy_id", "model_sha256", "policy_code_sha256", "writer_code_sha256", "s1_sha256", "n_targets", "provenance")},
        "workers": args.workers,
        "n_shards": n_shards,
        "shards_reused": reused,
        "shards_new": new,
        "run_sec": run_sec,
        "merge_sec": merge_sec,
        "total_sec": time.time() - t_all,
        "unlabeled_by_country": {c: {"s1": by_country[c], "links_per_s1": by_country_links[c] / by_country[c],
                                     "empty_pred_frac": by_country_empty[c] / by_country[c]} for c in by_country},
        **merged,
    }
    meta["s1_per_sec_new"] = (n_rows / run_sec) if (new and not reused) else None
    (args.out_dir / "output" / "inference_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: meta[k] for k in ("n_s1", "links", "empty_predictions", "run_sec", "merge_sec", "s1_per_sec_new", "shards_reused", "shards_new")}, indent=2))
    print(json.dumps(meta["unlabeled_by_country"], indent=2))


if __name__ == "__main__":
    main()
