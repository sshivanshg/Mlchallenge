"""Resumable sharded inference for frozen policy v6 (addrtok retrieval + H4 capacity).

Reads model/threshold from frozen_v6/selected.json. Requires train/test_aux_index_v5b.
Do not run full test until assess_fresh_v4 one-shot confirms gains vs v5.

  python3 run_infer_v6.py --split test --workers 4 --out-dir artifacts/submissions/v6_v1
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import pickle
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
from frozen_v6 import policy as P
from official_core import SOURCE_H, iter_tsv
from run_infer_v3b import iter_s1, merge, peak_rss_mb, sha256_file, shard_complete

REPO = SRC.parents[2]
SELECTED = json.loads((SRC / "frozen_v6" / "selected.json").read_text())
DEFAULT_MODEL = REPO / SELECTED["model_path"]
MODEL_SHA256 = SELECTED["model_sha256"]
MODEL_KEY = SELECTED["model_key"]
N_FEATURES = SELECTED["n_features"]
THRESHOLD, GATE, REL = SELECTED["threshold"], SELECTED["gate"], SELECTED["rel"]

_W: dict = {}


def load_booster(model_path: Path):
    with model_path.open("rb") as f:
        bundle = pickle.load(f)
    if bundle.get("policy") not in (P.POLICY_ID, None) and "addrtok" not in str(bundle.get("policy", "")):
        # accept capacity pickle saved by run_matcher_v6_addrtok
        if bundle.get("n_features") not in (None, N_FEATURES) and MODEL_KEY not in bundle and "model" not in bundle:
            raise RuntimeError(f"unexpected bundle keys {list(bundle)}")
    feats = bundle.get("feature_names") or (
        list(bundle.get("base_features", [])) + list(bundle.get("extra2_features", []))
    )
    if feats and feats != P.FEATURE_NAMES:
        raise RuntimeError("bundle feature order differs from frozen_v6")
    model = bundle[MODEL_KEY] if MODEL_KEY in bundle else bundle["model"]
    n_in = getattr(model, "n_features_in_", None) or getattr(model, "n_features_", N_FEATURES)
    if int(n_in) != N_FEATURES:
        raise RuntimeError(f"model feature count mismatch {n_in} != {N_FEATURES}")
    return model.booster_


def _init_worker(index_path, pair_path, aux_path, v5b_path, model_path, bounds, pair_bounds, aux_bounds, v5b_bounds, n_targets):
    conns = []
    for pth in (index_path, pair_path, aux_path, v5b_path):
        c = sqlite3.connect(f"file:{pth}?mode=ro", uri=True, check_same_thread=False)
        c.execute("PRAGMA cache_size=-196608")
        conns.append(c)
    _W["db"] = conns[0]
    _W["ret"] = P.Retriever(
        conns[0], conns[1], conns[2], conns[3], n_targets,
        bounds, pair_bounds, aux_bounds, v5b_bounds,
    )
    _W["booster"] = load_booster(Path(model_path))


def score_rows(rows, batch: int = 256):
    ret, db, booster = _W["ret"], _W["db"], _W["booster"]
    out = []
    for i in range(0, len(rows), batch):
        chunk = rows[i : i + batch]
        got = [ret.retrieve(r["business_name"], r["business_address"], r["country"]) for r in chunk]
        recs = P.load_records(db, {t for c, _, _ in got for t in c})
        mats, sizes = [], []
        for r, (c, sc, rt) in zip(chunk, got):
            if c:
                a = P.s1_view(r)
                mats.append(P.features(a, c, sc, rt, recs))
            sizes.append(len(c))
        probs = booster.predict(np.vstack(mats), num_threads=1) if mats else np.zeros(0)
        k = 0
        for r, (c, _, _), n in zip(chunk, got, sizes):
            pr = probs[k : k + n]
            k += n
            mx = float(pr.max()) if n else 0.0
            out.append((r["entity_id"], c, [t for t, p in zip(c, pr) if p >= THRESHOLD and mx >= GATE and p >= REL * mx]))
    return out


def process_shard(shard_id: int, rows: list[dict], shard_dir: str, run_hash: str) -> dict:
    t0 = time.time()
    base = Path(shard_dir) / f"shard_{shard_id:05d}"
    m_tmp, c_tmp = base.with_suffix(".match.tsv.tmp"), base.with_suffix(".cand.tsv.tmp")
    results = score_rows(rows)
    by_country, ccount, links = {}, [], 0
    with m_tmp.open("w", encoding="utf-8", newline="\n") as mf, c_tmp.open("w", encoding="utf-8", newline="\n") as cf:
        for r, (sid, c, m) in zip(rows, results):
            if sid != r["entity_id"]:
                raise RuntimeError("row order mismatch inside shard")
            mf.write(f"{sid}\t{','.join(m)}\n")
            cf.write(f"{sid}\t{','.join(c)}\n")
            d = by_country.setdefault(r["country"], Counter())
            d["s1"] += 1
            d["links"] += len(m)
            d["empty_pred"] += int(not m)
            d["cands"] += len(c)
            d["zero_cand"] += int(not c)
            links += len(m)
            ccount.append(len(c))
    m_path, c_path = base.with_suffix(".match.tsv"), base.with_suffix(".cand.tsv")
    os.replace(m_tmp, m_path)
    os.replace(c_tmp, c_path)
    manifest = {
        "shard_id": shard_id, "run_hash": run_hash, "n_s1": len(rows),
        "first_id": rows[0]["entity_id"], "last_id": rows[-1]["entity_id"], "links": links,
        "cand_mean": float(np.mean(ccount)), "cand_max": int(max(ccount)),
        "by_country": {k: dict(v) for k, v in by_country.items()},
        "match_sha256": sha256_file(m_path), "cand_sha256": sha256_file(c_path),
        "match_bytes": m_path.stat().st_size, "cand_bytes": c_path.stat().st_size,
        "sec": time.time() - t0, "worker_pid": os.getpid(), "worker_peak_rss_mb": peak_rss_mb(),
    }
    mtmp = base.with_suffix(".json.tmp")
    mtmp.write_text(json.dumps(manifest, indent=1) + "\n")
    os.replace(mtmp, base.with_suffix(".json"))
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=REPO / "student_resource" / "dataset")
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--shard-size", type=int, default=20000)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--ids-file", type=Path, default=None)
    p.add_argument("--stop-after-shards", type=int, default=0)
    args = p.parse_args()
    t_all = time.time()

    prov = require_official_dataset(args.dataset_root, require=(args.split,), check_hashes=True)
    mf = load_manifest()["files"]
    s1_rel = f"{args.split}/{args.split}_source1.tsv"
    s1_path = args.dataset_root / s1_rel
    wd = REPO / "artifacts" / "official_index"
    index = wd / ("targets_index_v2.sqlite" if args.split == "train" else "test_targets_index_v2.sqlite")
    pair = wd / f"{args.split}_pair_index_v3.sqlite"
    aux = wd / f"{args.split}_aux_index_v4.sqlite"
    v5b = wd / f"{args.split}_aux_index_v5b.sqlite"
    if not v5b.exists():
        raise SystemExit(f"missing {v5b}; build with build_aux_index_v5b.py --split {args.split}")
    pair_stats = json.loads(pair.with_suffix(".stats.json").read_text())
    aux_stats = json.loads(aux.with_suffix(".stats.json").read_text())
    v5b_stats = json.loads(v5b.with_suffix(".stats.json").read_text())
    for name, st in (("pair", pair_stats), ("aux", aux_stats), ("v5b", v5b_stats)):
        if st.get("split") != args.split:
            raise SystemExit(f"{name} index split mismatch")
    if not aux_stats.get("complete") or not v5b_stats.get("complete"):
        raise SystemExit("index lacks completion metadata")
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
    bounds = P.s3_boundaries(db, P.ROUTE_TABLES)
    db.close()
    c = sqlite3.connect(f"file:{pair}?mode=ro", uri=True)
    pair_bounds = P.s3_boundaries(c, P.PAIR_TABLES)
    c.close()
    c = sqlite3.connect(f"file:{aux}?mode=ro", uri=True)
    aux_bounds = P.s3_boundaries(c, P.AUX_TABLES)
    c.close()
    c = sqlite3.connect(f"file:{v5b}?mode=ro", uri=True)
    v5b_bounds = P.s3_boundaries(c, P.V5B_TABLES)
    c.close()

    config = {
        "policy_id": P.POLICY_ID,
        "policy_code_sha256": sha256_file(Path(P.__file__)),
        "selected": SELECTED,
        "writer_code_sha256": sha256_file(Path(__file__)),
        "model_sha256": model_sha, "model_key": MODEL_KEY,
        "cap": P.CAP, "threshold": THRESHOLD, "gate": GATE, "rel": REL,
        "feature_names": P.FEATURE_NAMES,
        "split": args.split, "s1_sha256": mf[s1_rel]["sha256"],
        "index": str(index), "index_bytes": index.stat().st_size,
        "pair_stats": pair_stats, "aux_stats": aux_stats, "v5b_stats": v5b_stats, "n_targets": n_targets,
        "shard_size": args.shard_size, "limit": args.limit,
        "ids_file_sha256": sha256_file(args.ids_file) if args.ids_file else None,
        "provenance": f"official_challenge_dataset:{prov.manifest_version}",
    }
    run_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    shard_dir = args.out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "run_config.json").write_text(json.dumps({"run_hash": run_hash, **config}, indent=2) + "\n")
    print(f"run_hash={run_hash[:16]} targets={n_targets:,} workers={args.workers}", flush=True)

    ctx = (str(index), str(pair), str(aux), str(v5b), str(args.model), bounds, pair_bounds, aux_bounds, v5b_bounds, n_targets)
    n_shards = reused = new = n_rows = 0
    t_run = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=ctx) as ex:
        pending = set()

        def drain(block_until):
            nonlocal new
            while len(pending) > block_until:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for f in done:
                    pending.discard(f)
                    m = f.result()
                    new += 1
                    print(f"  shard {m['shard_id']:05d} n={m['n_s1']} {m['sec']:.0f}s links={m['links']} "
                          f"rss={m['worker_peak_rss_mb']:.0f}MB done_new={new} elapsed={time.time()-t_run:.0f}s", flush=True)
                    if args.stop_after_shards and new >= args.stop_after_shards:
                        print("stop-after-shards reached; terminating workers before merge", flush=True)
                        for proc in list(getattr(ex, "_processes", {}).values()):
                            proc.terminate()
                        for proc in list(getattr(ex, "_processes", {}).values()):
                            proc.join(10)
                        os._exit(3)

        buf = []

        def submit(rows):
            nonlocal n_shards, reused
            if shard_complete(shard_dir, n_shards, run_hash, verify_digest=False):
                reused += 1
            else:
                pending.add(ex.submit(process_shard, n_shards, rows, str(shard_dir), run_hash))
            n_shards += 1

        for row in iter_s1(s1_path, ids_filter, args.limit):
            buf.append(row)
            n_rows += 1
            if len(buf) == args.shard_size:
                submit(buf)
                buf = []
                drain(2 * args.workers)
        if buf:
            submit(buf)
        drain(0)
    run_sec = time.time() - t_run
    t_merge = time.time()
    merged = merge(shard_dir, args.out_dir / "output", s1_path, ids_filter, args.limit, n_shards, args.shard_size, run_hash)
    if merged["n_s1"] != n_rows:
        raise RuntimeError(f"merged {merged['n_s1']} != streamed {n_rows}")
    agg = {}
    for i in range(n_shards):
        for ctry, d in json.loads((shard_dir / f"shard_{i:05d}.json").read_text())["by_country"].items():
            a = agg.setdefault(ctry, Counter())
            a.update(d)
    meta = {
        "run_hash": run_hash,
        **{k: config[k] for k in ("policy_id", "model_sha256", "policy_code_sha256", "writer_code_sha256", "s1_sha256", "n_targets", "provenance")},
        "workers": args.workers, "n_shards": n_shards, "shards_reused": reused, "shards_new": new,
        "run_sec": run_sec, "merge_sec": time.time() - t_merge, "total_sec": time.time() - t_all,
        "s1_per_sec_new": (n_rows / run_sec) if (new and not reused) else None,
        "unlabeled_by_country": {k: {"s1": v["s1"], "links_per_s1": v["links"] / v["s1"],
                                     "empty_pred_frac": v["empty_pred"] / v["s1"], "cand_mean": v["cands"] / v["s1"]}
                                 for k, v in agg.items()},
        **merged,
    }
    (args.out_dir / "output" / "inference_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: meta[k] for k in ("n_s1", "links", "empty_predictions", "run_sec", "merge_sec", "s1_per_sec_new", "shards_reused", "shards_new")}, indent=2))


if __name__ == "__main__":
    main()
