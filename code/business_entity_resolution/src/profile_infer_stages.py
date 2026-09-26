"""Bounded stage-timing profile for frozen v5 inference (official train indexes).

Measures wall time separately for:
  data_load, index_open, candidate_gen, record_fetch, pair_features,
  model_predict, tsv_write. Reports peak RSS and extrapolates full-test cost.

Does not write a submission. Safe after packaging; keep sequential with other
heavy index readers.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import pickle
import random
import resource
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from data.provenance import require_official_dataset
from frozen_v5 import policy as P
from official_core import SOURCE_H, iter_tsv
from run_infer_v5 import MODEL_KEY, MODEL_SHA256, N_FEATURES, SELECTED, THRESHOLD, load_booster

REPO = SRC.parents[2]
WD = REPO / "artifacts" / "official_index"


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-s1", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--fold", default="select", choices=["select", "fit", "assess"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    t_all = time.time()
    stages: dict[str, float] = {}

    ds = REPO / "student_resource" / "dataset"
    require_official_dataset(ds, require=("train",), check_hashes=True)

    t0 = time.time()
    fold = sqlite3.connect(f"file:{WD / 'official_70bc1d8a16c6_folds.sqlite'}?mode=ro", uri=True)
    rows = [r[0] for r in fold.execute("SELECT id FROM fold WHERE fold=? ORDER BY id", (args.fold,))]
    random.Random(args.seed).shuffle(rows)
    ids = rows[: args.n_s1]
    need = set(ids)
    s1 = {}
    for row in iter_tsv(ds / "train" / "train_source1.tsv", SOURCE_H):
        if row["entity_id"] in need:
            s1[row["entity_id"]] = row
            if len(s1) == len(need):
                break
    ordered = [s1[i] for i in ids]
    stages["data_load_normalize"] = time.time() - t0

    t0 = time.time()
    index = WD / "targets_index_v2.sqlite"
    pair = WD / "train_pair_index_v3.sqlite"
    aux = WD / "train_aux_index_v4.sqlite"
    model_path = REPO / SELECTED["model_path"]
    assert pickle  # noqa: keep import used for type checkers
    db = sqlite3.connect(f"file:{index}?mode=ro", uri=True, check_same_thread=False)
    db.execute("PRAGMA cache_size=-196608")
    pdb = sqlite3.connect(f"file:{pair}?mode=ro", uri=True, check_same_thread=False)
    pdb.execute("PRAGMA cache_size=-65536")
    adb = sqlite3.connect(f"file:{aux}?mode=ro", uri=True, check_same_thread=False)
    adb.execute("PRAGMA cache_size=-65536")
    n_targets = db.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
    bounds = P.s3_boundaries(db, P.ROUTE_TABLES)
    pair_bounds = P.s3_boundaries(pdb, P.PAIR_TABLES)
    aux_bounds = P.s3_boundaries(adb, P.AUX_TABLES)
    ret = P.Retriever(db, pdb, adb, n_targets, bounds, pair_bounds, aux_bounds)
    booster = load_booster(model_path)
    stages["index_open_model_load"] = time.time() - t0

    t_ret = t_fetch = t_feat = t_pred = 0.0
    n_cands = 0
    n_pairs = 0
    links = 0
    for i in range(0, len(ordered), args.batch):
        chunk = ordered[i : i + args.batch]
        t0 = time.time()
        got = [ret.retrieve(r["business_name"], r["business_address"], r["country"]) for r in chunk]
        t_ret += time.time() - t0
        need_ids = {t for c, _, _ in got for t in c}
        t0 = time.time()
        recs = P.load_records(db, need_ids) if need_ids else {}
        t_fetch += time.time() - t0
        mats, sizes = [], []
        t0 = time.time()
        for r, (c, sc, rt) in zip(chunk, got):
            n_cands += len(c)
            if c:
                a = P.s1_view(r)
                mats.append(P.features(a, c, sc, rt, recs))
            sizes.append(len(c))
        t_feat += time.time() - t0
        t0 = time.time()
        probs = booster.predict(np.vstack(mats), num_threads=1) if mats else np.zeros(0)
        t_pred += time.time() - t0
        k = 0
        for n in sizes:
            pr = probs[k : k + n]
            k += n
            links += int((pr >= THRESHOLD).sum()) if n else 0
            n_pairs += n

    stages["candidate_gen"] = t_ret
    stages["target_record_fetch"] = t_fetch
    stages["pair_features"] = t_feat
    stages["model_predict"] = t_pred

    t0 = time.time()
    # Simulate TSV write cost for this bounded set
    out_tmp = REPO / "artifacts" / "official_index" / f"_profile_write_{args.n_s1}.tsv"
    with out_tmp.open("w", encoding="utf-8", newline="\n") as f:
        for r in ordered:
            f.write(f"{r['entity_id']}\t\n")
    out_tmp.unlink(missing_ok=True)
    stages["tsv_write_sim"] = time.time() - t0

    wall = time.time() - t_all
    rss = peak_rss_mb()
    s1_per_sec = len(ordered) / max(wall, 1e-9)
    # Full test extrapolation from measured per-S1 stage costs (single worker).
    test_s1 = 1_732_544
    per = {k: v / len(ordered) for k, v in stages.items()}
    # 4-worker inference observed ~2.1x over single-worker ideal due to IO contention;
    # report both ideal and contended estimates.
    ideal_workers = 4
    ideal_sec = test_s1 * (per["candidate_gen"] + per["target_record_fetch"] + per["pair_features"] + per["model_predict"]) / ideal_workers
    contended_sec = ideal_sec * 2.1  # calibrated from v5 full run (19436s vs ~ideal)

    report = {
        "n_s1": len(ordered),
        "fold": args.fold,
        "seed": args.seed,
        "batch": args.batch,
        "n_targets": n_targets,
        "n_candidate_slots": n_cands,
        "n_pairs_scored": n_pairs,
        "links_at_threshold": links,
        "threshold": THRESHOLD,
        "model_key": MODEL_KEY,
        "model_sha256_expected": MODEL_SHA256,
        "wall_sec": wall,
        "peak_rss_mb": rss,
        "s1_per_sec": s1_per_sec,
        "stages_sec": stages,
        "stages_frac": {k: v / wall for k, v in stages.items()},
        "per_s1_ms": {k: 1000.0 * v for k, v in per.items()},
        "extrapolate_test_1p7M": {
            "workers": ideal_workers,
            "ideal_wall_sec": ideal_sec,
            "contended_wall_sec_x2_1": contended_sec,
            "uncertainty": "±25% depending on page-cache warmth and concurrent disk readers",
            "observed_v5_full_run_sec": 19436.4,
        },
        "disk_indexes_gb": {
            "targets": index.stat().st_size / 1e9,
            "pair": pair.stat().st_size / 1e9,
            "aux": aux.stat().st_size / 1e9,
        },
    }
    text = json.dumps(report, indent=2) + "\n"
    print(text, flush=True)
    out = args.out or (REPO / "reports" / "official" / "official_70bc1d8a16c6" / "runtime_profile_v5.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)


if __name__ == "__main__":
    main()
