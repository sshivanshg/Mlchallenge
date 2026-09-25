"""Instrumented blocking diagnostics on the frozen official selection sample.

Reproduces generate_candidates_v2 exactly (asserted), records route membership,
pre-cap pool and ranks, sweeps final caps, and attributes baseline misses.
Variants (one change at a time):
  v2          current production retrieval
  v3a         v2 + explicit S2/S3 allocation inside every route-level LIMIT
  v3b         v3a + token-pair and name-token x address-number routes
              (requires --pair-index built by build_pair_index.py)
Gold labels are used only after candidates are fixed, for scoring/attribution.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import resource
import sqlite3
import subprocess
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from data.provenance import load_manifest, require_official_dataset
from evaluation.metric_ext import score_predictions
from official_core import (
    SOURCE_H,
    generate_candidates_v2,
    iter_tsv,
    nfkc_casefold,
    norm_addr,
    norm_name,
    nums,
    sig_tokens,
    sorted_sig,
)
from run_official_bounded import GT_H

REPO = SRC.parents[2]
ROUTE_TABLES = ("by_name", "by_sig", "by_prefix", "by_token", "by_num")


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def s3_boundaries(db: sqlite3.Connection, tables) -> dict[str, int]:
    """Last rowid holding an S2 id; rows were inserted S2 file then S3 file."""
    out = {}
    for t in tables:
        hi = db.execute(f"SELECT MAX(rowid) FROM {t}").fetchone()[0]
        lo = 1
        first_id = db.execute(f"SELECT id FROM {t} WHERE rowid=?", (lo,)).fetchone()[0]
        last_id = db.execute(f"SELECT id FROM {t} WHERE rowid=?", (hi,)).fetchone()[0]
        if not first_id.startswith("S2-") or not last_id.startswith("S3-"):
            raise RuntimeError(f"{t}: unexpected source order {first_id}..{last_id}")
        # invariant: rowid lo is S2, rowid hi is S3
        while hi - lo > 1:
            mid = (lo + hi) // 2
            mid_id = db.execute(f"SELECT id FROM {t} WHERE rowid=?", (mid,)).fetchone()[0]
            if mid_id.startswith("S2-"):
                lo = mid
            else:
                hi = mid
        out[t] = lo
    return out


class Retriever:
    def __init__(self, db, n_targets, variant, bounds=None, pair_db=None, pair_bounds=None):
        self.db = db
        self.n_targets = n_targets
        self.variant = variant
        self.bounds = bounds or {}
        self.pair_db = pair_db
        self.pair_bounds = pair_bounds or {}
        self.df_cache: dict[str, int] = {}
        self.n_log = math.log(max(n_targets, 2))

    def token_df(self, key):
        if key not in self.df_cache:
            row = self.db.execute("SELECT df FROM token_df WHERE key=?", (key,)).fetchone()
            self.df_cache[key] = int(row[0]) if row else 0
        return self.df_cache[key]

    def _plain(self, db, table, key, limit):
        return [r[0] for r in db.execute(f"SELECT id FROM {table} WHERE key=? LIMIT ?", (key, limit))]

    def _split(self, db, bounds, table, key, limit):
        b = bounds[table]
        s2 = [r[0] for r in db.execute(f"SELECT id FROM {table} WHERE key=? AND rowid<=? LIMIT ?", (key, b, limit))]
        s3 = [r[0] for r in db.execute(f"SELECT id FROM {table} WHERE key=? AND rowid>? LIMIT ?", (key, b, limit))]
        if len(s2) + len(s3) <= limit:
            return s2 + s3
        n3 = min(len(s3), max(limit // 2, limit - len(s2)))
        return s2[: limit - n3] + s3[:n3]

    def lookup(self, table, key, limit):
        if self.variant == "v2":
            return self._plain(self.db, table, key, limit)
        return self._split(self.db, self.bounds, table, key, limit)

    def pair_lookup(self, table, key, limit):
        return self._split(self.pair_db, self.pair_bounds, table, key, limit)

    def retrieve(self, name, addr, country):
        """Return (ranked_ids_all, scores, routes, queried_keys)."""
        nn, aa = norm_name(name), norm_addr(addr)
        scores: dict[str, float] = defaultdict(float)
        routes: dict[str, set] = defaultdict(set)
        queried: dict[str, set] = defaultdict(set)

        def add(ids, w, route):
            for eid in ids:
                scores[eid] += w
                routes[eid].add(route)

        if nn:
            queried["name"].add(nn)
            add(self.lookup("by_name", nn, 200), 8.0, "name")
            sig = sorted_sig(nn)
            if sig:
                queried["sig"].add(sig)
                add(self.lookup("by_sig", sig, 200), 6.0, "sig")
            compact = "".join(nn.split())
            if len(compact) >= 4:
                queried["prefix"].add(compact[:4])
                add(self.lookup("by_prefix", compact[:4], 60), 0.8, "prefix")
            toks = sig_tokens(nn)
            ranked_toks = sorted(toks, key=lambda t: self.token_df(t) or self.n_targets)
            for tok in ranked_toks[:8]:
                df = self.token_df(tok) or 1
                idf = self.n_log - math.log(df)
                lim = min(2000, max(80, df)) if df <= 800 else 40
                queried["token"].add(tok)
                add(self.lookup("by_token", tok, lim), 1.2 * max(idf, 0.5), "token")
            if len(ranked_toks) >= 2:
                sets = []
                for tok in ranked_toks[:4]:
                    df = self.token_df(tok) or 1
                    if df > 5000:
                        continue
                    lim = min(2000, max(80, df)) if df <= 800 else 40
                    sets.append(set(self.lookup("by_token", tok, lim)))
                if len(sets) >= 2:
                    add(sets[0].intersection(*sets[1:]), 4.0, "tok_intersect")

            if self.variant == "v3b":
                uniq = []
                for t in sorted(set(toks), key=lambda t: self.token_df(t) or self.n_targets):
                    uniq.append(t)
                top = uniq[:6]
                for i in range(len(top)):
                    for j in range(i + 1, len(top)):
                        key = "|".join(sorted((top[i], top[j])))
                        queried["pair"].add(key)
                        add(self.pair_lookup("by_pair", key, 200), 6.0, "pair")
                anums = list(dict.fromkeys(nums(aa)))[:4]
                for t in top[:4]:
                    for n in anums:
                        key = f"{t}#{n}"
                        queried["toknum"].add(key)
                        add(self.pair_lookup("by_toknum", key, 100), 5.0, "toknum")

        for num in nums(aa):
            queried["num"].add(num)
            add(self.lookup("by_num", num, 80), 2.2, "num")

        if not scores:
            return [], scores, routes, queried
        ids = list(scores.keys())
        soft = {}
        for i in range(0, len(ids), 500):
            chunk = ids[i : i + 500]
            q = ",".join("?" * len(chunk))
            for eid, c in self.db.execute(f"SELECT id, country FROM targets WHERE id IN ({q})", chunk):
                soft[eid] = 0.55 if (country and c and nfkc_casefold(country) != nfkc_casefold(c)) else 1.0
        final = {e: s * soft.get(e, 1.0) for e, s in scores.items()}
        ranked = sorted(final.items(), key=lambda kv: (-kv[1], kv[0]))
        return [e for e, _ in ranked], final, routes, queried


def pct(vals, p):
    if not vals:
        return 0.0
    s = sorted(vals)
    return float(s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))])


def cap_metrics(cands: dict[str, list[str]], gt: dict[str, set[str]]) -> dict:
    hit = tot = s2h = s2t = s3h = s3t = full = nonsing = 0
    macro_r = []
    counts, n2, n3 = [], [], []
    for sid, t in gt.items():
        c = cands.get(sid, [])
        cs = set(c)
        counts.append(len(c))
        n2.append(sum(x.startswith("S2-") for x in c))
        n3.append(len(c) - n2[-1])
        if not t:
            continue
        nonsing += 1
        inter = t & cs
        hit += len(inter)
        tot += len(t)
        macro_r.append(len(inter) / len(t))
        full += int(t <= cs)
        for x in t:
            if x.startswith("S2-"):
                s2t += 1
                s2h += int(x in cs)
            else:
                s3t += 1
                s3h += int(x in cs)
    oracle = score_predictions(gt, {s: gt[s] & set(cands.get(s, [])) for s in gt})
    return {
        "micro_recall": hit / tot if tot else None,
        "macro_recall_nonsingleton": float(np.mean(macro_r)) if macro_r else None,
        "full_set_coverage": full / nonsing if nonsing else None,
        "oracle_macro_F0.5": oracle["macro_F0.5"],
        "s2_recall": s2h / s2t if s2t else None,
        "s3_recall": s3h / s3t if s3t else None,
        "cand_mean": float(np.mean(counts)),
        "cand_p95": pct(counts, 95),
        "cand_p99": pct(counts, 99),
        "cand_max": float(max(counts) if counts else 0),
        "alloc_s2_mean": float(np.mean(n2)),
        "alloc_s3_mean": float(np.mean(n3)),
        "zero_candidate_frac": sum(c == 0 for c in counts) / max(len(counts), 1),
        "true_total": tot,
        "true_retrieved": hit,
    }


def target_keys(rec):
    nn, aa = rec["name_norm"] or "", rec["addr_norm"] or ""
    compact = "".join(nn.split())
    return {
        "name": {nn} if nn else set(),
        "sig": {sorted_sig(nn)} - {""},
        "prefix": {compact[:4]} if len(compact) >= 4 else set(),
        "token": set(sig_tokens(nn)),
        "num": set(nums(aa)),
    }


def s1_all_keys(name, addr):
    nn, aa = norm_name(name), norm_addr(addr)
    compact = "".join(nn.split())
    return {
        "name": {nn} if nn else set(),
        "sig": {sorted_sig(nn)} - {""},
        "prefix": {compact[:4]} if len(compact) >= 4 else set(),
        "token": set(sig_tokens(nn)),
        "num": set(nums(aa)),
    }


CATS = [
    "1_target_absent_from_index",
    "2_no_shared_retrieval_key",
    "3_shared_key_not_queried",
    "4_discarded_by_route_limit",
    "5_in_pool_below_final_cap",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=REPO / "student_resource" / "dataset")
    p.add_argument("--work-dir", type=Path, default=REPO / "artifacts" / "official_index")
    p.add_argument("--index", type=Path, default=None)
    p.add_argument("--pair-index", type=Path, default=None)
    p.add_argument("--variant", choices=["v2", "v3a", "v3b"], default="v2")
    p.add_argument("--caps", default="120,300,500,1000")
    p.add_argument("--eval-s1", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--attribute-cap", type=int, default=120)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()
    caps = [int(x) for x in args.caps.split(",")]

    t_all = time.time()
    prov = require_official_dataset(args.dataset_root, require=("train",), check_hashes=True)
    manifest = load_manifest()
    gt_hash = manifest["files"]["train/train_ground_truth.tsv"]["sha256"]
    split_version = f"official_{gt_hash[:12]}"
    out_dir = args.out_dir or (REPO / "reports" / "official" / split_version / "retrieval_exp1")
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_db = sqlite3.connect(args.work_dir / f"{split_version}_folds.sqlite")
    rows = list(fold_db.execute("SELECT id FROM fold WHERE fold='select' ORDER BY id"))
    random.Random(args.seed).shuffle(rows)
    eval_ids = [r[0] for r in rows[: args.eval_s1]]
    fold_db.close()
    needed = set(eval_ids)

    s1_map = {}
    for row in iter_tsv(args.dataset_root / "train" / "train_source1.tsv", SOURCE_H):
        if row["entity_id"] in needed:
            s1_map[row["entity_id"]] = row

    index_path = args.index or (args.work_dir / "targets_index_v2.sqlite")
    db = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    n_targets = db.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
    bounds = s3_boundaries(db, ROUTE_TABLES) if args.variant != "v2" else {}
    pair_db = pair_bounds = None
    if args.variant == "v3b":
        if not args.pair_index or not args.pair_index.exists():
            raise SystemExit("--pair-index required for v3b")
        pair_db = sqlite3.connect(f"file:{args.pair_index}?mode=ro", uri=True)
        pair_bounds = s3_boundaries(pair_db, ("by_pair", "by_toknum"))
    ret = Retriever(db, n_targets, args.variant, bounds, pair_db, pair_bounds)

    # ---- candidate generation (labels not loaded yet) ----
    t0 = time.time()
    ranked_all, routes_all, queried_all, pool_sizes = {}, {}, {}, []
    parity_checked = parity_ok = 0
    for i, sid in enumerate(eval_ids):
        r = s1_map[sid]
        ranked, _, routes, queried = ret.retrieve(r["business_name"], r["business_address"], r["country"])
        ranked_all[sid] = ranked
        routes_all[sid] = routes
        queried_all[sid] = queried
        pool_sizes.append(len(ranked))
        if args.variant == "v2" and i < 200:
            ref = generate_candidates_v2(
                db, r["business_name"], r["business_address"], r["country"],
                n_targets=n_targets, df_cache={}, max_candidates=120,
            )
            parity_checked += 1
            parity_ok += int(ref == ranked[:120])
        if (i + 1) % 1000 == 0:
            print(f"  [{args.variant}] retrieved {i+1}/{len(eval_ids)} rss={rss_mb():.0f}MB", flush=True)
    gen_sec = time.time() - t0

    # ---- labels (retrospective only) ----
    gt = {i: set() for i in eval_ids}
    for row in iter_tsv(args.dataset_root / "train" / "train_ground_truth.tsv", GT_H):
        sid = row["source1_entity_id"]
        if sid in needed:
            gt[sid] = {x for x in row["matched_entity_ids"].split(",") if x} if row["matched_entity_ids"] else set()

    sweep = {}
    for cap in caps + ["pool"]:
        cands = {s: (ranked_all[s] if cap == "pool" else ranked_all[s][:cap]) for s in eval_ids}
        sweep[str(cap)] = cap_metrics(cands, gt)
        print(f"  cap={cap}: " + json.dumps({k: sweep[str(cap)][k] for k in ("micro_recall", "oracle_macro_F0.5", "s2_recall", "s3_recall", "cand_mean")}), flush=True)

    # route contribution among retrieved true matches at attribute cap
    route_true = Counter()
    route_only_true = Counter()
    for sid in eval_ids:
        top = set(ranked_all[sid][: args.attribute_cap])
        for t in gt[sid] & top:
            rs = routes_all[sid].get(t, set())
            for rr in rs:
                route_true[rr] += 1
            if len(rs) == 1:
                route_only_true[next(iter(rs))] += 1

    # ---- miss attribution at attribute cap ----
    miss_pairs = []
    for sid in eval_ids:
        top = set(ranked_all[sid][: args.attribute_cap])
        for t in gt[sid] - top:
            miss_pairs.append((sid, t))
    miss_ids = sorted({t for _, t in miss_pairs})
    trecs = {}
    for i in range(0, len(miss_ids), 500):
        chunk = miss_ids[i : i + 500]
        q = ",".join("?" * len(chunk))
        for eid, nn, aa in db.execute(f"SELECT id, name_norm, addr_norm FROM targets WHERE id IN ({q})", chunk):
            trecs[eid] = {"name_norm": nn, "addr_norm": aa}

    cat_counts = Counter()
    cat_src = defaultdict(Counter)
    cat_shared_route = defaultdict(Counter)
    recovered_by_cat = defaultdict(lambda: defaultdict(set))
    examples = defaultdict(list)
    pair_cat: dict[tuple[str, str], str] = {}
    for sid, t in miss_pairs:
        r = s1_map[sid]
        pool = routes_all[sid]
        if t not in trecs:
            cat = CATS[0]
        elif t in pool:
            cat = CATS[4]
        else:
            tk = target_keys(trecs[t])
            sk = s1_all_keys(r["business_name"], r["business_address"])
            shared = {k: tk[k] & sk[k] for k in tk if tk[k] & sk[k]}
            if not shared:
                cat = CATS[1]
            else:
                q = queried_all[sid]
                queried_shared = {k: v & q.get(k, set()) for k, v in shared.items() if v & q.get(k, set())}
                if queried_shared:
                    cat = CATS[3]
                    for k in queried_shared:
                        cat_shared_route[cat][k] += 1
                else:
                    cat = CATS[2]
                    for k in shared:
                        cat_shared_route[cat][k] += 1
        pair_cat[(sid, t)] = cat
        cat_counts[cat] += 1
        cat_src[cat][t[:2]] += 1
        recovered_by_cat[cat][sid].add(t)
        if len(examples[cat]) < 6:
            examples[cat].append({"s1": sid, "target": t, "s1_name": r["business_name"],
                                  "target_name_norm": trecs.get(t, {}).get("name_norm")})

    base_cands = {s: set(ranked_all[s][: args.attribute_cap]) for s in eval_ids}
    base_oracle = score_predictions(gt, {s: gt[s] & base_cands[s] for s in eval_ids})["macro_F0.5"]
    impact = {}
    for cat in CATS:
        add_map = recovered_by_cat.get(cat, {})
        pred = {s: (gt[s] & base_cands[s]) | add_map.get(s, set()) for s in eval_ids}
        impact[cat] = score_predictions(gt, pred)["macro_F0.5"] - base_oracle

    # ---- slice breakdown of truth links at the attribution cap ----
    truth_ids = sorted({t for s in eval_ids for t in gt[s]})
    raw_name = {}
    for i in range(0, len(truth_ids), 500):
        chunk = truth_ids[i : i + 500]
        q = ",".join("?" * len(chunk))
        for eid, name in db.execute(f"SELECT id, name FROM targets WHERE id IN ({q})", chunk):
            raw_name[eid] = name or ""

    def script(name):
        letters = [ch for ch in name if ch.isalpha()]
        if not letters:
            return "no_letters"
        return "latin" if all("LATIN" in unicodedata.name(ch, "") for ch in letters) else "non_latin"

    ambiguity = {}
    for sid in eval_ids:
        nn = norm_name(s1_map[sid]["business_name"])
        n_same = db.execute("SELECT COUNT(*) FROM (SELECT 1 FROM by_name WHERE key=? LIMIT 51)", (nn,)).fetchone()[0] if nn else 0
        ambiguity[sid] = "0" if n_same == 0 else "1" if n_same == 1 else "2-5" if n_same <= 5 else "6-50" if n_same <= 50 else ">50"

    def mult(k):
        return "1" if k == 1 else "2-3" if k <= 3 else "4-5" if k <= 5 else "6+"

    slices = defaultdict(lambda: defaultdict(lambda: Counter()))
    for sid in eval_ids:
        top = set(ranked_all[sid][: args.attribute_cap])
        for t in gt[sid]:
            keys = {
                "source": t[:2],
                "country": s1_map[sid]["country"],
                "target_script": script(raw_name.get(t, "")),
                "s1_name_ambiguity(exact-name postings)": ambiguity[sid],
                "s1_match_multiplicity": mult(len(gt[sid])),
            }
            for dim, b in keys.items():
                c = slices[dim][b]
                c["truth_links"] += 1
                c["retrieved"] += int(t in top)
                if t not in top:
                    c[pair_cat[(sid, t)]] += 1
    slice_report = {
        dim: {b: {**dict(c), "recall": c["retrieved"] / c["truth_links"]} for b, c in sorted(bs.items())}
        for dim, bs in slices.items()
    }

    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    report = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provenance": f"official_challenge_dataset:{prov.manifest_version}",
        "split_version": split_version,
        "fold": "select_bounded_n5000 (seed 42, same IDs as Stage 4)",
        "variant": args.variant,
        "code_revision": rev,
        "index": str(index_path),
        "pair_index": str(args.pair_index) if args.pair_index else None,
        "n_eval_s1": len(eval_ids),
        "n_target_universe": n_targets,
        "route_budgets": {"name": 200, "sig": 200, "prefix": 60, "token_rare(df<=800)": "min(2000,max(80,df))",
                          "token_common": 40, "num": 80, "pair": 200 if args.variant == "v3b" else None,
                          "toknum": 100 if args.variant == "v3b" else None,
                          "source_split": args.variant != "v2"},
        "s3_rowid_boundaries": bounds,
        "parity_vs_generate_candidates_v2": {"checked": parity_checked, "identical_top120": parity_ok} if args.variant == "v2" else None,
        "pool_size": {"mean": float(np.mean(pool_sizes)), "p50": pct(pool_sizes, 50), "p95": pct(pool_sizes, 95),
                      "p99": pct(pool_sizes, 99), "max": float(max(pool_sizes))},
        "cap_sweep": sweep,
        "route_hits_on_retrieved_truth": dict(route_true),
        "route_sole_source_of_retrieved_truth": dict(route_only_true),
        "attribution_cap": args.attribute_cap,
        "baseline_oracle_at_attribution_cap": base_oracle,
        "miss_attribution": {
            "denominator_missed_true_links": len(miss_pairs),
            "sampling": "none (all misses)",
            "counts": {c: cat_counts.get(c, 0) for c in CATS},
            "by_source": {c: dict(cat_src.get(c, {})) for c in CATS},
            "shared_key_types": {c: dict(v) for c, v in cat_shared_route.items()},
            "oracle_gain_if_category_recovered": impact,
            "examples": dict(examples),
        },
        "slices_at_attribution_cap": slice_report,
        "runtime_sec": {"retrieval": gen_sec, "total": time.time() - t_all},
        "retrieval_ms_per_s1": 1000 * gen_sec / max(len(eval_ids), 1),
        "peak_rss_mb": rss_mb(),
    }
    suffix = "" if args.attribute_cap == 120 else f"_attr{args.attribute_cap}"
    out = out_dir / f"blocking_diag_{args.variant}{suffix}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("variant", "parity_vs_generate_candidates_v2", "pool_size", "runtime_sec", "peak_rss_mb")}, indent=2))
    print(json.dumps(report["miss_attribution"]["counts"], indent=2))
    print(json.dumps(report["miss_attribution"]["oracle_gain_if_category_recovered"], indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
