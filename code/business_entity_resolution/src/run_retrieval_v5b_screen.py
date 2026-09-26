"""Screen v5b retrieval routes (selective addrtok + rare name 4-grams) on fixed select IDs.

Uses the same 5,000 selection S1s (seed 42) and full target universe as v4all attribution.
Compares candidate-oracle macro F0.5 and incremental true-link recovery vs frozen v4all.
Does not train a matcher. Safe to run after test inference completes (IO heavy).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from build_aux_index_v5b import addr_tokens, char_ngrams, compact_alnum
from frozen_v4.policy import CAP, Retriever, load_records, norm_addr, norm_name, s3_boundaries
from evaluation.metric_ext import score_predictions
from official_core import SOURCE_H, iter_tsv

REPO = SRC.parents[2]
WD = REPO / "artifacts" / "official_index"
REP = REPO / "reports" / "official" / "official_70bc1d8a16c6" / "retrieval_exp2"


class RetrieverV5b(Retriever):
    """v4all + optional by_addrtok + by_ngram4 from a v5b aux DB."""

    def __init__(self, db, pair_db, aux_db, v5b_db, n_targets, bounds, pair_bounds, aux_bounds, v5b_bounds,
                 use_addrtok=True, use_ngram=True, addrtok_max_df=5000, ngram_max_df=2000):
        super().__init__(db, pair_db, aux_db, n_targets, bounds, pair_bounds, aux_bounds)
        self.v5b_db = v5b_db
        self.v5b_bounds = v5b_bounds
        self.use_addrtok = use_addrtok
        self.use_ngram = use_ngram
        self.addrtok_max_df = addrtok_max_df
        self.ngram_max_df = ngram_max_df
        self.adf_cache: dict[str, int] = {}
        self.gdf_cache: dict[str, int] = {}

    def addr_df(self, key: str) -> int:
        v = self.adf_cache.get(key)
        if v is None:
            row = self.v5b_db.execute("SELECT df FROM addr_token_df WHERE key=?", (key,)).fetchone()
            v = int(row[0]) if row else 0
            if len(self.adf_cache) > 200_000:
                self.adf_cache.clear()
            self.adf_cache[key] = v
        return v

    def ngram_df(self, key: str) -> int:
        v = self.gdf_cache.get(key)
        if v is None:
            row = self.v5b_db.execute("SELECT df FROM ngram4_df WHERE key=?", (key,)).fetchone()
            v = int(row[0]) if row else 0
            if len(self.gdf_cache) > 200_000:
                self.gdf_cache.clear()
            self.gdf_cache[key] = v
        return v

    def retrieve(self, name, addr, country):
        cands, scores, routes = super().retrieve(name, addr, country)
        # Re-run scoring dict from parent is capped; rebuild pool then re-cap.
        # Parent already capped — for incremental routes we must re-retrieve with extras.
        return self.retrieve_full(name, addr, country)

    def retrieve_full(self, name, addr, country):
        # Copy parent body with extra routes before final cap.
        from frozen_v4.policy import (
            MAX_NUMS,
            compact_prefix,
            nfkc_casefold,
            nums,
            sig_tokens,
            sorted_sig,
        )

        nn, aa = norm_name(name), norm_addr(addr)
        scores: dict[str, float] = defaultdict(float)
        routes: dict[str, set] = defaultdict(set)

        def add(ids, w, route):
            for eid in ids:
                scores[eid] += w
                routes[eid].add(route)

        lk = lambda t, k, n: self._split(self.db, self.bounds, t, k, n)
        plk = lambda t, k, n: self._split(self.pair_db, self.pair_bounds, t, k, n)
        alk = lambda t, k, n: self._split(self.aux_db, self.aux_bounds, t, k, n)
        vlk = lambda t, k, n: self._split(self.v5b_db, self.v5b_bounds, t, k, n)

        if nn:
            add(lk("by_name", nn, 200), 8.0, "name")
            sig = sorted_sig(nn)
            if sig:
                add(lk("by_sig", sig, 200), 6.0, "sig")
            compact = "".join(nn.split())
            if len(compact) >= 4:
                add(lk("by_prefix", compact[:4], 60), 0.8, "prefix")
            toks = sig_tokens(nn)
            ranked_toks = sorted(toks, key=lambda t: self.token_df(t) or self.n_targets)
            for tok in ranked_toks[:8]:
                df = self.token_df(tok) or 1
                idf = self.n_log - math.log(df)
                lim = min(2000, max(80, df)) if df <= 800 else 40
                add(lk("by_token", tok, lim), 1.2 * max(idf, 0.5), "token")
            if len(ranked_toks) >= 2:
                sets = []
                for tok in ranked_toks[:4]:
                    df = self.token_df(tok) or 1
                    if df > 5000:
                        continue
                    lim = min(2000, max(80, df)) if df <= 800 else 40
                    sets.append(set(lk("by_token", tok, lim)))
                if len(sets) >= 2:
                    add(sets[0].intersection(*sets[1:]), 4.0, "tok_intersect")
            top = sorted(set(toks), key=lambda t: self.token_df(t) or self.n_targets)[:6]
            for i in range(len(top)):
                for j in range(i + 1, len(top)):
                    add(plk("by_pair", "|".join(sorted((top[i], top[j]))), 200), 6.0, "pair")
            anums4 = list(dict.fromkeys(nums(aa)))[:4]
            for t in top[:4]:
                for n in anums4:
                    add(plk("by_toknum", f"{t}#{n}", 100), 5.0, "toknum")
        for num in nums(aa):
            add(lk("by_num", num, 80), 2.2, "num")
        anums = list(dict.fromkeys(nums(aa)))[:MAX_NUMS]
        for n in anums:
            for t in addr_tokens(aa):
                add(alk("by_numtok", f"{n}#{t}", 50), 4.0, "numtok")
        srt = sorted(anums)
        for i in range(len(srt)):
            for j in range(i + 1, len(srt)):
                add(alk("by_numnum", f"{srt[i]}#{srt[j]}", 50), 4.0, "numnum")
        if nn:
            cp = compact_prefix(nn)
            if cp:
                add(alk("by_cpre8", cp, 100), 3.0, "cpre8")

        # --- v5b routes ---
        if self.use_addrtok:
            for t in addr_tokens(aa):
                df = self.addr_df(t)
                if not df or df > self.addrtok_max_df:
                    continue
                idf = self.n_log - math.log(df)
                lim = min(400, max(40, df)) if df <= 800 else min(120, max(40, df // 20))
                add(vlk("by_addrtok", t, lim), 2.5 * max(idf, 0.5), "addrtok")
        if self.use_ngram and nn:
            grams = char_ngrams(compact_alnum(nn), 4)
            ranked = sorted(grams, key=lambda g: self.ngram_df(g) or self.n_targets)[:8]
            for g in ranked:
                df = self.ngram_df(g)
                if not df or df > self.ngram_max_df:
                    continue
                idf = self.n_log - math.log(df)
                lim = min(200, max(40, df)) if df <= 400 else 40
                add(vlk("by_ngram4", g, lim), 1.8 * max(idf, 0.5), "ngram4")

        if not scores:
            return [], {}, {}
        ids = list(scores.keys())
        soft = {}
        for i in range(0, len(ids), 500):
            chunk = ids[i : i + 500]
            q = ",".join("?" * len(chunk))
            for eid, c in self.db.execute(f"SELECT id, country FROM targets WHERE id IN ({q})", chunk):
                soft[eid] = 0.55 if (country and c and nfkc_casefold(country) != nfkc_casefold(c)) else 1.0
        final = {e: s * soft.get(e, 1.0) for e, s in scores.items()}
        ranked = sorted(final.items(), key=lambda kv: (-kv[1], kv[0]))
        cands = [e for e, _ in ranked[:CAP]]
        return cands, {e: final[e] for e in cands}, {e: routes[e] for e in cands}


def load_eval(n_eval: int, seed: int = 42):
    fold = sqlite3.connect(f"file:{WD / 'official_70bc1d8a16c6_folds.sqlite'}?mode=ro", uri=True)
    rows = [r[0] for r in fold.execute("SELECT id FROM fold WHERE fold='select' ORDER BY id")]
    random.Random(seed).shuffle(rows)
    ids = rows[:n_eval]
    need = set(ids)
    s1 = {}
    root = REPO / "student_resource" / "dataset" / "train"
    for row in iter_tsv(root / "train_source1.tsv", SOURCE_H):
        if row["entity_id"] in need:
            s1[row["entity_id"]] = row
    gt = {i: set() for i in ids}
    for row in iter_tsv(root / "train_ground_truth.tsv", ["source1_entity_id", "matched_entity_ids"]):
        sid = row["source1_entity_id"]
        if sid in need and row["matched_entity_ids"]:
            gt[sid] = set(row["matched_entity_ids"].split(","))
    return ids, s1, gt


def eval_retriever(retriever, ids, s1, gt, label: str):
    t0 = time.time()
    cands = {}
    route_hits = defaultdict(int)
    sole = defaultdict(int)
    for i, sid in enumerate(ids):
        a = s1[sid]
        c, sc, rt = retriever.retrieve_full(a["business_name"], a["business_address"], a.get("country", ""))
        cands[sid] = set(c)
        truth = gt[sid]
        for t in truth & cands[sid]:
            rs = rt.get(t, set())
            for r in rs:
                route_hits[r] += 1
            if len(rs) == 1:
                sole[next(iter(rs))] += 1
        if (i + 1) % 500 == 0:
            print(f"  [{label}] {i+1}/{len(ids)} {time.time()-t0:.0f}s", flush=True)
    pred_oracle = {s: gt[s] & cands[s] for s in ids}
    m = score_predictions(gt, pred_oracle)
    true_total = sum(len(gt[s]) for s in ids)
    true_hit = sum(len(pred_oracle[s]) for s in ids)
    cand_sizes = [len(cands[s]) for s in ids]
    return {
        "label": label,
        "n_s1": len(ids),
        "micro_recall": true_hit / true_total if true_total else None,
        "oracle_macro_F0.5": m["macro_F0.5"],
        "true_total": true_total,
        "true_retrieved": true_hit,
        "cand_mean": float(np.mean(cand_sizes)),
        "cand_p95": float(np.percentile(cand_sizes, 95)),
        "zero_candidate_frac": float(np.mean([x == 0 for x in cand_sizes])),
        "route_hits_on_retrieved_truth": dict(route_hits),
        "route_sole_source_of_retrieved_truth": dict(sole),
        "runtime_sec": time.time() - t0,
        "cands": cands,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-eval", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--v5b-index", type=Path, default=WD / "train_aux_index_v5b.sqlite")
    p.add_argument("--variants", default="baseline,addrtok,ngram,both")
    args = p.parse_args()
    REP.mkdir(parents=True, exist_ok=True)
    if not args.v5b_index.exists():
        raise SystemExit(f"missing {args.v5b_index}; build with build_aux_index_v5b.py first")

    ids, s1, gt = load_eval(args.n_eval, args.seed)
    print(f"eval_s1={len(ids)} truth_links={sum(len(gt[s]) for s in ids)}", flush=True)

    db = sqlite3.connect(f"file:{WD / 'targets_index_v2.sqlite'}?mode=ro", uri=True)
    pair = sqlite3.connect(f"file:{WD / 'train_pair_index_v3.sqlite'}?mode=ro", uri=True)
    aux = sqlite3.connect(f"file:{WD / 'train_aux_index_v4.sqlite'}?mode=ro", uri=True)
    v5b = sqlite3.connect(f"file:{args.v5b_index}?mode=ro", uri=True)
    n_targets = db.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
    bounds = s3_boundaries(db, ("by_name", "by_sig", "by_prefix", "by_token", "by_num"))
    pair_bounds = s3_boundaries(pair, ("by_pair", "by_toknum"))
    aux_bounds = s3_boundaries(aux, ("by_numtok", "by_numnum", "by_cpre8"))
    v5b_bounds = s3_boundaries(v5b, ("by_addrtok", "by_ngram4"))

    variants = {
        "baseline": (False, False),
        "addrtok": (True, False),
        "ngram": (False, True),
        "both": (True, True),
    }
    wanted = [v.strip() for v in args.variants.split(",") if v.strip()]
    results = {}
    base_cands = None
    for name in wanted:
        use_a, use_g = variants[name]
        ret = RetrieverV5b(
            db, pair, aux, v5b, n_targets, bounds, pair_bounds, aux_bounds, v5b_bounds,
            use_addrtok=use_a, use_ngram=use_g,
        )
        # For true baseline, use frozen Retriever to guarantee parity
        if name == "baseline":
            base_ret = Retriever(db, pair, aux, n_targets, bounds, pair_bounds, aux_bounds)
            t0 = time.time()
            cands = {}
            for i, sid in enumerate(ids):
                a = s1[sid]
                c, _, _ = base_ret.retrieve(a["business_name"], a["business_address"], a.get("country", ""))
                cands[sid] = set(c)
                if (i + 1) % 500 == 0:
                    print(f"  [baseline] {i+1}/{len(ids)} {time.time()-t0:.0f}s", flush=True)
            pred = {s: gt[s] & cands[s] for s in ids}
            m = score_predictions(gt, pred)
            true_total = sum(len(gt[s]) for s in ids)
            true_hit = sum(len(pred[s]) for s in ids)
            sizes = [len(cands[s]) for s in ids]
            out = {
                "label": "baseline",
                "n_s1": len(ids),
                "micro_recall": true_hit / true_total,
                "oracle_macro_F0.5": m["macro_F0.5"],
                "true_total": true_total,
                "true_retrieved": true_hit,
                "cand_mean": float(np.mean(sizes)),
                "cand_p95": float(np.percentile(sizes, 95)),
                "runtime_sec": time.time() - t0,
                "cands": cands,
            }
        else:
            out = eval_retriever(ret, ids, s1, gt, name)
        results[name] = {k: v for k, v in out.items() if k != "cands"}
        if name == "baseline":
            base_cands = out["cands"]
        else:
            # incremental links vs baseline
            if base_cands is None:
                raise SystemExit("baseline must be first in --variants")
            new_links = 0
            for s in ids:
                new_links += len((gt[s] & out["cands"][s]) - (gt[s] & base_cands[s]))
            results[name]["incremental_true_links_vs_baseline"] = new_links
            results[name]["oracle_delta_vs_baseline"] = (
                results[name]["oracle_macro_F0.5"] - results["baseline"]["oracle_macro_F0.5"]
            )
        print(json.dumps(results[name], indent=2), flush=True)

    report = {
        "hypothesis": "selective rare address tokens (+ optional rare name 4-grams) recover no-shared-key misses",
        "fold": f"select_bounded_n{args.n_eval}",
        "seed": args.seed,
        "n_target_universe": n_targets,
        "v5b_index": str(args.v5b_index),
        "variants": results,
        "baseline_oracle_reference_v4all": 0.9418279818962985,
    }
    outp = REP / f"v5b_screen_n{args.n_eval}.json"
    outp.write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", outp)


if __name__ == "__main__":
    main()
