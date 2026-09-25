"""Frozen inference policy v4all_cap300_m2 (selected 2026-09-25 on the selection fold).

Retrieval = frozen v3b routes + address-number x address-token, address-number pairs,
and 8-char compact-name prefix (aux index). Matcher = LightGBM on 9 base features
plus candidate-competition / retrieval / route / missing-field features, trained on
every fit-fold candidate. Operation order mirrors the research path
(run_blocking_diag.Retriever v4all + run_matcher_v4.block_features) so float
score sums and candidate ranking are identical. Do not edit; version a new package.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from frozen_v3b.policy import (  # frozen, immutable
    FEATURE_NAMES as BASE_FEATURES,
    PAIR_TABLES,
    ROUTE_TABLES,
    feat_vec,
    load_records,
    nfkc_casefold,
    norm_addr,
    norm_name,
    nums,
    s1_view,
    s3_boundaries,
    sig_tokens,
    sorted_sig,
)

POLICY_ID = "v4all_cap300_lgbm_m2_thr0.70"
CAP = 300
THRESHOLD = 0.7
AUX_TABLES = ("by_numtok", "by_numnum", "by_cpre8")
ROUTES = ["name", "sig", "prefix", "token", "tok_intersect", "pair", "toknum", "num", "numtok", "numnum", "cpre8"]
EXTRA_FEATURES = (
    ["rank_frac", "log_score", "score_ratio_to_top", "n_cands", "name_s_rank", "name_s_gap_to_best_other",
     "n_cands_name_ge_0.9", "addr_s_rank", "s1_addr_missing", "t_addr_missing", "num_conflict",
     "same_source_count_name_ge_0.9"]
    + [f"route_{r}" for r in ROUTES]
)
FEATURE_NAMES = BASE_FEATURES + EXTRA_FEATURES

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


class Retriever:
    def __init__(self, db, pair_db, aux_db, n_targets, bounds, pair_bounds, aux_bounds):
        self.db, self.pair_db, self.aux_db = db, pair_db, aux_db
        self.n_targets = n_targets
        self.bounds, self.pair_bounds, self.aux_bounds = bounds, pair_bounds, aux_bounds
        self.df_cache: dict[str, int] = {}
        self.n_log = math.log(max(n_targets, 2))

    def token_df(self, key):
        v = self.df_cache.get(key)
        if v is None:
            row = self.db.execute("SELECT df FROM token_df WHERE key=?", (key,)).fetchone()
            v = int(row[0]) if row else 0
            if len(self.df_cache) > 500_000:
                self.df_cache.clear()
            self.df_cache[key] = v
        return v

    @staticmethod
    def _split(db, bounds, table, key, limit):
        b = bounds[table]
        s2 = [r[0] for r in db.execute(f"SELECT id FROM {table} WHERE key=? AND rowid<=? LIMIT ?", (key, b, limit))]
        s3 = [r[0] for r in db.execute(f"SELECT id FROM {table} WHERE key=? AND rowid>? LIMIT ?", (key, b, limit))]
        if len(s2) + len(s3) <= limit:
            return s2 + s3
        n3 = min(len(s3), max(limit // 2, limit - len(s2)))
        return s2[: limit - n3] + s3[:n3]

    def retrieve(self, name, addr, country):
        """Return (top-CAP candidate ids, final scores, route membership)."""
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


def block_features(a, cands, scores, routes, recs) -> np.ndarray:
    """All 32 features for every candidate of one S1 (base 9 + extra)."""
    base = np.stack([feat_vec(a, recs[t]) for t in cands])
    n = len(cands)
    sc = np.asarray([scores[t] for t in cands], dtype=np.float32)
    name_s, addr_s = base[:, 0], base[:, 3]
    order_name = (-name_s).argsort(kind="stable")
    name_rank = np.empty(n, dtype=np.float32)
    name_rank[order_name] = np.arange(n)
    order_addr = (-addr_s).argsort(kind="stable")
    addr_rank = np.empty(n, dtype=np.float32)
    addr_rank[order_addr] = np.arange(n)
    top2 = np.sort(name_s)[::-1][:2]
    best_other = np.where(name_s >= top2[0], top2[1] if n > 1 else 0.0, top2[0])
    strong = name_s >= 0.9
    src = np.asarray([t.startswith("S2-") for t in cands])
    same_src_strong = np.where(src, (strong & src).sum(), (strong & ~src).sum()).astype(np.float32)
    s1_addr_missing = float(not (a.get("addr_norm") or "").strip())
    t_addr_missing = np.asarray([float(not (recs[t]["addr_norm"] or "").strip()) for t in cands], dtype=np.float32)
    a_nums = set(nums(a.get("addr_norm") or ""))
    num_conflict = np.asarray(
        [float(bool(a_nums) and bool(tn := set(nums(recs[t]["addr_norm"] or ""))) and not (a_nums & tn)) for t in cands],
        dtype=np.float32,
    )
    route_m = np.asarray([[float(r in routes.get(t, ())) for r in ROUTES] for t in cands], dtype=np.float32)
    extra = np.column_stack([
        np.arange(n, dtype=np.float32) / CAP,
        np.log1p(sc),
        sc / max(float(sc.max()), 1e-6),
        np.full(n, n / CAP, dtype=np.float32),
        name_rank / CAP,
        name_s - best_other,
        np.full(n, strong.sum(), dtype=np.float32),
        addr_rank / CAP,
        np.full(n, s1_addr_missing, dtype=np.float32),
        t_addr_missing,
        num_conflict,
        same_src_strong,
        route_m,
    ])
    return np.hstack([base, extra])
