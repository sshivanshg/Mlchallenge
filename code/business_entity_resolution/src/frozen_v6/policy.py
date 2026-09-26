"""Frozen inference policy v6 (candidate: v5b addrtok retrieval + H4 capacity LightGBM).

Retrieval = frozen v4all routes + selective rare address-token postings from
train/test_aux_index_v5b (by_addrtok, DF<=5000). Name 4-gram route is rejected
(select oracle Δ −0.0027 via cap displacement). Matcher features are the frozen
v5 43-feature set unchanged so H4/v5 boosters transfer; route_addrtok is NOT a
feature (avoids dimension change). Do not edit; version a new package.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from frozen_v4.policy import (  # frozen, immutable
    AUX_TABLES,
    CAP,
    MAX_NUMS,
    PAIR_TABLES,
    ROUTE_TABLES,
    ROUTES,
    Retriever as RetrieverV4,
    block_features,
    compact_prefix,
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
from frozen_v4.policy import FEATURE_NAMES as V4_FEATURES
from frozen_v4.policy import addr_tokens

POLICY_ID = "v6_addrtok_cap300_lgbm_h4"
V5B_TABLES = ("by_addrtok",)
ADDRTOK_MAX_DF = 5000
EXTRA2_NAMES = [
    "name_jw", "name_token_set", "addr_token_set", "addr_partial", "first_num_equal",
    "long_num_shared", "long_num_conflict", "name_len_ratio", "addr_len_ratio",
    "name_tok_contain_s1", "name_tok_contain_t",
]
FEATURE_NAMES = V4_FEATURES + EXTRA2_NAMES


class Retriever(RetrieverV4):
    """v4all retriever + selective by_addrtok before final CAP."""

    def __init__(self, db, pair_db, aux_db, v5b_db, n_targets, bounds, pair_bounds, aux_bounds, v5b_bounds,
                 addrtok_max_df: int = ADDRTOK_MAX_DF):
        super().__init__(db, pair_db, aux_db, n_targets, bounds, pair_bounds, aux_bounds)
        self.v5b_db = v5b_db
        self.v5b_bounds = v5b_bounds
        self.addrtok_max_df = addrtok_max_df
        self.adf_cache: dict[str, int] = {}

    def addr_df(self, key: str) -> int:
        v = self.adf_cache.get(key)
        if v is None:
            row = self.v5b_db.execute("SELECT df FROM addr_token_df WHERE key=?", (key,)).fetchone()
            v = int(row[0]) if row else 0
            if len(self.adf_cache) > 200_000:
                self.adf_cache.clear()
            self.adf_cache[key] = v
        return v

    def retrieve(self, name, addr, country):
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

        # v5b selective address tokens (kept); ngram rejected on select screen
        for t in addr_tokens(aa):
            df = self.addr_df(t)
            if not df or df > self.addrtok_max_df:
                continue
            idf = self.n_log - math.log(df)
            lim = min(400, max(40, df)) if df <= 800 else min(120, max(40, df // 20))
            add(vlk("by_addrtok", t, lim), 2.5 * max(idf, 0.5), "addrtok")

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


def extra2(a, recs, cands) -> np.ndarray:
    na = a["name_norm"] or ""
    aa = a["addr_norm"] or ""
    a_nums = nums(aa)
    a_long = {n for n in a_nums if len(n) >= 5}
    ta = set(sig_tokens(na))
    rows = []
    for t in cands:
        b = recs[t]
        nb, ab = b["name_norm"] or "", b["addr_norm"] or ""
        b_nums = nums(ab)
        b_long = {n for n in b_nums if len(n) >= 5}
        tb = set(sig_tokens(nb))
        rows.append([
            JaroWinkler.normalized_similarity(na, nb) if (na or nb) else 0.0,
            fuzz.token_set_ratio(na, nb) / 100.0 if (na or nb) else 0.0,
            fuzz.token_set_ratio(aa, ab) / 100.0 if (aa or ab) else 0.0,
            fuzz.partial_ratio(aa, ab) / 100.0 if (aa or ab) else 0.0,
            float(bool(a_nums) and bool(b_nums) and a_nums[0] == b_nums[0]),
            float(bool(a_long & b_long)),
            float(bool(a_long) and bool(b_long) and not (a_long & b_long)),
            min(len(na), len(nb)) / max(len(na), len(nb), 1),
            min(len(aa), len(ab)) / max(len(aa), len(ab), 1),
            len(ta & tb) / len(ta) if ta else 0.0,
            len(ta & tb) / len(tb) if tb else 0.0,
        ])
    return np.asarray(rows, dtype=np.float32)


def features(a, cands, scores, routes, recs) -> np.ndarray:
    return np.hstack([block_features(a, cands, scores, routes, recs), extra2(a, recs, cands)])
