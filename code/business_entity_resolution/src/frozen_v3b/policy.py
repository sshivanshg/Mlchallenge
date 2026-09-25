"""Frozen inference policy v3b_cap300_lgbm (selected 2026-09-25).

Self-contained copy of the retrieval, normalization, and feature code that
produced the selected model, so later research edits to shared modules cannot
change inference. Selection evidence: reports/official/official_70bc1d8a16c6/
retrieval_exp1/summary.json. Do not edit; create a new versioned package.
"""

from __future__ import annotations

import math
import re
import sqlite3
import unicodedata
from collections import defaultdict

import numpy as np
from rapidfuzz import fuzz

POLICY_ID = "v3b_cap300_lgbm_thr0.98_nogate"
CAP = 300
THRESHOLD = 0.98
FEATURE_NAMES = [
    "name_token_sort",
    "name_jaccard",
    "name_exact",
    "addr_token_sort",
    "num_overlap",
    "country_equal",
    "name_addr_conflict",
    "name_partial",
    "addr_jaccard",
]
ROUTE_TABLES = ("by_name", "by_sig", "by_prefix", "by_token", "by_num")
PAIR_TABLES = ("by_pair", "by_toknum")

_AND = re.compile(r"\s*&\s*")
_NON_ALNUM = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")
_DIGIT = re.compile(r"\d+")
_DOMAIN = re.compile(r"\.(com|net|org|io|co|in|fr|us)\b", re.I)
_LEGAL = {"corp": "corporation", "inc": "incorporated", "ltd": "limited", "llc": "limited",
          "pvt": "private", "co": "company", "llp": "limited"}
_STREET = {"rd": "road", "st": "street", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
           "ln": "lane", "dr": "drive"}
STOP = {"the", "and", "of", "a", "an", "for", "to", "in", "on", "at", "by"}
LEGAL_STOP = {"private", "limited", "corporation", "incorporated", "company", "companies", "group",
              "holdings", "services", "service", "solutions", "international", "india", "llc", "ltd",
              "inc", "corp", "pvt"}


def accent_fold(text: str) -> str:
    t = unicodedata.normalize("NFKD", text or "")
    return "".join(c for c in t if not unicodedata.combining(c))


def nfkc_casefold(text: str) -> str:
    return accent_fold(unicodedata.normalize("NFKC", text or "")).casefold()


def norm_name(text: str) -> str:
    t = nfkc_casefold(text)
    t = _DOMAIN.sub(" ", t)
    t = _AND.sub(" and ", t)
    t = _NON_ALNUM.sub(" ", t)
    t = _WS.sub(" ", t).strip()
    return " ".join(_LEGAL.get(tok, tok) for tok in t.split())


def norm_addr(text: str) -> str:
    t = nfkc_casefold(text)
    t = _AND.sub(" and ", t)
    t = _NON_ALNUM.sub(" ", t)
    t = _WS.sub(" ", t).strip()
    mapping = {**_LEGAL, **_STREET}
    return " ".join(mapping.get(tok, tok) for tok in t.split())


def nums(text: str) -> list[str]:
    return [n for n in _DIGIT.findall(text or "") if len(n) >= 3]


def sig_tokens(text: str) -> list[str]:
    return [t for t in text.split() if len(t) >= 3 and t not in STOP and t not in LEGAL_STOP]


def sorted_sig(text: str) -> str:
    toks = sorted(set(sig_tokens(text)))
    return " ".join(toks) if toks else ""


def s3_boundaries(db: sqlite3.Connection, tables) -> dict[str, int]:
    """Last rowid holding an S2 id; index rows were inserted S2 file then S3 file."""
    out = {}
    for t in tables:
        hi = db.execute(f"SELECT MAX(rowid) FROM {t}").fetchone()[0]
        lo = 1
        first_id = db.execute(f"SELECT id FROM {t} WHERE rowid=?", (lo,)).fetchone()[0]
        last_id = db.execute(f"SELECT id FROM {t} WHERE rowid=?", (hi,)).fetchone()[0]
        if not first_id.startswith("S2-") or not last_id.startswith("S3-"):
            raise RuntimeError(f"{t}: unexpected source order {first_id}..{last_id}")
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if db.execute(f"SELECT id FROM {t} WHERE rowid=?", (mid,)).fetchone()[0].startswith("S2-"):
                lo = mid
            else:
                hi = mid
        out[t] = lo
    return out


class Retriever:
    def __init__(self, db, pair_db, n_targets, bounds, pair_bounds):
        self.db = db
        self.pair_db = pair_db
        self.n_targets = n_targets
        self.bounds = bounds
        self.pair_bounds = pair_bounds
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

    def candidates(self, name, addr, country) -> list[str]:
        nn, aa = norm_name(name), norm_addr(addr)
        scores: dict[str, float] = defaultdict(float)

        def add(ids, w):
            for eid in ids:
                scores[eid] += w

        lk = lambda table, key, limit: self._split(self.db, self.bounds, table, key, limit)
        plk = lambda table, key, limit: self._split(self.pair_db, self.pair_bounds, table, key, limit)
        if nn:
            add(lk("by_name", nn, 200), 8.0)
            sig = sorted_sig(nn)
            if sig:
                add(lk("by_sig", sig, 200), 6.0)
            compact = "".join(nn.split())
            if len(compact) >= 4:
                add(lk("by_prefix", compact[:4], 60), 0.8)
            toks = sig_tokens(nn)
            ranked_toks = sorted(toks, key=lambda t: self.token_df(t) or self.n_targets)
            for tok in ranked_toks[:8]:
                df = self.token_df(tok) or 1
                idf = self.n_log - math.log(df)
                lim = min(2000, max(80, df)) if df <= 800 else 40
                add(lk("by_token", tok, lim), 1.2 * max(idf, 0.5))
            if len(ranked_toks) >= 2:
                sets = []
                for tok in ranked_toks[:4]:
                    df = self.token_df(tok) or 1
                    if df > 5000:
                        continue
                    lim = min(2000, max(80, df)) if df <= 800 else 40
                    sets.append(set(lk("by_token", tok, lim)))
                if len(sets) >= 2:
                    add(sets[0].intersection(*sets[1:]), 4.0)
            top = sorted(set(toks), key=lambda t: self.token_df(t) or self.n_targets)[:6]
            for i in range(len(top)):
                for j in range(i + 1, len(top)):
                    add(plk("by_pair", "|".join(sorted((top[i], top[j]))), 200), 6.0)
            anums = list(dict.fromkeys(nums(aa)))[:4]
            for t in top[:4]:
                for n in anums:
                    add(plk("by_toknum", f"{t}#{n}", 100), 5.0)
        for num in nums(aa):
            add(lk("by_num", num, 80), 2.2)
        if not scores:
            return []
        ids = list(scores.keys())
        soft = {}
        for i in range(0, len(ids), 500):
            chunk = ids[i : i + 500]
            q = ",".join("?" * len(chunk))
            for eid, c in self.db.execute(f"SELECT id, country FROM targets WHERE id IN ({q})", chunk):
                soft[eid] = 0.55 if (country and c and nfkc_casefold(country) != nfkc_casefold(c)) else 1.0
        final = {e: s * soft.get(e, 1.0) for e, s in scores.items()}
        ranked = sorted(final.items(), key=lambda kv: (-kv[1], kv[0]))
        return [e for e, _ in ranked[:CAP]]


def load_records(db: sqlite3.Connection, ids) -> dict[str, dict]:
    ids = list(ids)
    out = {}
    for i in range(0, len(ids), 500):
        chunk = ids[i : i + 500]
        q = ",".join("?" * len(chunk))
        for eid, name, addr, country, nn, aa in db.execute(
            f"SELECT id, name, addr, country, name_norm, addr_norm FROM targets WHERE id IN ({q})", chunk
        ):
            out[eid] = {"business_name": name, "business_address": addr, "country": country,
                        "name_norm": nn, "addr_norm": aa}
    missing = set(ids) - set(out)
    if missing:
        raise RuntimeError(f"{len(missing)} candidate ids missing from target table, e.g. {sorted(missing)[:3]}")
    return out


def feat_vec(a: dict, b: dict) -> np.ndarray:
    na = a.get("name_norm") or norm_name(a["business_name"])
    nb = b.get("name_norm") or norm_name(b["business_name"])
    aa = a.get("addr_norm") or norm_addr(a["business_address"])
    ab = b.get("addr_norm") or norm_addr(b["business_address"])
    name_s = fuzz.token_sort_ratio(na, nb) / 100.0 if (na or nb) else 0.0
    addr_s = fuzz.token_sort_ratio(aa, ab) / 100.0 if (aa or ab) else 0.0
    ta, tb = set(sig_tokens(na)), set(sig_tokens(nb))
    jac = (len(ta & tb) / len(ta | tb)) if (ta or tb) else 0.0
    ra, rb = set(sig_tokens(aa)), set(sig_tokens(ab))
    aj = (len(ra & rb) / len(ra | rb)) if (ra or rb) else 0.0
    exact = 1.0 if na and nb and na == nb else 0.0
    na_n, nb_n = set(nums(aa)), set(nums(ab))
    num = (len(na_n & nb_n) / len(na_n | nb_n)) if (na_n and nb_n) else 0.0
    c_eq = 1.0 if a.get("country") and b.get("country") and nfkc_casefold(a["country"]) == nfkc_casefold(b["country"]) else 0.0
    conflict = 1.0 if (name_s >= 0.85 and addr_s > 0 and addr_s < 0.45) else 0.0
    partial = fuzz.partial_ratio(na, nb) / 100.0 if (na or nb) else 0.0
    return np.asarray([name_s, jac, exact, addr_s, num, c_eq, conflict, partial, aj], dtype=np.float32)


def s1_view(row: dict) -> dict:
    return {**row, "name_norm": norm_name(row["business_name"]), "addr_norm": norm_addr(row["business_address"])}
