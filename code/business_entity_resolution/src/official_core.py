"""Shared official ER helpers: normalization, index v2, candidates, features."""

from __future__ import annotations

import math
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz

_AND = re.compile(r"\s*&\s*")
_NON_ALNUM = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")
_DIGIT = re.compile(r"\d+")
_DOMAIN = re.compile(r"\.(com|net|org|io|co|in|fr|us)\b", re.I)
_LEGAL = {
    "corp": "corporation",
    "inc": "incorporated",
    "ltd": "limited",
    "llc": "limited",
    "pvt": "private",
    "co": "company",
    "llp": "limited",
}
_STREET = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "ln": "lane",
    "dr": "drive",
}
STOP = {"the", "and", "of", "a", "an", "for", "to", "in", "on", "at", "by"}
LEGAL_STOP = {
    "private",
    "limited",
    "corporation",
    "incorporated",
    "company",
    "companies",
    "group",
    "holdings",
    "services",
    "service",
    "solutions",
    "international",
    "india",
    "llc",
    "ltd",
    "inc",
    "corp",
    "pvt",
}

SOURCE_H = ["entity_id", "business_name", "business_address", "country"]
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


def iter_tsv(path: Path, expected_header: list[str]):
    import csv

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if list(reader.fieldnames or []) != expected_header:
            raise ValueError(f"Bad header in {path}: {reader.fieldnames}")
        for row in reader:
            yield row


def build_target_index_v2(dataset_root: Path, db_path: Path, split: str = "train") -> dict:
    """Stream S2+S3 into SQLite inverted indexes."""
    import time

    if db_path.exists():
        db_path.unlink()
    db = sqlite3.connect(db_path)
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute(
        "CREATE TABLE targets (id TEXT PRIMARY KEY, name TEXT, addr TEXT, country TEXT, "
        "name_norm TEXT, addr_norm TEXT) WITHOUT ROWID"
    )
    db.execute("CREATE TABLE by_name (key TEXT, id TEXT)")
    db.execute("CREATE TABLE by_prefix (key TEXT, id TEXT)")
    db.execute("CREATE TABLE by_token (key TEXT, id TEXT)")
    db.execute("CREATE TABLE by_num (key TEXT, id TEXT)")
    db.execute("CREATE TABLE by_sig (key TEXT, id TEXT)")
    db.execute("CREATE TABLE token_df (key TEXT PRIMARY KEY, df INTEGER) WITHOUT ROWID")

    stats = Counter()
    tok_df: Counter = Counter()
    batch_t, batch_n, batch_p, batch_tok, batch_num, batch_sig = [], [], [], [], [], []
    t0 = time.time()

    def flush():
        nonlocal batch_t, batch_n, batch_p, batch_tok, batch_num, batch_sig
        if batch_t:
            db.executemany("INSERT INTO targets VALUES (?,?,?,?,?,?)", batch_t)
            batch_t.clear()
        if batch_n:
            db.executemany("INSERT INTO by_name VALUES (?,?)", batch_n)
            batch_n.clear()
        if batch_p:
            db.executemany("INSERT INTO by_prefix VALUES (?,?)", batch_p)
            batch_p.clear()
        if batch_tok:
            db.executemany("INSERT INTO by_token VALUES (?,?)", batch_tok)
            batch_tok.clear()
        if batch_num:
            db.executemany("INSERT INTO by_num VALUES (?,?)", batch_num)
            batch_num.clear()
        if batch_sig:
            db.executemany("INSERT INTO by_sig VALUES (?,?)", batch_sig)
            batch_sig.clear()

    for source in (2, 3):
        path = dataset_root / split / f"{split}_source{source}.tsv"
        for row in iter_tsv(path, SOURCE_H):
            eid = row["entity_id"]
            nn = norm_name(row["business_name"])
            aa = norm_addr(row["business_address"])
            batch_t.append((eid, row["business_name"], row["business_address"], row["country"], nn, aa))
            if nn:
                batch_n.append((nn, eid))
                compact = "".join(nn.split())
                if len(compact) >= 3:
                    batch_p.append((compact[:4], eid))
                seen = set()
                for tok in sig_tokens(nn):
                    if tok in seen:
                        continue
                    seen.add(tok)
                    batch_tok.append((tok, eid))
                    tok_df[tok] += 1
                sig = sorted_sig(nn)
                if sig:
                    batch_sig.append((sig, eid))
            for num in nums(aa):
                batch_num.append((num, eid))
            stats["targets"] += 1
            if stats["targets"] % 50000 == 0:
                flush()
                db.commit()
                print(f"  indexed {split} targets: {stats['targets']:,}", flush=True)
    flush()
    print("  writing token_df + indexes...", flush=True)
    db.executemany("INSERT INTO token_df VALUES (?,?)", list(tok_df.items()))
    db.execute("CREATE INDEX IF NOT EXISTS idx_name ON by_name(key)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_prefix ON by_prefix(key)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_token ON by_token(key)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_num ON by_num(key)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_sig ON by_sig(key)")
    db.commit()
    db.close()
    stats["n_tokens"] = len(tok_df)
    stats["runtime_sec"] = time.time() - t0
    stats["split"] = split
    return dict(stats)


def _lookup(db: sqlite3.Connection, table: str, key: str, limit: int) -> list[str]:
    cur = db.execute(f"SELECT id FROM {table} WHERE key=? LIMIT ?", (key, limit))
    return [r[0] for r in cur.fetchall()]


def _token_df(db: sqlite3.Connection, key: str, cache: dict[str, int]) -> int:
    if key in cache:
        return cache[key]
    row = db.execute("SELECT df FROM token_df WHERE key=?", (key,)).fetchone()
    cache[key] = int(row[0]) if row else 0
    return cache[key]


def generate_candidates_v2(
    db: sqlite3.Connection,
    name: str,
    addr: str,
    country: str,
    *,
    n_targets: int,
    df_cache: dict[str, int],
    max_candidates: int = 120,
    rare_df: int = 800,
) -> list[str]:
    nn, aa = norm_name(name), norm_addr(addr)
    scores: dict[str, float] = defaultdict(float)
    n_log = math.log(max(n_targets, 2))

    if nn:
        for eid in _lookup(db, "by_name", nn, 200):
            scores[eid] += 8.0
        sig = sorted_sig(nn)
        if sig:
            for eid in _lookup(db, "by_sig", sig, 200):
                scores[eid] += 6.0
        compact = "".join(nn.split())
        if len(compact) >= 4:
            for eid in _lookup(db, "by_prefix", compact[:4], 60):
                scores[eid] += 0.8
        toks = sig_tokens(nn)
        ranked_toks = sorted(toks, key=lambda t: _token_df(db, t, df_cache) or n_targets)
        for tok in ranked_toks[:8]:
            df = _token_df(db, tok, df_cache) or 1
            idf = n_log - math.log(df)
            lim = min(2000, max(80, df)) if df <= rare_df else 40
            w = 1.2 * max(idf, 0.5)
            for eid in _lookup(db, "by_token", tok, lim):
                scores[eid] += w
        if len(ranked_toks) >= 2:
            sets = []
            for tok in ranked_toks[:4]:
                df = _token_df(db, tok, df_cache) or 1
                if df > 5000:
                    continue
                lim = min(2000, max(80, df)) if df <= rare_df else 40
                sets.append(set(_lookup(db, "by_token", tok, lim)))
            if len(sets) >= 2:
                inter = sets[0].intersection(*sets[1:])
                for eid in inter:
                    scores[eid] += 4.0

    for num in nums(aa):
        for eid in _lookup(db, "by_num", num, 80):
            scores[eid] += 2.2

    if not scores:
        return []

    ids = list(scores.keys())
    soft: dict[str, float] = {}
    for i in range(0, len(ids), 500):
        chunk = ids[i : i + 500]
        q = ",".join("?" * len(chunk))
        for eid, c in db.execute(f"SELECT id, country FROM targets WHERE id IN ({q})", chunk):
            if country and c and nfkc_casefold(country) != nfkc_casefold(c):
                soft[eid] = 0.55
            else:
                soft[eid] = 1.0
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1] * soft.get(kv[0], 1.0), kv[0]))
    return [e for e, _ in ranked[:max_candidates]]


def load_records(db: sqlite3.Connection, ids: list[str]) -> dict[str, dict]:
    out = {}
    for i in range(0, len(ids), 500):
        chunk = ids[i : i + 500]
        q = ",".join("?" * len(chunk))
        for eid, name, addr, country, nn, aa in db.execute(
            f"SELECT id, name, addr, country, name_norm, addr_norm FROM targets WHERE id IN ({q})", chunk
        ):
            out[eid] = {
                "entity_id": eid,
                "business_name": name,
                "business_address": addr,
                "country": country,
                "name_norm": nn,
                "addr_norm": aa,
            }
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
    c_eq = (
        1.0
        if a.get("country") and b.get("country") and nfkc_casefold(a["country"]) == nfkc_casefold(b["country"])
        else 0.0
    )
    conflict = 1.0 if (name_s >= 0.85 and addr_s > 0 and addr_s < 0.45) else 0.0
    partial = fuzz.partial_ratio(na, nb) / 100.0 if (na or nb) else 0.0
    return np.asarray(
        [name_s, jac, exact, addr_s, num, c_eq, conflict, partial, aj], dtype=np.float32
    )


def accept_match(prob: float, feats: np.ndarray, threshold: float) -> bool:
    """Precision-oriented accept: high thr plus name/addr corroboration."""
    if prob < threshold:
        return False
    name_s, jac, exact, addr_s, num, c_eq, conflict, partial, aj = feats
    if conflict >= 1.0 and num < 0.1:
        return False
    # Require at least one strong corroborating signal
    if exact >= 1.0:
        return True
    if name_s >= 0.88 and (addr_s >= 0.55 or num >= 0.25 or jac >= 0.6):
        return True
    if name_s >= 0.92:
        return True
    if jac >= 0.7 and (addr_s >= 0.5 or num >= 0.2):
        return True
    if partial >= 0.95 and addr_s >= 0.6:
        return True
    # Very high model confidence still needs mild name evidence
    if prob >= max(threshold, 0.98) and name_s >= 0.80:
        return True
    return False
