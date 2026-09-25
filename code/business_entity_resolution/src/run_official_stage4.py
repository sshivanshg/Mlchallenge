"""Official Stage 4 experiments on challenge data only.

Improves blocking (IDF / rare-token priority, accent fold, sorted signatures),
trains logistic on fit-fold hard negatives, tunes threshold on select,
optionally scores untouched assess. Never invents metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sqlite3
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.provenance import load_manifest, require_official_dataset
from evaluation.metric_ext import candidate_diagnostics, score_predictions
from run_official_bounded import (
    SOURCE_H,
    GT_H,
    append_exp,
    exact_match,
    iter_tsv,
    load_records,
    nums,
    pair_score,
)

REPO = SRC.parents[2]
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
# Extremely common legal/org tokens — never use alone as blocking keys.
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


def sig_tokens(text: str) -> list[str]:
    return [t for t in text.split() if len(t) >= 3 and t not in STOP and t not in LEGAL_STOP]


def sorted_sig(text: str) -> str:
    toks = sorted(set(sig_tokens(text)))
    return " ".join(toks) if toks else ""


def build_target_index_v2(dataset_root: Path, db_path: Path) -> dict:
    """Rebuild S2+S3 index with accent-folded norms, rare-token focus, sorted sigs."""
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
        path = dataset_root / "train" / f"train_source{source}.tsv"
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
                print(f"  indexed targets: {stats['targets']:,}", flush=True)
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
        # Prefer rare tokens; fetch full posting for rare keys.
        ranked_toks = sorted(toks, key=lambda t: _token_df(db, t, df_cache) or n_targets)
        for tok in ranked_toks[:8]:
            df = _token_df(db, tok, df_cache) or 1
            idf = n_log - math.log(df)
            lim = min(2000, max(80, df)) if df <= rare_df else 40
            w = 1.2 * max(idf, 0.5)
            for eid in _lookup(db, "by_token", tok, lim):
                scores[eid] += w
        # Intersection boost: entities hit by >=2 rare tokens
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
        # street/house numbers are strong when combined with name evidence later
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


def weighted_score_v2(a: dict, b: dict) -> float:
    f = feat_vec(a, b)
    # Down-weight when name↔address contradict
    conflict = f[6]
    base = (
        0.32 * f[0]
        + 0.12 * f[1]
        + 0.10 * f[2]
        + 0.18 * f[3]
        + 0.12 * f[4]
        + 0.05 * f[5]
        + 0.08 * f[7]
        + 0.03 * f[8]
    )
    if conflict:
        base *= 0.55
    return float(np.clip(base, 0, 1))


def apply_contradiction_gate(preds: dict[str, set[str]], s1_map: dict, trecs_cache: dict[str, dict]) -> dict[str, set[str]]:
    """Drop high-name / low-address merges (singleton FP cutter)."""
    out = {k: set() for k in preds}
    for sid, tids in preds.items():
        a = s1_map[sid]
        aa = {**a, "name_norm": norm_name(a["business_name"]), "addr_norm": norm_addr(a["business_address"])}
        for tid in tids:
            b = trecs_cache.get(tid)
            if b is None:
                continue
            f = feat_vec(aa, b)
            if f[6] >= 1.0 and f[4] < 0.1:
                continue
            out[sid].add(tid)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Official Stage-4 ER experiments")
    p.add_argument("--dataset-root", type=Path, default=REPO / "student_resource" / "dataset")
    p.add_argument("--reports-root", type=Path, default=REPO / "reports" / "official")
    p.add_argument("--work-dir", type=Path, default=REPO / "artifacts" / "official_index")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-s1", type=int, default=5000)
    p.add_argument("--fit-s1", type=int, default=8000, help="Bounded fit S1s for logistic hard-negatives")
    p.add_argument("--assess-s1", type=int, default=3000, help="Untouched assess sample; 0 to skip")
    p.add_argument("--max-candidates", type=int, default=120)
    p.add_argument("--rebuild-index", action="store_true")
    p.add_argument("--skip-index-build", action="store_true")
    args = p.parse_args()

    t_all = time.time()
    print("== provenance ==", flush=True)
    prov = require_official_dataset(args.dataset_root, require=("train", "test"), check_hashes=True)
    manifest = load_manifest()
    gt_hash = manifest["files"]["train/train_ground_truth.tsv"]["sha256"]
    split_version = f"official_{gt_hash[:12]}"
    provenance = f"official_challenge_dataset:{prov.manifest_version}"
    reports = args.reports_root / split_version
    reports.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    split_db = args.work_dir / f"{split_version}_folds.sqlite"
    if not split_db.exists():
        raise SystemExit(f"Frozen split missing: {split_db}. Run run_official_bounded.py first.")

    fold_db = sqlite3.connect(split_db)
    rng = random.Random(args.seed)

    def sample_fold(fold: str, n: int) -> list[str]:
        rows = list(fold_db.execute("SELECT id FROM fold WHERE fold=? ORDER BY id", (fold,)))
        rng.shuffle(rows)
        return [r[0] for r in rows[: min(n, len(rows))]]

    select_ids = sample_fold("select", args.eval_s1)
    fit_ids = sample_fold("fit", args.fit_s1)
    assess_ids = sample_fold("assess", args.assess_s1) if args.assess_s1 > 0 else []
    all_needed = set(select_ids) | set(fit_ids) | set(assess_ids)
    print(
        f"== folds: select={len(select_ids)} fit={len(fit_ids)} assess={len(assess_ids)} ==",
        flush=True,
    )

    gt: dict[str, set[str]] = {i: set() for i in all_needed}
    for row in iter_tsv(args.dataset_root / "train" / "train_ground_truth.tsv", GT_H):
        sid = row["source1_entity_id"]
        if sid in all_needed:
            gt[sid] = {x for x in row["matched_entity_ids"].split(",") if x} if row["matched_entity_ids"] else set()

    s1_map: dict[str, dict] = {}
    for row in iter_tsv(args.dataset_root / "train" / "train_source1.tsv", SOURCE_H):
        if row["entity_id"] in all_needed:
            s1_map[row["entity_id"]] = row

    index_path = args.work_dir / "targets_index_v2.sqlite"
    if args.rebuild_index and index_path.exists():
        index_path.unlink()
    if not args.skip_index_build and not index_path.exists():
        print("== building improved S2/S3 index v2 ==", flush=True)
        t0 = time.time()
        stats = build_target_index_v2(args.dataset_root, index_path)
        stats["runtime_sec"] = time.time() - t0
        (reports / "index_v2_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
        print(json.dumps(stats, indent=2), flush=True)
    if not index_path.exists():
        raise SystemExit(f"Index missing: {index_path}")

    idb = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    n_targets = idb.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
    df_cache: dict[str, int] = {}

    def gen_all(ids: list[str]) -> dict[str, list[str]]:
        out = {}
        for i, sid in enumerate(ids):
            rec = s1_map[sid]
            out[sid] = generate_candidates_v2(
                idb,
                rec["business_name"],
                rec["business_address"],
                rec["country"],
                n_targets=n_targets,
                df_cache=df_cache,
                max_candidates=args.max_candidates,
            )
            if (i + 1) % 500 == 0:
                print(f"  candidates {i+1}/{len(ids)}", flush=True)
        return out

    # ----- E1: improved blocking on same select sample -----
    print("== E1: improved blocking + weighted_v2 on select ==", flush=True)
    t0 = time.time()
    select_cands = gen_all(select_ids)
    block_time = time.time() - t0
    select_gt = {i: gt[i] for i in select_ids}
    block_diag = candidate_diagnostics(select_cands, select_gt, n_targets)
    block_diag.update(
        {
            "runtime_sec": block_time,
            "n_eval_s1": len(select_ids),
            "n_target_universe": n_targets,
            "provenance": provenance,
            "split_version": split_version,
            "fold": "select_bounded",
            "blocking": "v2_idf_sig_accent",
            "max_candidates": args.max_candidates,
        }
    )
    (reports / "blocking_stage4_e1.json").write_text(json.dumps(block_diag, indent=2) + "\n")
    print(json.dumps(block_diag, indent=2), flush=True)

    exp_csv = reports / "experiments.csv"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def log(eid, hyp, model, metrics, fold, **kw):
        append_exp(
            exp_csv,
            {
                "experiment_id": eid,
                "timestamp": ts,
                "hypothesis": hyp,
                "validation_version": split_version,
                "blocking_config": kw.get("blocking_config", "v2_idf_sig_accent"),
                "feature_config": kw.get("feature_config", ""),
                "model": model,
                "decision_rule": kw.get("decision_rule", ""),
                "threshold": kw.get("threshold", ""),
                "candidate_recall": kw.get("candidate_recall", block_diag["micro_candidate_recall"]),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "macro_f0.5": metrics.get("macro_F0.5"),
                "singleton_accuracy": metrics.get("singleton_accuracy"),
                "avg_candidates": kw.get("avg_candidates", block_diag["avg_candidates"]),
                "runtime": kw.get("runtime", ""),
                "notes": kw.get("notes", ""),
                "data_provenance": provenance,
                "fold": fold,
            },
        )
        print(
            f"{eid} [{fold}]: macro_F0.5={metrics['macro_F0.5']:.4f} "
            f"P={metrics.get('precision')} R={metrics.get('recall')} sing={metrics.get('singleton_accuracy')}",
            flush=True,
        )

    # Score weighted_v2 with threshold search on select
    t0 = time.time()
    all_scores = []
    trec_cache: dict[str, dict] = {}
    for sid in select_ids:
        a = s1_map[sid]
        aa = {**a, "name_norm": norm_name(a["business_name"]), "addr_norm": norm_addr(a["business_address"])}
        missing = [c for c in select_cands[sid] if c not in trec_cache]
        if missing:
            trec_cache.update(load_records(idb, missing))
        for tid in select_cands[sid]:
            b = trec_cache.get(tid)
            if not b:
                continue
            all_scores.append((sid, tid, weighted_score_v2(aa, b)))

    best_t, best_m, best_pred = 0.5, None, None
    for thr in np.linspace(0.35, 0.95, 25):
        pred = {i: set() for i in select_ids}
        for sid, tid, sc in all_scores:
            if sc >= thr:
                pred[sid].add(tid)
        m = score_predictions(select_gt, pred)
        if best_m is None or m["macro_F0.5"] > best_m["macro_F0.5"]:
            best_t, best_m, best_pred = float(thr), m, pred
    log(
        "E1_weighted_v2_select",
        "IDF/sig/accent blocking + weighted_v2 with contradiction soft penalty; thr on select",
        "weighted_v2",
        best_m,
        "select_bounded",
        runtime=time.time() - t0,
        threshold=best_t,
        decision_rule="global_threshold_on_select",
        feature_config="token_sort+jac+num+partial+conflict_penalty",
        notes="selection-fold score; not untouched assess",
    )

    # E1b contradiction hard gate
    gated = apply_contradiction_gate(best_pred, s1_map, trec_cache)
    m_gate = score_predictions(select_gt, gated)
    log(
        "E1b_contradiction_gate_select",
        "Hard drop name-high/addr-low pairs without number overlap",
        "weighted_v2+gate",
        m_gate,
        "select_bounded",
        threshold=best_t,
        decision_rule="threshold_then_contradiction_gate",
        notes="selection-fold",
    )

    # ----- E2: logistic on fit hard negatives -----
    print("== E2: logistic on fit candidates; tune on select ==", flush=True)
    t0 = time.time()
    fit_cands = gen_all(fit_ids)
    X_rows, y_rows = [], []
    for sid in fit_ids:
        a = s1_map[sid]
        aa = {**a, "name_norm": norm_name(a["business_name"]), "addr_norm": norm_addr(a["business_address"])}
        truth = gt[sid]
        missing = [c for c in fit_cands[sid] if c not in trec_cache]
        if missing:
            trec_cache.update(load_records(idb, missing))
        pos = [c for c in fit_cands[sid] if c in truth]
        neg = [c for c in fit_cands[sid] if c not in truth]
        # keep all positives in candidates; sample hard negs
        rng.shuffle(neg)
        neg = neg[: max(8, 3 * max(len(pos), 1))]
        for tid in pos:
            b = trec_cache.get(tid)
            if b:
                X_rows.append(feat_vec(aa, b))
                y_rows.append(1)
        for tid in neg:
            b = trec_cache.get(tid)
            if b:
                X_rows.append(feat_vec(aa, b))
                y_rows.append(0)
    X = np.stack(X_rows) if X_rows else np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    y = np.asarray(y_rows, dtype=np.int32)
    print(f"  logistic train pairs: n={len(y)} pos={int(y.sum())} neg={int((1-y).sum())}", flush=True)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed, solver="lbfgs")
    clf.fit(Xs, y)

    # score select with logistic
    select_pair = []
    for sid in select_ids:
        a = s1_map[sid]
        aa = {**a, "name_norm": norm_name(a["business_name"]), "addr_norm": norm_addr(a["business_address"])}
        for tid in select_cands[sid]:
            b = trec_cache.get(tid)
            if not b:
                continue
            select_pair.append((sid, tid, feat_vec(aa, b)))
    if select_pair:
        Xsel = np.stack([f for _, _, f in select_pair])
        probs = clf.predict_proba(scaler.transform(Xsel))[:, 1]
    else:
        probs = np.array([])
    best_lt, best_lm, best_lpred = 0.5, None, None
    for thr in np.linspace(0.2, 0.95, 31):
        pred = {i: set() for i in select_ids}
        for (sid, tid, _), pr in zip(select_pair, probs):
            if pr >= thr:
                pred[sid].add(tid)
        m = score_predictions(select_gt, pred)
        if best_lm is None or m["macro_F0.5"] > best_lm["macro_F0.5"]:
            best_lt, best_lm, best_lpred = float(thr), m, pred
    log(
        "E2_logistic_select",
        "Logistic fit on fit-fold hard negatives from v2 candidates; thr tuned on select",
        "logistic",
        best_lm,
        "select_bounded",
        runtime=time.time() - t0,
        threshold=best_lt,
        decision_rule="global_threshold_on_select",
        feature_config=",".join(FEATURE_NAMES),
        notes="selection-fold; model fit on disjoint fit S1s",
    )

    gated_l = apply_contradiction_gate(best_lpred, s1_map, trec_cache)
    m_lg = score_predictions(select_gt, gated_l)
    log(
        "E2b_logistic_gate_select",
        "Logistic + contradiction gate on select",
        "logistic+gate",
        m_lg,
        "select_bounded",
        threshold=best_lt,
        decision_rule="threshold_then_contradiction_gate",
        notes="selection-fold",
    )

    # ----- E3: untouched assess with frozen threshold from select -----
    assess_summary = None
    if assess_ids:
        print("== E3: untouched assess with frozen select threshold ==", flush=True)
        t0 = time.time()
        assess_cands = gen_all(assess_ids)
        assess_gt = {i: gt[i] for i in assess_ids}
        a_block = candidate_diagnostics(assess_cands, assess_gt, n_targets)
        a_block.update(
            {
                "runtime_sec": time.time() - t0,
                "fold": "assess_bounded",
                "provenance": provenance,
                "split_version": split_version,
            }
        )
        (reports / "blocking_stage4_assess.json").write_text(json.dumps(a_block, indent=2) + "\n")

        # choose best select model for assess: max of E1 / E2 / gated
        select_leaders = [
            ("weighted_v2", best_t, best_m, "weighted"),
            ("weighted_v2+gate", best_t, m_gate, "weighted_gate"),
            ("logistic", best_lt, best_lm, "logistic"),
            ("logistic+gate", best_lt, m_lg, "logistic_gate"),
        ]
        leader = max(select_leaders, key=lambda x: x[2]["macro_F0.5"])
        use_model, use_thr, _, tag = leader
        print(f"  assess using select-leader={use_model} thr={use_thr}", flush=True)

        pred_a = {i: set() for i in assess_ids}
        for sid in assess_ids:
            a = s1_map[sid]
            aa = {**a, "name_norm": norm_name(a["business_name"]), "addr_norm": norm_addr(a["business_address"])}
            missing = [c for c in assess_cands[sid] if c not in trec_cache]
            if missing:
                trec_cache.update(load_records(idb, missing))
            feats = []
            tids = []
            for tid in assess_cands[sid]:
                b = trec_cache.get(tid)
                if not b:
                    continue
                feats.append(feat_vec(aa, b))
                tids.append(tid)
            if not feats:
                continue
            Xa = np.stack(feats)
            if "logistic" in tag:
                pr = clf.predict_proba(scaler.transform(Xa))[:, 1]
            else:
                pr = np.array(
                    [
                        weighted_score_v2(aa, trec_cache[tid])
                        for tid in tids
                    ]
                )
            chosen = {tid for tid, sc in zip(tids, pr) if sc >= use_thr}
            if "gate" in tag:
                tmp = {sid: chosen}
                chosen = apply_contradiction_gate(tmp, s1_map, trec_cache)[sid]
            pred_a[sid] = chosen

        m_assess = score_predictions(assess_gt, pred_a)
        log(
            f"E3_assess_{tag}",
            f"Untouched assess with frozen select threshold from {use_model}",
            use_model,
            m_assess,
            "assess_bounded",
            runtime=time.time() - t0,
            threshold=use_thr,
            decision_rule="frozen_threshold_from_select",
            candidate_recall=a_block["micro_candidate_recall"],
            avg_candidates=a_block["avg_candidates"],
            notes="UNTOUCHED assessment; threshold not retuned",
        )
        assess_summary = {
            "model": use_model,
            "threshold": use_thr,
            "metrics": m_assess,
            "blocking": a_block,
        }

    # Error analysis vs E1 best
    err = {"singleton_fp": 0, "fp": 0, "fn_blocking": 0, "fn_threshold": 0, "examples": []}
    for sid, truth in select_gt.items():
        pred = best_lpred.get(sid, set()) if best_lm["macro_F0.5"] >= best_m["macro_F0.5"] else best_pred.get(sid, set())
        cset = set(select_cands.get(sid, []))
        if not truth and pred:
            err["singleton_fp"] += 1
        err["fp"] += len(pred - truth)
        for m_id in truth - pred:
            if m_id not in cset:
                err["fn_blocking"] += 1
                if len(err["examples"]) < 15:
                    err["examples"].append(
                        {"type": "fn_blocking", "s1": sid, "miss": m_id, "name": s1_map[sid]["business_name"]}
                    )
            else:
                err["fn_threshold"] += 1
                if len(err["examples"]) < 15:
                    err["examples"].append(
                        {"type": "fn_threshold", "s1": sid, "miss": m_id, "name": s1_map[sid]["business_name"]}
                    )
    (reports / "error_analysis_stage4_select.json").write_text(json.dumps(err, indent=2) + "\n")

    summary = {
        "provenance": provenance,
        "split_version": split_version,
        "stage": 4,
        "n_select": len(select_ids),
        "n_fit": len(fit_ids),
        "n_assess": len(assess_ids),
        "n_target_universe": n_targets,
        "E1_blocking": block_diag,
        "E1_weighted_v2_select": {**best_m, "threshold": best_t},
        "E1b_gate_select": m_gate,
        "E2_logistic_select": {**best_lm, "threshold": best_lt},
        "E2b_logistic_gate_select": m_lg,
        "E3_assess": assess_summary,
        "error_counts": {k: v for k, v in err.items() if k != "examples"},
        "runtime_total_sec": time.time() - t_all,
        "comparison_to_stage3": {
            "M0_weighted_select_was": 0.44474171904753546,
            "blocking_recall_was": 0.30411418005316077,
        },
    }
    (reports / "stage4_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("== STAGE4 DONE ==", json.dumps({
        "reports": str(reports),
        "E1_F0.5": best_m["macro_F0.5"],
        "E2_F0.5": best_lm["macro_F0.5"],
        "assess_F0.5": None if not assess_summary else assess_summary["metrics"]["macro_F0.5"],
        "blocking_recall": block_diag["micro_candidate_recall"],
    }, indent=2), flush=True)
    idb.close()
    fold_db.close()


if __name__ == "__main__":
    main()
