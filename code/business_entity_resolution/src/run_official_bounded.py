"""Official-data bounded experiments (Stages 1–3).

Streaming / disk-backed. Never loads full S2/S3 into pandas.
Uses only verified challenge TSVs under student_resource/dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.provenance import load_manifest, require_official_dataset, sha256_file
from evaluation.metric_ext import candidate_diagnostics, score_predictions

REPO = SRC.parents[2]
_AND = re.compile(r"\s*&\s*")
_NON_ALNUM = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")
_DIGIT = re.compile(r"\d+")
_LEGAL = {"corp": "corporation", "inc": "incorporated", "ltd": "limited", "llc": "limited", "pvt": "private", "co": "company"}
_STREET = {"rd": "road", "st": "street", "ave": "avenue", "av": "avenue", "blvd": "boulevard", "ln": "lane", "dr": "drive"}
STOP = {"the", "and", "of", "a", "an", "for", "to", "in", "on", "at", "by"}


def nfkc_casefold(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").casefold()


def norm_name(text: str) -> str:
    t = nfkc_casefold(text)
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
    return [t for t in text.split() if len(t) >= 3 and t not in STOP]


def nums(text: str) -> list[str]:
    return [n for n in _DIGIT.findall(text or "") if len(n) >= 3]


def iter_tsv(path: Path, expected_header: list[str]):
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if list(reader.fieldnames or []) != expected_header:
            raise ValueError(f"Bad header in {path}: {reader.fieldnames}")
        for row in reader:
            yield row


SOURCE_H = ["entity_id", "business_name", "business_address", "country"]
GT_H = ["source1_entity_id", "matched_entity_ids"]


class UnionFind:
    def __init__(self):
        self.p: dict[str, str] = {}

    def add(self, x: str) -> None:
        self.p.setdefault(x, x)

    def find(self, x: str) -> str:
        self.add(x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def freeze_official_split(
    dataset_root: Path,
    *,
    seed: int = 42,
    fit_frac: float = 0.70,
    select_frac: float = 0.15,
) -> tuple[dict, set[str], set[str], set[str], dict[str, int], dict[str, str]]:
    """Component-disjoint fit/select/assess S1 split from streaming GT."""
    t0 = time.time()
    uf = UnionFind()
    gt_card: dict[str, int] = {}
    countries: dict[str, str] = {}
    target_owners: dict[str, list[str]] = defaultdict(list)

    for row in iter_tsv(dataset_root / "train" / "train_source1.tsv", SOURCE_H):
        countries[row["entity_id"]] = row["country"]
        uf.add(row["entity_id"])

    for row in iter_tsv(dataset_root / "train" / "train_ground_truth.tsv", GT_H):
        sid = row["source1_entity_id"]
        ids = [x for x in row["matched_entity_ids"].split(",") if x] if row["matched_entity_ids"] else []
        gt_card[sid] = len(ids)
        uf.add(sid)
        for tid in ids:
            target_owners[tid].append(sid)

    for owners in target_owners.values():
        if len(owners) < 2:
            continue
        base = owners[0]
        for o in owners[1:]:
            uf.union(base, o)

    comps: dict[str, list[str]] = defaultdict(list)
    for sid in countries:
        comps[uf.find(sid)].append(sid)
    components = [sorted(v) for _, v in sorted(comps.items(), key=lambda kv: kv[1][0])]
    rng = random.Random(seed)
    order = list(range(len(components)))
    rng.shuffle(order)

    n = len(countries)
    n_fit = int(round(n * fit_frac))
    n_select = int(round(n * select_frac))
    fit, select, assess = set(), set(), set()
    for idx in order:
        comp = components[idx]
        if len(fit) < n_fit:
            fit.update(comp)
        elif len(select) < n_select:
            select.update(comp)
        else:
            assess.update(comp)
    if not assess or not fit or not select:
        raise RuntimeError("empty fold after component assignment; adjust fractions")

    def fold_stats(ids: set[str]) -> dict:
        cards = [gt_card.get(i, 0) for i in ids]
        cc = Counter(countries[i] for i in ids)
        return {
            "n_s1": len(ids),
            "singleton_rate": sum(c == 0 for c in cards) / max(len(cards), 1),
            "mean_matches": float(sum(cards) / max(len(cards), 1)),
            "country_counts": dict(cc),
        }

    meta = {
        "seed": seed,
        "fit_frac": fit_frac,
        "select_frac": select_frac,
        "strategy": "positive-component-disjoint fit/select/assess",
        "runtime_sec": time.time() - t0,
        "fit": fold_stats(fit),
        "select": fold_stats(select),
        "assess": fold_stats(assess),
    }
    return meta, fit, select, assess, gt_card, countries


def build_target_index(dataset_root: Path, db_path: Path, max_posting: int = 200) -> dict:
    """Stream S2+S3 into SQLite inverted indexes (full eligible universe)."""
    if db_path.exists():
        db_path.unlink()
    db = sqlite3.connect(db_path)
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("CREATE TABLE targets (id TEXT PRIMARY KEY, name TEXT, addr TEXT, country TEXT, name_norm TEXT, addr_norm TEXT) WITHOUT ROWID")
    db.execute("CREATE TABLE by_name (key TEXT, id TEXT)")
    db.execute("CREATE TABLE by_prefix (key TEXT, id TEXT)")
    db.execute("CREATE TABLE by_token (key TEXT, id TEXT)")
    db.execute("CREATE TABLE by_num (key TEXT, id TEXT)")

    stats = Counter()
    batch_t, batch_n, batch_p, batch_tok, batch_num = [], [], [], [], []

    def flush():
        nonlocal batch_t, batch_n, batch_p, batch_tok, batch_num
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

    for source in (2, 3):
        path = dataset_root / "train" / f"train_source{source}.tsv"
        for row in iter_tsv(path, SOURCE_H):
            eid = row["entity_id"]
            nn = norm_name(row["business_name"])
            aa = norm_addr(row["business_address"])
            batch_t.append((eid, row["business_name"], row["business_address"], row["country"], nn, aa))
            if nn:
                batch_n.append((nn, eid))
                pref = "".join(nn.split())[:3]
                if pref:
                    batch_p.append((pref, eid))
                for tok in sig_tokens(nn):
                    batch_tok.append((tok, eid))
            for num in nums(aa):
                batch_num.append((num, eid))
            stats["targets"] += 1
            if stats["targets"] % 50000 == 0:
                flush()
                db.commit()
                print(f"  indexed targets: {stats['targets']:,}", flush=True)
    flush()
    print("  creating index keys...", flush=True)
    db.execute("CREATE INDEX IF NOT EXISTS idx_name ON by_name(key)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_prefix ON by_prefix(key)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_token ON by_token(key)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_num ON by_num(key)")
    db.commit()
    db.close()
    stats["max_posting"] = max_posting
    return dict(stats)


def _lookup(db: sqlite3.Connection, table: str, key: str, limit: int) -> list[str]:
    cur = db.execute(f"SELECT id FROM {table} WHERE key=? LIMIT ?", (key, limit))
    return [r[0] for r in cur.fetchall()]


def generate_candidates_for_s1(
    db: sqlite3.Connection,
    name: str,
    addr: str,
    country: str,
    *,
    max_per_route: int = 40,
    max_candidates: int = 50,
) -> list[str]:
    nn, aa = norm_name(name), norm_addr(addr)
    scores: dict[str, float] = defaultdict(float)
    if nn:
        for eid in _lookup(db, "by_name", nn, max_per_route):
            scores[eid] += 5.0
        pref = "".join(nn.split())[:3]
        if pref:
            for eid in _lookup(db, "by_prefix", pref, max_per_route):
                scores[eid] += 1.0
        for tok in sig_tokens(nn):
            for eid in _lookup(db, "by_token", tok, max_per_route):
                scores[eid] += 1.5
    for num in nums(aa):
        for eid in _lookup(db, "by_num", num, max_per_route):
            scores[eid] += 2.0
    if not scores:
        return []
    # soft country preference via join
    ids = list(scores.keys())
    # fetch countries in chunks
    soft = {}
    for i in range(0, len(ids), 500):
        chunk = ids[i : i + 500]
        q = ",".join("?" * len(chunk))
        for eid, c in db.execute(f"SELECT id, country FROM targets WHERE id IN ({q})", chunk):
            if country and c and nfkc_casefold(country) != nfkc_casefold(c):
                soft[eid] = 0.5
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


def pair_score(a: dict, b: dict) -> float:
    """Deterministic interpretable score in [0,1]."""
    na, nb = a.get("name_norm") or norm_name(a["business_name"]), b.get("name_norm") or norm_name(b["business_name"])
    aa, ab = a.get("addr_norm") or norm_addr(a["business_address"]), b.get("addr_norm") or norm_addr(b["business_address"])
    name_s = fuzz.token_sort_ratio(na, nb) / 100.0 if (na or nb) else 0.0
    addr_s = fuzz.token_sort_ratio(aa, ab) / 100.0 if (aa or ab) else 0.0
    exact = 1.0 if na and nb and na == nb else 0.0
    ta, tb = set(sig_tokens(na)), set(sig_tokens(nb))
    jac = (len(ta & tb) / len(ta | tb)) if (ta or tb) else 0.0
    na_nums, nb_nums = set(nums(aa)), set(nums(ab))
    num = (len(na_nums & nb_nums) / len(na_nums | nb_nums)) if (na_nums and nb_nums) else 0.0
    c_eq = 1.0 if a.get("country") and b.get("country") and nfkc_casefold(a["country"]) == nfkc_casefold(b["country"]) else 0.0
    return float(np.clip(0.35 * name_s + 0.15 * jac + 0.20 * addr_s + 0.15 * num + 0.10 * exact + 0.05 * c_eq, 0, 1))


def exact_match(a: dict, b: dict) -> bool:
    na, nb = a.get("name_norm") or norm_name(a["business_name"]), b.get("name_norm") or norm_name(b["business_name"])
    aa, ab = a.get("addr_norm") or norm_addr(a["business_address"]), b.get("addr_norm") or norm_addr(b["business_address"])
    if not na or not nb or na != nb:
        return False
    if aa and ab:
        return aa == ab
    # name exact + country equal if both present
    ca, cb = a.get("country", ""), b.get("country", "")
    return bool(ca and cb and nfkc_casefold(ca) == nfkc_casefold(cb))


def append_exp(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "experiment_id", "timestamp", "hypothesis", "validation_version", "blocking_config",
        "feature_config", "model", "decision_rule", "threshold", "candidate_recall",
        "precision", "recall", "macro_f0.5", "singleton_accuracy", "avg_candidates",
        "runtime", "notes", "data_provenance", "fold",
    ]
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


def main() -> None:
    p = argparse.ArgumentParser(description="Official bounded ER experiments")
    p.add_argument("--dataset-root", type=Path, default=REPO / "student_resource" / "dataset")
    p.add_argument("--reports-root", type=Path, default=REPO / "reports" / "official")
    p.add_argument("--work-dir", type=Path, default=REPO / "artifacts" / "official_index")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-s1", type=int, default=5000, help="Bounded real S1 count from select fold for Stage2/3")
    p.add_argument("--max-candidates", type=int, default=50)
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

    # ----- Stage 1: ensure audit exists -----
    audit_path = reports / "official_audit.json"
    public_audit = REPO / "reports" / "eda" / "official_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        print("== Stage 1: using existing split-local audit ==", flush=True)
    elif public_audit.exists():
        audit = json.loads(public_audit.read_text())
        audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
        print("== Stage 1: reused reports/eda/official_audit.json ==", flush=True)
    else:
        from run_official_audit import run as run_audit

        print("== Stage 1: streaming audit ==", flush=True)
        audit = run_audit(args.dataset_root, public_audit, args.work_dir)
        audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    print(json.dumps({"empty_baseline": audit["empty_baseline"], "singleton_rate": audit["ground_truth"]["singleton_rate"]}, indent=2))

    # ----- Freeze / load split (ids in sqlite to avoid huge JSON) -----
    split_db = args.work_dir / f"{split_version}_folds.sqlite"
    split_meta_path = REPO / "experiments" / "splits" / f"{split_version}_meta.json"
    if not split_db.exists():
        print("== freezing component-disjoint fit/select/assess split ==", flush=True)
        meta, fit, select, assess, gt_card, countries = freeze_official_split(
            args.dataset_root, seed=args.seed
        )
        db = sqlite3.connect(split_db)
        db.execute("CREATE TABLE fold (id TEXT PRIMARY KEY, fold TEXT, card INTEGER, country TEXT) WITHOUT ROWID")
        rows = []
        for sid in fit:
            rows.append((sid, "fit", gt_card.get(sid, 0), countries.get(sid, "")))
        for sid in select:
            rows.append((sid, "select", gt_card.get(sid, 0), countries.get(sid, "")))
        for sid in assess:
            rows.append((sid, "assess", gt_card.get(sid, 0), countries.get(sid, "")))
        db.executemany("INSERT INTO fold VALUES (?,?,?,?)", rows)
        db.commit()
        # meta without full id lists
        slim = {k: v for k, v in meta.items() if k not in {"fit_ids", "select_ids", "assess_ids"}}
        slim["gt_hash"] = gt_hash
        slim["provenance"] = provenance
        split_meta_path.parent.mkdir(parents=True, exist_ok=True)
        split_meta_path.write_text(json.dumps(slim, indent=2) + "\n")
        db.close()
        print(json.dumps(slim, indent=2))
    else:
        slim = json.loads(split_meta_path.read_text()) if split_meta_path.exists() else {}
        print(f"== using frozen split {split_version} ==", flush=True)

    # Sample bounded select S1s for Stage 2/3 (still REAL ids from frozen select fold)
    fold_db = sqlite3.connect(split_db)
    select_rows = list(fold_db.execute("SELECT id, card, country FROM fold WHERE fold='select' ORDER BY id"))
    rng = random.Random(args.seed)
    rng.shuffle(select_rows)
    # stratify lightly: keep singleton ratio similar
    n_eval = min(args.eval_s1, len(select_rows))
    eval_rows = select_rows[:n_eval]
    eval_ids = [r[0] for r in eval_rows]
    print(f"== Stage 2/3 eval S1s: {len(eval_ids)} from select fold (full S2/S3 universe) ==", flush=True)

    # Load GT only for eval ids
    eval_gt: dict[str, set[str]] = {i: set() for i in eval_ids}
    needed = set(eval_ids)
    for row in iter_tsv(args.dataset_root / "train" / "train_ground_truth.tsv", GT_H):
        sid = row["source1_entity_id"]
        if sid in needed:
            eval_gt[sid] = {x for x in row["matched_entity_ids"].split(",") if x} if row["matched_entity_ids"] else set()

    # Load S1 records for eval
    s1_map = {}
    for row in iter_tsv(args.dataset_root / "train" / "train_source1.tsv", SOURCE_H):
        if row["entity_id"] in needed:
            s1_map[row["entity_id"]] = row

    # ----- Build / open target index -----
    index_db_path = args.work_dir / "targets_index.sqlite"
    if not args.skip_index_build and not index_db_path.exists():
        print("== building full S2/S3 inverted index (disk) ==", flush=True)
        t0 = time.time()
        idx_stats = build_target_index(args.dataset_root, index_db_path)
        idx_stats["runtime_sec"] = time.time() - t0
        (reports / "index_stats.json").write_text(json.dumps(idx_stats, indent=2) + "\n")
        print(json.dumps(idx_stats, indent=2))
    idb = sqlite3.connect(f"file:{index_db_path}?mode=ro", uri=True)
    n_targets = idb.execute("SELECT COUNT(*) FROM targets").fetchone()[0]

    # ----- Stage 2: candidates -----
    print("== Stage 2: candidate generation ==", flush=True)
    t0 = time.time()
    candidates: dict[str, list[str]] = {}
    for i, sid in enumerate(eval_ids):
        rec = s1_map[sid]
        candidates[sid] = generate_candidates_for_s1(
            idb, rec["business_name"], rec["business_address"], rec["country"],
            max_candidates=args.max_candidates,
        )
        if (i + 1) % 500 == 0:
            print(f"  candidates {i+1}/{len(eval_ids)}", flush=True)
    block_time = time.time() - t0
    block_diag = candidate_diagnostics(candidates, eval_gt, n_targets)
    block_diag["runtime_sec"] = block_time
    block_diag["n_eval_s1"] = len(eval_ids)
    block_diag["n_target_universe"] = n_targets
    block_diag["provenance"] = provenance
    block_diag["split_version"] = split_version
    block_diag["fold"] = "select_bounded"
    (reports / "blocking_baseline.json").write_text(json.dumps(block_diag, indent=2) + "\n")
    (reports / "blocking_baseline.md").write_text(
        "\n".join([
            "# Official blocking baseline (bounded select S1, full S2/S3)",
            "",
            f"**Provenance:** `{provenance}`",
            f"**Split:** `{split_version}` fold=select_bounded n={len(eval_ids)}",
            "",
            f"- micro candidate recall: {block_diag['micro_candidate_recall']}",
            f"- S2 recall: {block_diag['s2_recall']}; S3 recall: {block_diag['s3_recall']}",
            f"- full-set coverage: {block_diag['full_set_coverage']}",
            f"- oracle macro F0.5: {block_diag['oracle_macro_F0.5']}",
            f"- avg/p50/p95/p99/max candidates: {block_diag['avg_candidates']:.2f}/{block_diag['median_candidates']}/{block_diag['p95_candidates']}/{block_diag['p99_candidates']}/{block_diag['max_candidates']}",
            f"- zero-candidate frac: {block_diag['zero_candidate_frac']}",
            f"- reduction ratio: {block_diag['reduction_ratio']}",
            f"- runtime_sec: {block_time:.2f}",
            f"- true matches missed: {block_diag['true_matches_missed']}",
            "",
        ]),
        encoding="utf-8",
    )
    print(json.dumps(block_diag, indent=2))

    # ----- Stage 3: baselines -----
    print("== Stage 3: end-to-end baselines ==", flush=True)
    exp_csv = reports / "experiments.csv"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def log(eid, hyp, model, metrics, **kw):
        append_exp(exp_csv, {
            "experiment_id": eid,
            "timestamp": ts,
            "hypothesis": hyp,
            "validation_version": split_version,
            "blocking_config": kw.get("blocking_config", "name+prefix+token+num"),
            "feature_config": kw.get("feature_config", ""),
            "model": model,
            "decision_rule": kw.get("decision_rule", ""),
            "threshold": kw.get("threshold", ""),
            "candidate_recall": block_diag["micro_candidate_recall"],
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "macro_f0.5": metrics.get("macro_F0.5"),
            "singleton_accuracy": metrics.get("singleton_accuracy"),
            "avg_candidates": block_diag["avg_candidates"],
            "runtime": kw.get("runtime", ""),
            "notes": kw.get("notes", ""),
            "data_provenance": provenance,
            "fold": "select_bounded",
        })
        print(f"{eid}: macro_F0.5={metrics['macro_F0.5']:.4f} P={metrics.get('precision')} R={metrics.get('recall')} sing={metrics.get('singleton_accuracy')}", flush=True)

    # B0 empty
    t0 = time.time()
    pred_empty = {i: set() for i in eval_ids}
    m0 = score_predictions(eval_gt, pred_empty)
    log("B0_empty", "All-empty abstention on bounded select S1", "empty", m0, runtime=time.time()-t0, decision_rule="always_empty")

    # B1 exact on full universe via name index (not limited to candidates) — still official
    t0 = time.time()
    pred_exact = {i: set() for i in eval_ids}
    for sid in eval_ids:
        a = s1_map[sid]
        nn = norm_name(a["business_name"])
        if not nn:
            continue
        # candidates from exact name key only
        ids = _lookup(idb, "by_name", nn, 100)
        trecs = load_records(idb, ids)
        for tid, b in trecs.items():
            if exact_match({**a, "name_norm": nn, "addr_norm": norm_addr(a["business_address"])}, b):
                pred_exact[sid].add(tid)
    m1 = score_predictions(eval_gt, pred_exact)
    log("B1_exact_norm", "Normalized exact name+(addr|country) via name index", "exact_normalized", m1, runtime=time.time()-t0, blocking_config="exact_name_index", feature_config="name_norm+addr_norm")

    # B2 exact restricted to Stage2 candidates
    t0 = time.time()
    pred_exact_c = {i: set() for i in eval_ids}
    for sid in eval_ids:
        a = s1_map[sid]
        cands = candidates[sid]
        trecs = load_records(idb, cands)
        aa = {**a, "name_norm": norm_name(a["business_name"]), "addr_norm": norm_addr(a["business_address"])}
        for tid, b in trecs.items():
            if exact_match(aa, b):
                pred_exact_c[sid].add(tid)
    m2 = score_predictions(eval_gt, pred_exact_c)
    log("B2_exact_on_candidates", "Exact normalized on Stage2 candidates", "exact_on_candidates", m2, runtime=time.time()-t0)

    # M0 weighted similarity on candidates + threshold search on SAME select fold
    # NOTE: this fold is selection fold — report as selection score, not untouched assess.
    t0 = time.time()
    # collect scores
    all_scores = []  # (sid, tid, score)
    for sid in eval_ids:
        a = s1_map[sid]
        aa = {**a, "name_norm": norm_name(a["business_name"]), "addr_norm": norm_addr(a["business_address"])}
        trecs = load_records(idb, candidates[sid])
        for tid, b in trecs.items():
            all_scores.append((sid, tid, pair_score(aa, b)))

    best_t, best_m = 0.5, None
    grid = np.linspace(0.35, 0.95, 25)
    for thr in grid:
        pred = {i: set() for i in eval_ids}
        for sid, tid, sc in all_scores:
            if sc >= thr:
                pred[sid].add(tid)
        m = score_predictions(eval_gt, pred)
        if best_m is None or m["macro_F0.5"] > best_m["macro_F0.5"]:
            best_t, best_m = float(thr), m
            best_pred = pred
    log(
        "M0_weighted_select",
        "Weighted string features on candidates; threshold tuned on select fold (NOT untouched assess)",
        "weighted_similarity",
        best_m,
        runtime=time.time() - t0,
        threshold=best_t,
        decision_rule="global_threshold_on_select",
        feature_config="token_sort+jaccard+num+exact+country",
        notes="selection-fold score; do not claim as untouched assessment",
    )

    # Error analysis on best_pred
    err = {"singleton_fp": 0, "fp": 0, "fn_blocking": 0, "fn_threshold": 0, "examples": []}
    for sid, truth in eval_gt.items():
        pred = best_pred.get(sid, set())
        cset = set(candidates.get(sid, []))
        if not truth and pred:
            err["singleton_fp"] += 1
            if len(err["examples"]) < 8:
                err["examples"].append({"type": "singleton_fp", "s1": sid, "pred": sorted(pred)[:5], "name": s1_map[sid]["business_name"]})
        err["fp"] += len(pred - truth)
        for m in truth - pred:
            if m not in cset:
                err["fn_blocking"] += 1
                if len(err["examples"]) < 20:
                    err["examples"].append({"type": "fn_blocking", "s1": sid, "miss": m, "name": s1_map[sid]["business_name"]})
            else:
                err["fn_threshold"] += 1
                if len(err["examples"]) < 20:
                    err["examples"].append({"type": "fn_threshold", "s1": sid, "miss": m, "name": s1_map[sid]["business_name"]})
    (reports / "error_analysis_select.json").write_text(json.dumps(err, indent=2) + "\n")

    summary = {
        "provenance": provenance,
        "split_version": split_version,
        "fold": "select_bounded",
        "n_eval_s1": len(eval_ids),
        "n_target_universe": n_targets,
        "stage1_empty_baseline_full_train": audit["empty_baseline"],
        "stage2_blocking": block_diag,
        "stage3": {
            "B0_empty": m0,
            "B1_exact_norm": m1,
            "B2_exact_on_candidates": m2,
            "M0_weighted_select": {**best_m, "threshold": best_t},
        },
        "error_counts": {k: v for k, v in err.items() if k != "examples"},
        "runtime_total_sec": time.time() - t_all,
        "next_experiments": [
            "Raise token/num route budgets or add char n-gram TF-IDF ANN for fn_blocking cluster",
            "Train logistic on fit-fold hard negatives; tune threshold on select; score untouched assess",
            "Add name↔address contradiction gate to cut singleton_fp without losing recall",
        ],
    }
    (reports / "stage123_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("== DONE ==", json.dumps({"reports": str(reports), "macro_best_select": best_m["macro_F0.5"], "threshold": best_t}, indent=2))
    idb.close()
    fold_db.close()


if __name__ == "__main__":
    main()
