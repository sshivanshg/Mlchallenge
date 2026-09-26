"""Audit normalization damage to non-Latin (and mixed-script) business names.

Quantifies how often norm_name / compact_alnum collapse or empty non-Latin strings
on the official train S1 selection slice — relevant to Latin→non-Latin retrieval misses.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import sys
import unicodedata
from collections import Counter
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from build_aux_index_v5b import compact_alnum
from frozen_v4.policy import norm_addr, norm_name
from official_core import SOURCE_H, iter_tsv

REPO = SRC.parents[2]
WD = REPO / "artifacts" / "official_index"
LATIN_RE = re.compile(r"[A-Za-z]")
# letters that are not Basic Latin / Latin-1 supplement letters roughly → non-Latin script
NONLATIN_LETTER = re.compile(r"[^\u0000-\u024F\s0-9\W_]", re.UNICODE)


def script_tag(text: str) -> str:
    has_latin = bool(LATIN_RE.search(text or ""))
    has_non = bool(NONLATIN_LETTER.search(text or ""))
    if has_latin and has_non:
        return "mixed"
    if has_non:
        return "nonlatin"
    if has_latin:
        return "latin"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-s1", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    fold = sqlite3.connect(f"file:{WD / 'official_70bc1d8a16c6_folds.sqlite'}?mode=ro", uri=True)
    rows = [r[0] for r in fold.execute("SELECT id FROM fold WHERE fold='select' ORDER BY id")]
    random.Random(args.seed).shuffle(rows)
    ids = rows[: args.n_s1]
    need = set(ids)

    gt = {i: set() for i in ids}
    for row in iter_tsv(REPO / "student_resource" / "dataset" / "train" / "train_ground_truth.tsv",
                        ["source1_entity_id", "matched_entity_ids"]):
        if row["source1_entity_id"] in need and row["matched_entity_ids"]:
            gt[row["source1_entity_id"]] = set(row["matched_entity_ids"].split(","))

    # Load S1 + all truth targets' raw names from sources (stream)
    s1 = {}
    for row in iter_tsv(REPO / "student_resource" / "dataset" / "train" / "train_source1.tsv", SOURCE_H):
        if row["entity_id"] in need:
            s1[row["entity_id"]] = row

    need_t = set().union(*(gt[i] for i in ids))
    targets = {}
    for src in (2, 3):
        path = REPO / "student_resource" / "dataset" / "train" / f"train_source{src}.tsv"
        for row in iter_tsv(path, SOURCE_H):
            if row["entity_id"] in need_t:
                targets[row["entity_id"]] = row
                if len(targets) == len(need_t):
                    break
        if len(targets) == len(need_t):
            break

    s1_script = Counter()
    emptied_name = 0
    emptied_compact = 0
    shortened_gt = 0  # truth target name becomes empty after norm while raw non-empty
    pair_script = Counter()
    examples = []
    for sid in ids:
        a = s1[sid]
        tag = script_tag(a["business_name"])
        s1_script[tag] += 1
        nn = norm_name(a["business_name"])
        if a["business_name"].strip() and not nn.strip():
            emptied_name += 1
            if len(examples) < 20:
                examples.append({"kind": "s1_name_emptied", "id": sid, "raw": a["business_name"]})
        if a["business_name"].strip() and not compact_alnum(nn):
            emptied_compact += 1
        for tid in gt[sid]:
            b = targets.get(tid)
            if not b:
                continue
            pair_script[f"{tag}->{script_tag(b['business_name'])}"] += 1
            bn = norm_name(b["business_name"])
            if b["business_name"].strip() and not bn.strip():
                shortened_gt += 1
                if len(examples) < 40:
                    examples.append({"kind": "gt_name_emptied", "s1": sid, "t": tid, "raw": b["business_name"]})

    report = {
        "n_s1": len(ids),
        "seed": args.seed,
        "s1_script_counts": dict(s1_script),
        "s1_norm_name_emptied": emptied_name,
        "s1_compact_alnum_emptied": emptied_compact,
        "gt_target_norm_name_emptied": shortened_gt,
        "truth_pair_script_flows": dict(pair_script),
        "examples": examples,
        "note": "compact_alnum strips non [a-z0-9]; non-Latin names yield empty ngrams — expected v5b ngram miss mode",
    }
    text = json.dumps(report, indent=2) + "\n"
    print(text, flush=True)
    out = args.out or (REPO / "reports" / "official" / "official_70bc1d8a16c6" / "retrieval_exp2" / "norm_nonlatin_audit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)


if __name__ == "__main__":
    main()
