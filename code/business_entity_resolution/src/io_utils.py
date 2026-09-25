"""TSV I/O helpers — always tab-separated, string dtypes, no NA coercion."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def write_tsv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, encoding="utf-8")


def read_sources(data_dir: str | Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    s1 = read_tsv(data_dir / f"{split}_source1.tsv")
    s2 = read_tsv(data_dir / f"{split}_source2.tsv")
    s3 = read_tsv(data_dir / f"{split}_source3.tsv")
    for name, df in (("s1", s1), ("s2", s2), ("s3", s3)):
        missing = [c for c in SOURCE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{split} {name} missing columns: {missing}")
    return s1, s2, s3


def read_ground_truth(path: str | Path) -> dict[str, set[str]]:
    df = read_tsv(path)
    out: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        s1 = row["source1_entity_id"]
        raw = row.get("matched_entity_ids", "")
        ids = {x for x in str(raw).split(",") if x} if str(raw).strip() else set()
        out[s1] = ids
    return out


def write_id_list_tsv(
    path: str | Path,
    s1_ids: Iterable[str],
    mapping: dict[str, list[str]],
    id_col: str,
    list_col: str,
) -> None:
    rows = []
    for s1 in s1_ids:
        ids = mapping.get(s1, [])
        # stable, deduped, no spaces
        seen = set()
        cleaned = []
        for i in ids:
            if i and i not in seen:
                seen.add(i)
                cleaned.append(i)
        rows.append({id_col: s1, list_col: ",".join(cleaned)})
    write_tsv(pd.DataFrame(rows, columns=[id_col, list_col]), path)
