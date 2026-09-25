"""TSV I/O and data hashing (tab-separated, string dtypes)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def write_tsv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, encoding="utf-8")


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_sources(data_dir: str | Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    frames = []
    for i in (1, 2, 3):
        df = read_tsv(data_dir / f"{split}_source{i}.tsv")
        missing = [c for c in SOURCE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{split} source{i} missing {missing}")
        frames.append(df)
    return frames[0], frames[1], frames[2]


def read_ground_truth(path: str | Path) -> dict[str, set[str]]:
    df = read_tsv(path)
    out: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        raw = str(row.get("matched_entity_ids", ""))
        ids = {x for x in raw.split(",") if x} if raw.strip() else set()
        out[str(row["source1_entity_id"])] = ids
    return out


def dataset_hashes(train_dir: Path, test_dir: Path | None = None) -> dict[str, str]:
    files = {
        "train_source1": train_dir / "train_source1.tsv",
        "train_source2": train_dir / "train_source2.tsv",
        "train_source3": train_dir / "train_source3.tsv",
        "train_ground_truth": train_dir / "train_ground_truth.tsv",
    }
    if test_dir is not None:
        for i in (1, 2, 3):
            files[f"test_source{i}"] = test_dir / f"test_source{i}.tsv"
    return {k: file_sha256(v) for k, v in files.items() if v.is_file()}


def write_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
