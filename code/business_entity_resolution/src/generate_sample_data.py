"""DEV-ONLY tiny stub generator. Do NOT use for competition research or scoring.

Prefer the official dumps via ``bash scripts/download_dataset.sh`` (release dataset-v1).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

from io_utils import write_tsv

LEGAL = ["Inc", "Corp", "LLC", "Ltd", "Pvt Ltd", "Limited", "SAS", "SARL", ""]
STREETS = [
    "Main Street",
    "Oak Road",
    "Park Avenue",
    "High Street",
    "MG Road",
    "Rue de Rivoli",
    "Rue Victor Hugo",
]


def mut_name(name: str, rng: random.Random) -> str:
    out = name
    if rng.random() < 0.4:
        out = (
            out.replace("Corporation", "Corp")
            .replace("Limited", "Ltd")
            .replace("Private", "Pvt")
        )
    if rng.random() < 0.2:
        out = out.replace(" and ", " & ")
    if rng.random() < 0.15 and len(out) > 4:
        i = rng.randrange(len(out) - 1)
        out = out[:i] + out[i + 1] + out[i] + out[i + 2 :]
    return out


def mut_addr(addr: str, rng: random.Random) -> str:
    out = addr.replace("Street", "St").replace("Road", "Rd").replace("Avenue", "Ave")
    if rng.random() < 0.2:
        out = out + " Near Landmark"
    return out


def make_entity(i: int, country: str, rng: random.Random) -> dict[str, str]:
    base = f"Acme Business {i} {rng.choice(['Solutions', 'Trading', 'Services', 'Group'])}"
    name = f"{base} {rng.choice(LEGAL)}".strip()
    street = rng.choice(STREETS)
    num = rng.randint(1, 999)
    if country == "US":
        addr = f"{num} {street}, City {i % 50}, {(i * 7) % 90000 + 10000}"
    elif country == "India":
        addr = f"{num}, {street}, Area {i % 40}, {(i * 13) % 900000 + 100000}"
    else:
        # France or any future open-set country label
        addr = f"{num} {street}, {75000 + (i % 20)}, Ville {i % 30}"
    return {"name": name, "address": addr, "country": country}


def _add_decoys(
    rng: random.Random,
    country: str,
    s2: list[dict],
    s3: list[dict],
    counters: list[int],
) -> None:
    for _ in range(rng.randint(0, 2)):
        decoy = make_entity(rng.randint(10000, 40000), country, rng)
        if rng.random() < 0.5:
            counters[0] += 1
            s2.append(
                {
                    "entity_id": f"S2-{counters[0]:05d}",
                    "business_name": decoy["name"],
                    "business_address": decoy["address"],
                    "country": country,
                }
            )
        else:
            counters[1] += 1
            s3.append(
                {
                    "entity_id": f"S3-{counters[1]:05d}",
                    "business_name": decoy["name"],
                    "business_address": decoy["address"],
                    "country": country,
                }
            )


def generate_split(
    n: int,
    countries: list[str],
    seed: int,
    with_gt: bool,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    rng = random.Random(seed)
    s1, s2, s3, gt = [], [], [], []
    counters = [0, 0]  # s2, s3
    for i in range(1, n + 1):
        country = countries[(i - 1) % len(countries)]
        ent = make_entity(i, country, rng)
        s1_id = f"S1-{i:05d}"
        s1.append(
            {
                "entity_id": s1_id,
                "business_name": ent["name"],
                "business_address": ent["address"],
                "country": country,
            }
        )
        matches: list[str] = []
        if rng.random() < 0.72:
            if rng.random() < 0.85:
                counters[0] += 1
                sid = f"S2-{counters[0]:05d}"
                s2.append(
                    {
                        "entity_id": sid,
                        "business_name": mut_name(ent["name"], rng),
                        "business_address": mut_addr(ent["address"], rng),
                        "country": country,
                    }
                )
                matches.append(sid)
            if rng.random() < 0.55:
                counters[1] += 1
                sid = f"S3-{counters[1]:05d}"
                s3.append(
                    {
                        "entity_id": sid,
                        "business_name": mut_name(ent["name"], rng),
                        "business_address": mut_addr(ent["address"], rng),
                        "country": country,
                    }
                )
                matches.append(sid)
        _add_decoys(rng, country, s2, s3, counters)
        if with_gt:
            gt.append({"source1_entity_id": s1_id, "matched_entity_ids": ",".join(matches)})
    return s1, s2, s3, gt


def generate(out_dir: Path, n_train: int = 120, n_test: int = 80, seed: int = 42) -> None:
    out_dir = Path(out_dir)
    existing = out_dir / "train" / "train_source1.tsv"
    if existing.is_file():
        with existing.open(encoding="utf-8") as f:
            n_lines = sum(1 for _ in f)
        if n_lines >= 100_000:
            raise RuntimeError(
                f"Refusing to overwrite official dataset at {existing} ({n_lines} lines). "
                "Use bash scripts/download_dataset.sh for real data."
            )
    train_dir = out_dir / "train"
    test_dir = out_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    s1, s2, s3, gt = generate_split(n_train, ["US", "India"], seed, with_gt=True)
    write_tsv(pd.DataFrame(s1), train_dir / "train_source1.tsv")
    write_tsv(pd.DataFrame(s2), train_dir / "train_source2.tsv")
    write_tsv(pd.DataFrame(s3), train_dir / "train_source3.tsv")
    write_tsv(pd.DataFrame(gt), train_dir / "train_ground_truth.tsv")

    # Test includes France (unseen in train) to exercise open-set country handling.
    s1, s2, s3, _ = generate_split(n_test, ["US", "India", "France"], seed + 1, with_gt=False)
    write_tsv(pd.DataFrame(s1), test_dir / "test_source1.tsv")
    write_tsv(pd.DataFrame(s2), test_dir / "test_source2.tsv")
    write_tsv(pd.DataFrame(s3), test_dir / "test_source3.tsv")
    print(f"Wrote synthetic dataset to {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="dataset")
    p.add_argument("--n-train", type=int, default=120)
    p.add_argument("--n-test", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    generate(Path(args.out_dir), args.n_train, args.n_test, args.seed)


if __name__ == "__main__":
    main()
