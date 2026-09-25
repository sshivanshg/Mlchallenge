"""Harder synthetic fixture for isolated unit tests only (NOT competition data)."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

# Local import when run as script
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.load import write_tsv

GENERIC_NAMES = [
    "Global Trading",
    "Premier Services",
    "National Solutions",
    "United Business Group",
    "Metro Consultants",
]
UNIQUE_STEMS = [
    "Zephyr Quokka",
    "Indigo Narwhal",
    "Copper Finch",
    "Velvet Orchid",
    "Marble Badger",
    "Silver Maple",
    "Amber Otter",
    "Crimson Lotus",
]


def mut_name(name: str, rng: random.Random) -> str:
    out = name
    reps = [
        ("Corporation", "Corp"),
        ("Incorporated", "Inc"),
        ("Limited", "Ltd"),
        ("Private", "Pvt"),
        (" and ", " & "),
    ]
    for a, b in reps:
        if a in out and rng.random() < 0.5:
            out = out.replace(a, b)
    if rng.random() < 0.25 and len(out) > 6:
        i = rng.randrange(1, len(out) - 1)
        out = out[:i] + out[i + 1 :]  # deletion typo
    if rng.random() < 0.15:
        toks = out.split()
        if len(toks) > 2:
            j = rng.randrange(len(toks) - 1)
            toks[j], toks[j + 1] = toks[j + 1], toks[j]
            out = " ".join(toks)
    return out


def mut_addr(addr: str, rng: random.Random) -> str:
    out = (
        addr.replace("Street", "St")
        .replace("Road", "Rd")
        .replace("Avenue", "Ave")
        .replace("Boulevard", "Blvd")
    )
    if rng.random() < 0.3:
        out = out + rng.choice(["", " Near Landmark", " Opp Park"])
    if rng.random() < 0.2:
        # drop trailing postal-like token
        parts = out.split(",")
        if len(parts) > 1:
            out = ",".join(parts[:-1])
    return out


def addr_for(i: int, country: str, rng: random.Random) -> str:
    streets = {
        "US": ["Main Street", "Oak Road", "Park Avenue", "Broadway"],
        "India": ["MG Road", "Nehru Street", "Ring Road", "Lake Avenue"],
        "France": ["Rue de Rivoli", "Rue Victor Hugo", "Boulevard Haussmann", "Avenue Victor"],
    }
    street = rng.choice(streets.get(country, streets["US"]))
    num = 10 + (i * 17) % 900
    if country == "US":
        return f"{num} {street}, City {i % 40}, {10000 + (i * 97) % 89999}"
    if country == "India":
        return f"{num}, {street}, Area {i % 30}, {110000 + (i * 91) % 700000}"
    return f"{num} {street}, {75000 + (i % 20)}, Ville {i % 25}"


def generate(out_dir: Path, n_train: int = 400, n_test: int = 200, seed: int = 7) -> None:
    """Unit-test-only stub writer. Forbidden from competition dataset paths."""
    from data.provenance import ProvenanceError, assert_not_synthetic_output_target

    out_dir = Path(out_dir)
    assert_not_synthetic_output_target(out_dir)
    # Extra: only allow under a tests/ tree
    if "tests" not in out_dir.resolve().parts:
        raise ProvenanceError(
            f"Synthetic generator may only write under a tests/ directory, got {out_dir}"
        )

    rng = random.Random(seed)
    train_dir, test_dir = out_dir / "train", out_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    def build(n: int, countries: list[str], seed_local: int, with_gt: bool):
        rng = random.Random(seed_local)
        s1, s2, s3, gt = [], [], [], []
        c2 = c3 = 0
        # Shared address pool to create address collisions
        shared_addrs = [addr_for(9000 + k, countries[k % len(countries)], rng) for k in range(12)]

        for i in range(1, n + 1):
            country = countries[(i - 1) % len(countries)]
            if rng.random() < 0.25:
                stem = rng.choice(GENERIC_NAMES) + f" {rng.choice(['LLC', 'Ltd', 'Inc', 'Corp'])}"
            else:
                stem = (
                    f"{UNIQUE_STEMS[i % len(UNIQUE_STEMS)]} {i} "
                    f"{rng.choice(['Solutions', 'Trading', 'Services', 'Holdings'])} "
                    f"{rng.choice(['Inc', 'Ltd', 'LLC', 'Corp', 'Pvt Ltd'])}"
                )
            if rng.random() < 0.15:
                address = shared_addrs[i % len(shared_addrs)]
            else:
                address = addr_for(i, country, rng)
            s1_id = f"S1-{i:05d}"
            s1.append(
                {
                    "entity_id": s1_id,
                    "business_name": stem,
                    "business_address": address,
                    "country": country,
                }
            )
            matches = []
            # ~35% singletons (harder abstention problem)
            if rng.random() < 0.65:
                if rng.random() < 0.85:
                    c2 += 1
                    sid = f"S2-{c2:05d}"
                    s2.append(
                        {
                            "entity_id": sid,
                            "business_name": mut_name(stem, rng),
                            "business_address": mut_addr(address, rng),
                            "country": country,
                        }
                    )
                    matches.append(sid)
                if rng.random() < 0.5:
                    c3 += 1
                    sid = f"S3-{c3:05d}"
                    s3.append(
                        {
                            "entity_id": sid,
                            "business_name": mut_name(stem, rng),
                            "business_address": mut_addr(address, rng),
                            "country": country,
                        }
                    )
                    matches.append(sid)
            # Hard distractors: similar generic name, shared address, near-typo
            for _ in range(rng.randint(1, 3)):
                mode = rng.choice(["generic", "shared_addr", "near_name"])
                if mode == "generic":
                    dname = rng.choice(GENERIC_NAMES) + f" {rng.choice(['LLC', 'Inc', 'Ltd'])}"
                    daddr = addr_for(rng.randint(20000, 30000), country, rng)
                elif mode == "shared_addr":
                    dname = f"Other Co {rng.randint(1,999)} Ltd"
                    daddr = address if rng.random() < 0.5 else shared_addrs[rng.randrange(len(shared_addrs))]
                else:
                    dname = mut_name(stem, rng)
                    daddr = addr_for(rng.randint(40000, 50000), country, rng)
                if rng.random() < 0.5:
                    c2 += 1
                    s2.append(
                        {
                            "entity_id": f"S2-{c2:05d}",
                            "business_name": dname,
                            "business_address": daddr,
                            "country": country,
                        }
                    )
                else:
                    c3 += 1
                    s3.append(
                        {
                            "entity_id": f"S3-{c3:05d}",
                            "business_name": dname,
                            "business_address": daddr,
                            "country": country,
                        }
                    )
            if with_gt:
                gt.append({"source1_entity_id": s1_id, "matched_entity_ids": ",".join(matches)})
        return s1, s2, s3, gt

    s1, s2, s3, gt = build(n_train, ["US", "India"], seed, True)
    write_tsv(pd.DataFrame(s1), train_dir / "train_source1.tsv")
    write_tsv(pd.DataFrame(s2), train_dir / "train_source2.tsv")
    write_tsv(pd.DataFrame(s3), train_dir / "train_source3.tsv")
    write_tsv(pd.DataFrame(gt), train_dir / "train_ground_truth.tsv")

    s1, s2, s3, _ = build(n_test, ["US", "India", "France"], seed + 1, False)
    write_tsv(pd.DataFrame(s1), test_dir / "test_source1.tsv")
    write_tsv(pd.DataFrame(s2), test_dir / "test_source2.tsv")
    write_tsv(pd.DataFrame(s3), test_dir / "test_source3.tsv")

    (out_dir / "PROVENANCE.txt").write_text(
        "unit_test_synthetic_stub\n"
        "NOT official AWS challenge data.\n"
        "Allowed only under tests/ for isolated unit tests.\n"
    )
    print(f"Wrote unit-test synthetic stub to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-train", type=int, default=400)
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    generate(args.out_dir, args.n_train, args.n_test, args.seed)
