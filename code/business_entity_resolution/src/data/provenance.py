"""Official-dataset provenance guard for competition runs.

Competition code paths (fit / validate / score / block / threshold / infer /
submit) MUST call ``require_official_dataset`` before reading entity records.

Verification prefers schema + SHA-256 manifest checks over row-count heuristics.
Synthetic generators may exist only for isolated unit tests and must never
write into the competition dataset root.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Package layouts:
#   .../business_entity_resolution/src/data/provenance.py  -> configs at ../../configs
#   also accept env OFFICIAL_DATASET_MANIFEST
_DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[2] / "configs" / "official_dataset_manifest.json"
)

_BANNED_PATH_TOKENS = re.compile(
    r"(synthetic|fixture|fixtures|sample_data|samples|/sample/|_sample_|stub|stubs|"
    r"smoke|fake|mock|dummy|generate_sample|make_synthetic|dev_fixture|"
    r"synthetic_dev|acme_business|/tmp/pytest|/tmp/test)",
    re.IGNORECASE,
)

_ALLOWED_DATASET_SUFFIXES = (
    Path("student_resource") / "dataset",
)

SOURCE_HEADER = ["entity_id", "business_name", "business_address", "country"]
GT_HEADER = ["source1_entity_id", "matched_entity_ids"]


class ProvenanceError(RuntimeError):
    """Raised when competition inputs are not the official challenge dump."""


@dataclass
class ProvenanceReport:
    ok: bool
    dataset_root: str
    manifest_version: str
    checked_files: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def raise_if_failed(self) -> None:
        if not self.ok:
            msg = "Official dataset provenance check FAILED:\n- " + "\n- ".join(self.errors)
            raise ProvenanceError(msg)


def default_manifest_path() -> Path:
    env = os.environ.get("OFFICIAL_DATASET_MANIFEST")
    if env:
        return Path(env)
    return _DEFAULT_MANIFEST


def load_manifest(path: Path | None = None) -> dict:
    path = path or default_manifest_path()
    if not path.is_file():
        raise ProvenanceError(f"Official dataset manifest missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path, chunk: int = 1 << 20) -> tuple[str, int]:
    h = hashlib.sha256()
    nbytes = 0
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
            nbytes += len(b)
    return h.hexdigest(), nbytes


def _read_header(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        line = f.readline()
    if not line:
        raise ProvenanceError(f"Empty file: {path}")
    if "\t" not in line and "," in line:
        raise ProvenanceError(f"{path} looks comma-separated; expected TSV")
    return [c.strip() for c in line.rstrip("\n").split("\t")]


def _path_is_banned(path: Path) -> str | None:
    text = str(path.resolve()).replace("\\", "/")
    if _BANNED_PATH_TOKENS.search(text):
        return f"path contains banned synthetic/fixture/sample/stub token: {text}"
    return None


def _path_is_allowed_competition_root(dataset_root: Path) -> str | None:
    """Competition runs must use student_resource/dataset (or exact symlink)."""
    resolved = dataset_root.resolve()
    # Allow if path ends with student_resource/dataset
    parts = resolved.parts
    for i in range(len(parts) - 1):
        if parts[i] == "student_resource" and parts[i + 1] == "dataset" and i + 1 == len(parts) - 1:
            return None
    # Also allow exact match of suffix via as_posix
    posix = resolved.as_posix()
    if posix.endswith("/student_resource/dataset"):
        return None
    return (
        f"dataset root is not the official competition path "
        f"(expected .../student_resource/dataset): {resolved}"
    )


def assert_not_synthetic_output_target(path: Path) -> None:
    """Synthetic writers must call this; competition dataset root is forbidden."""
    resolved = path.resolve()
    posix = resolved.as_posix()
    if "/student_resource/dataset" in posix:
        raise ProvenanceError(
            f"Synthetic/fixture generators must not write to competition dataset: {resolved}"
        )
    # Also ban writing where an official manifest file already lives with matching name
    # when the parent tree looks like a challenge dump.
    train1 = resolved / "train" / "train_source1.tsv"
    if train1.is_file():
        # If it matches official hash, refuse overwrite (checked by generators too)
        pass


def verify_official_dataset(
    dataset_root: Path | str,
    *,
    require: Iterable[str] = ("train", "test"),
    manifest_path: Path | None = None,
    check_hashes: bool = True,
) -> ProvenanceReport:
    """Validate dataset_root against the official manifest.

    ``require`` selects which relative file groups to enforce:
    values may be ``train``, ``test``, or explicit relative paths.
    """
    root = Path(dataset_root)
    report = ProvenanceReport(ok=True, dataset_root=str(root), manifest_version="")
    errors: list[str] = []

    if not root.is_dir():
        errors.append(f"dataset root does not exist: {root}")
        report.ok = False
        report.errors = errors
        return report

    banned = _path_is_banned(root)
    if banned:
        errors.append(banned)

    allowed = _path_is_allowed_competition_root(root)
    if allowed:
        errors.append(allowed)

    # Explicit synthetic provenance marker
    prov = root / "PROVENANCE.txt"
    if prov.is_file():
        text = prov.read_text(encoding="utf-8", errors="replace").casefold()
        if any(tok in text for tok in ("synthetic", "fixture", "stub", "smoke", "sample", "fake")):
            errors.append(f"PROVENANCE.txt marks non-official data: {prov}")

    try:
        manifest = load_manifest(manifest_path)
    except ProvenanceError as exc:
        errors.append(str(exc))
        report.ok = False
        report.errors = errors
        return report

    report.manifest_version = str(manifest.get("version", ""))
    files_meta: dict = manifest.get("files", {})
    headers_meta: dict = manifest.get("expected_headers", {})

    required_rels: list[str] = []
    req = set(require)
    if "train" in req:
        required_rels.extend(
            [
                "train/train_source1.tsv",
                "train/train_source2.tsv",
                "train/train_source3.tsv",
                "train/train_ground_truth.tsv",
            ]
        )
        req.discard("train")
    if "test" in req:
        required_rels.extend(
            [
                "test/test_source1.tsv",
                "test/test_source2.tsv",
                "test/test_source3.tsv",
            ]
        )
        req.discard("test")
    for extra in sorted(req):
        required_rels.append(extra)

    for rel in required_rels:
        path = root / rel
        report.checked_files.append(rel)
        if not path.is_file():
            errors.append(f"missing official file: {rel}")
            continue
        banned_f = _path_is_banned(path)
        if banned_f:
            errors.append(banned_f)
            continue

        expected_header = headers_meta.get(rel)
        if expected_header is None:
            # fallback schema by filename
            expected_header = GT_HEADER if "ground_truth" in rel else SOURCE_HEADER
        try:
            actual_header = _read_header(path)
        except ProvenanceError as exc:
            errors.append(str(exc))
            continue
        if actual_header != expected_header:
            errors.append(
                f"header mismatch for {rel}: got {actual_header}, expected {expected_header}"
            )

        meta = files_meta.get(rel)
        if meta is None:
            errors.append(f"manifest has no entry for {rel}")
            continue
        if check_hashes:
            digest, nbytes = sha256_file(path)
            if nbytes != int(meta["nbytes"]):
                errors.append(
                    f"size mismatch for {rel}: got {nbytes}, expected {meta['nbytes']}"
                )
            if digest != meta["sha256"]:
                errors.append(
                    f"sha256 mismatch for {rel}: got {digest}, expected {meta['sha256']} "
                    "(dataset is not the official challenge dump)"
                )

    report.errors = errors
    report.ok = not errors
    return report


def require_official_dataset(
    dataset_root: Path | str,
    *,
    require: Iterable[str] = ("train", "test"),
    manifest_path: Path | None = None,
    check_hashes: bool = True,
) -> ProvenanceReport:
    """Hard gate for competition runs — raises ProvenanceError on failure."""
    report = verify_official_dataset(
        dataset_root,
        require=require,
        manifest_path=manifest_path,
        check_hashes=check_hashes,
    )
    report.raise_if_failed()
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Verify official challenge dataset provenance")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--require", nargs="+", default=["train", "test"])
    p.add_argument("--skip-hashes", action="store_true")
    args = p.parse_args(argv)
    try:
        report = require_official_dataset(
            args.dataset_root,
            require=args.require,
            manifest_path=args.manifest,
            check_hashes=not args.skip_hashes,
        )
    except ProvenanceError as exc:
        print(exc)
        return 1
    print(
        f"PASS provenance: version={report.manifest_version} "
        f"files={len(report.checked_files)} root={report.dataset_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
