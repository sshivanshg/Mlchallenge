"""Assemble <team_name>_submission.zip from a validated inference output directory.

Layout (student_resource/README.md, "Final Submission Package"):
  output/{matching_results,candidate_pairs}.tsv
  code/business_entity_resolution/{src/,configs/,README.md,requirements.txt}
  Documentation_template.md
Refuses to package unless the official validator passed on exactly these files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CODE = REPO / "code" / "business_entity_resolution"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True, help="dir with matching_results.tsv + candidate_pairs.tsv")
    p.add_argument("--team-name", required=True)
    p.add_argument("--dest", type=Path, default=REPO / "artifacts" / "packages")
    p.add_argument("--skip-validate", action="store_true")
    args = p.parse_args()
    out = args.output_dir.resolve()
    m, c = out / "matching_results.tsv", out / "candidate_pairs.tsv"
    if not args.skip_validate:
        r = subprocess.run(
            [sys.executable, "utils/validate_submission.py", "--matching", str(m), "--candidate", str(c),
             "--test-dir", "dataset/test"],
            cwd=REPO / "student_resource", capture_output=True, text=True,
        )
        print(r.stdout[-2000:])
        if r.returncode != 0 or "PASS" not in r.stdout:
            raise SystemExit("validator did not PASS; not packaging")
    args.dest.mkdir(parents=True, exist_ok=True)
    zpath = args.dest / f"{args.team_name}_submission.zip"
    tmp = zpath.with_suffix(".zip.tmp")
    files = [(m, "output/matching_results.tsv"), (c, "output/candidate_pairs.tsv")]
    for sub in ("src", "configs"):
        for f in sorted((CODE / sub).rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts and f.suffix not in {".pyc", ".sqlite"}:
                files.append((f, f"code/business_entity_resolution/{f.relative_to(CODE)}"))
    for name in ("README.md", "requirements.txt"):
        files.append((CODE / name, f"code/business_entity_resolution/{name}"))
    files.append((REPO / "student_resource" / "Documentation_template.md", "Documentation_template.md"))
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for src, arc in files:
            z.write(src, arc)
    with zipfile.ZipFile(tmp) as z:
        bad = z.testzip()
        if bad:
            raise SystemExit(f"corrupt member {bad}")
        names = z.namelist()
    tmp.replace(zpath)
    manifest = {
        "zip": str(zpath),
        "zip_sha256": sha256(zpath),
        "zip_bytes": zpath.stat().st_size,
        "members": len(names),
        "matching_sha256": sha256(m),
        "candidate_sha256": sha256(c),
        "git_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip(),
    }
    zpath.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
