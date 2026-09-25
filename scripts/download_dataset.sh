#!/usr/bin/env bash
# Download challenge dataset zips from the GitHub release and extract into student_resource/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-$ROOT/student_resource}"
TAG="${DATASET_RELEASE_TAG:-dataset-v1}"
REPO="${DATASET_REPO:-sshivanshg/Mlchallenge}"
TMP="${TMPDIR:-/tmp}/mlchallenge-dataset-$$"

mkdir -p "$TMP" "$DEST"
echo "Downloading release assets ($REPO @ $TAG) ..."
if command -v gh >/dev/null 2>&1; then
  gh release download "$TAG" -R "$REPO" -D "$TMP" -p 'dataset_*.zip'
else
  echo "gh CLI not found; falling back to curl from GitHub release API"
  api="https://api.github.com/repos/${REPO}/releases/tags/${TAG}"
  urls=$(curl -fsSL "$api" | python3 -c "import sys,json; print('\n'.join(a['browser_download_url'] for a in json.load(sys.stdin).get('assets',[]) if a['name'].startswith('dataset_')))")
  while IFS= read -r url; do
    [[ -z "$url" ]] && continue
    echo "GET $url"
    curl -fL "$url" -o "$TMP/$(basename "$url")"
  done <<< "$urls"
fi

echo "Extracting into $DEST ..."
unzip -o "$TMP/dataset_train.zip" -d "$DEST"
unzip -o "$TMP/dataset_test.zip" -d "$DEST"
rm -rf "$TMP"

echo "Done. Dataset layout:"
find "$DEST/dataset" -type f -name '*.tsv' | sort
du -sh "$DEST/dataset" "$DEST/dataset/train" "$DEST/dataset/test"
