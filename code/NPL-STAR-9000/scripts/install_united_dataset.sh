#!/usr/bin/env bash
# Put united-cryptonite-wordplay-dataset where configs/base.yaml expects it.
#
#   scripts/install_united_dataset.sh ~/Downloads/CrypticCrossword-main.zip
#   scripts/install_united_dataset.sh /path/to/united-cryptonite-wordplay-dataset
#
# Copies only that one directory out of the repo/zip. The .jsonl.gz files stay
# gzipped - the loader reads them as-is, and unzipped train.jsonl is ~205 MB.
set -euo pipefail

SRC="${1:-}"
DEST="data/raw/united-cryptonite-wordplay-dataset"
INNER="united-cryptonite-wordplay-dataset"

if [[ -z "$SRC" ]]; then
  echo "usage: $0 <CrypticCrossword zip | united-cryptonite-wordplay-dataset dir>" >&2
  exit 2
fi
if [[ -e "$DEST" ]]; then
  echo "$DEST already exists - remove it first if you want to replace it." >&2
  exit 1
fi

mkdir -p "$(dirname "$DEST")"

if [[ -d "$SRC" ]]; then
  cp -R "$SRC" "$DEST"
elif [[ "$SRC" == *.zip ]]; then
  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  # Only the dataset directory, wherever it sits inside the archive.
  unzip -q "$SRC" "*/${INNER}/*" -d "$TMP"
  FOUND="$(find "$TMP" -type d -name "$INNER" -print -quit)"
  if [[ -z "$FOUND" ]]; then
    echo "No '$INNER' directory inside $SRC" >&2
    exit 1
  fi
  cp -R "$FOUND" "$DEST"
else
  echo "$SRC is neither a directory nor a .zip" >&2
  exit 2
fi

echo "installed -> $DEST"
ls -la "$DEST"
echo
echo "next:"
echo "  python -m cryptic_star.data.united -c configs/base.yaml"
echo "  cat data/processed/united_report.json"
