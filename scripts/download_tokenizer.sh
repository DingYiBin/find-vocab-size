#!/usr/bin/env bash
# Download the Qwen3-4B tokenizer into ./models/Qwen/Qwen3-4B (git-ignored).
# Usage: ./scripts/download_tokenizer.sh
#
# Tries huggingface.co first; if unreachable (firewalled network), falls back
# to the hf-mirror.com mirror (override with HF_ENDPOINT). Uses curl to avoid
# SSL failures behind corporate MITM proxies whose CA is in the system store.
set -euo pipefail

DEST="$(cd "$(dirname "$0")/.." && pwd)/models/Qwen/Qwen3-4B"
REPO_PATH="Qwen/Qwen3-4B/resolve/main"
FILES=(tokenizer.json tokenizer_config.json vocab.json merges.txt)
mkdir -p "$DEST"

probe() { curl -sfL --max-time 10 -o /dev/null "https://huggingface.co/$REPO_PATH/tokenizer_config.json"; }

if probe 2>/dev/null; then
  BASE="https://huggingface.co/$REPO_PATH"
else
  BASE="${HF_ENDPOINT:-https://hf-mirror.com}/$REPO_PATH"
fi
echo ">> downloading tokenizer from $BASE"

for f in "${FILES[@]}"; do
  if [[ -s "$DEST/$f" ]]; then
    echo ">> $f already exists, skipping (delete it to re-download)"
    continue
  fi
  curl -fSL --retry 3 --create-dirs -o "$DEST/$f" "$BASE/$f"
  echo ">> $f done"
done

echo ">> contents of $DEST:"
ls -la "$DEST"
