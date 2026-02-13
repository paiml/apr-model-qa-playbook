#!/usr/bin/env bash
# List available golden outputs for all model families
# Corpus structure: $HF_CORPUS/<model>/<version>/manifest.json

set -euo pipefail

HF_CORPUS="${1:?Usage: parity-list.sh <corpus-path>}"

echo "=== Available Golden Outputs ==="

while IFS= read -r -d '' manifest; do
    version_dir=$(dirname "${manifest}")
    model_version=$(echo "${version_dir}" | sed "s|${HF_CORPUS}/||")
    echo ""
    echo "--- ${model_version} ---"
    cargo run --bin apr-qa -- parity -m "${model_version}" -c "${HF_CORPUS}" --list 2>/dev/null || true
done < <(find "${HF_CORPUS}" -name "manifest.json" -type f -print0 2>/dev/null)
