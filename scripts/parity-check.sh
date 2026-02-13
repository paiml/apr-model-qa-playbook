#!/usr/bin/env bash
# HF Parity Self-Check (all model families)
# Implements Popperian self-consistency: golden outputs must match themselves.
# Corpus structure: $HF_CORPUS/<model>/<version>/manifest.json

set -euo pipefail

HF_CORPUS="${1:?Usage: parity-check.sh <corpus-path>}"

echo "=== HF Parity Self-Check (all model families) ==="

found=0

while IFS= read -r -d '' manifest; do
    version_dir=$(dirname "${manifest}")
    model_version=$(echo "${version_dir}" | sed "s|${HF_CORPUS}/||")
    echo "Checking: ${model_version}"
    cargo run --bin apr-qa -- parity -m "${model_version}" -c "${HF_CORPUS}" --self-check || exit 1
    echo ""
    found=1
done < <(find "${HF_CORPUS}" -name "manifest.json" -type f -print0 2>/dev/null)

if [ "${found}" -eq 0 ]; then
    echo "No golden corpora found in ${HF_CORPUS}"
    exit 1
fi

echo "All parity self-checks passed!"
