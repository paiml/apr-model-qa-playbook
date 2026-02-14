#!/usr/bin/env bash
# certify-quickstart.sh - Interactive model certification helper
#
# Usage:
#   ./scripts/certify-quickstart.sh <model-family> [tier]
#
# Examples:
#   ./scripts/certify-quickstart.sh qwen-coder mvp
#   ./scripts/certify-quickstart.sh llama smoke
#   ./scripts/certify-quickstart.sh deepseek-coder  # defaults to mvp

set -euo pipefail

FAMILY="${1:?Usage: $0 <model-family> [tier]}"
TIER="${2:-mvp}"

echo "=== APR Model Certification ==="
echo "Family: ${FAMILY}"
echo "Tier:   ${TIER}"
echo ""

# Step 1: Find matching playbooks
echo "--- Step 1: Finding playbooks ---"
PLAYBOOKS=$(find playbooks/models/ -name "*${FAMILY}*${TIER}*" -o -name "*${FAMILY}*" 2>/dev/null | sort)

if [ -z "${PLAYBOOKS}" ]; then
    echo "No playbooks found for family '${FAMILY}'"
    echo ""
    echo "Available families:"
    ls playbooks/models/ | sed 's/-[a-z]*\.playbook\.yaml//' | sort -u | head -20
    exit 1
fi

echo "Found playbooks:"
echo "${PLAYBOOKS}" | while read -r p; do echo "  - ${p}"; done
echo ""

# Step 2: Dry run
PLAYBOOK=$(echo "${PLAYBOOKS}" | head -1)
echo "--- Step 2: Dry run (${PLAYBOOK}) ---"
cargo run --bin apr-qa -- certify \
    --family "${FAMILY}" \
    --tier "${TIER}" \
    --dry-run 2>&1 || true
echo ""

# Step 3: Prompt to continue
echo "--- Step 3: Ready to certify ---"
echo ""
echo "To run certification:"
echo "  cargo run --bin apr-qa -- certify --family ${FAMILY} --tier ${TIER}"
echo ""
echo "To run a specific playbook:"
echo "  cargo run --bin apr-qa -- run ${PLAYBOOK}"
echo ""
echo "After certification:"
echo "  make update-certifications"
