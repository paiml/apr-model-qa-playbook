#!/usr/bin/env bash
# PMAT Coverage Compliance Check
# Verifies library coverage meets the 95% threshold.
# Uses --lib to exclude binary code (main.rs files).
# NO GAMING: All library code must meet the threshold.

set -euo pipefail

THRESHOLD=95

echo "Checking PMAT coverage compliance (>= ${THRESHOLD}%)..."

COVERAGE=$(cargo llvm-cov --workspace --lib 2>&1 | \
    grep "^TOTAL" | awk '{print $10}' | tr -d '%')

if [ -z "${COVERAGE}" ]; then
    echo "ERROR: Could not parse coverage from cargo llvm-cov output"
    exit 1
fi

RESULT=$(echo "${COVERAGE} >= ${THRESHOLD}" | bc)

if [ "${RESULT}" -eq 1 ]; then
    echo "Coverage: ${COVERAGE}% (PMAT compliant)"
else
    echo "Coverage: ${COVERAGE}% (below ${THRESHOLD}% threshold)"
    exit 1
fi
