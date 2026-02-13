#!/usr/bin/env bash
# check-docs-consistency.sh — Prevent documentation drift at interface boundaries
#
# Checks:
#   1. All workspace crates appear in CLAUDE.md crate structure
#   2. All workspace crates appear in book architecture overview
#   3. All CLI subcommands appear in book CLI reference
#   4. Gateway count is consistent across docs (G0-G4 = 5 gateways)
#   5. README certification table is not stale (> 7 days)
#   6. README model count matches models.csv
#
# Exit 0 = all checks pass, Exit 1 = drift detected

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ERRORS=0
WARNINGS=0

red()    { printf '\033[1;31m%s\033[0m\n' "$*"; }
yellow() { printf '\033[1;33m%s\033[0m\n' "$*"; }
green()  { printf '\033[1;32m%s\033[0m\n' "$*"; }
info()   { printf '  %s\n' "$*"; }

fail() { red "FAIL: $*"; ERRORS=$((ERRORS + 1)); }
warn() { yellow "WARN: $*"; WARNINGS=$((WARNINGS + 1)); }
pass() { green "PASS: $*"; }

echo "=== Documentation Consistency Check ==="
echo ""

# ---------------------------------------------------------------------------
# 1. All workspace crates appear in CLAUDE.md
# ---------------------------------------------------------------------------
echo "--- Check 1: Workspace crates in CLAUDE.md ---"

mapfile -t CRATES < <(grep -oP '"crates/\K[^"]+' "${ROOT}/Cargo.toml" | sort -u)
CLAUDE_MD="${ROOT}/CLAUDE.md"

for crate in "${CRATES[@]}"; do
    if grep -q "${crate}" "${CLAUDE_MD}"; then
        info "${crate} ... found"
    else
        fail "${crate} missing from CLAUDE.md crate structure"
    fi
done
echo ""

# ---------------------------------------------------------------------------
# 2. All workspace crates appear in book architecture overview
# ---------------------------------------------------------------------------
echo "--- Check 2: Workspace crates in book/architecture/overview.md ---"

BOOK_OVERVIEW="${ROOT}/book/src/architecture/overview.md"

for crate in "${CRATES[@]}"; do
    if grep -q "${crate}" "${BOOK_OVERVIEW}"; then
        info "${crate} ... found"
    else
        fail "${crate} missing from book/src/architecture/overview.md"
    fi
done
echo ""

# ---------------------------------------------------------------------------
# 3. All CLI subcommands appear in book CLI reference
# ---------------------------------------------------------------------------
echo "--- Check 3: CLI subcommands in book/reference/cli.md ---"

CLI_MAIN="${ROOT}/crates/apr-qa-cli/src/main.rs"
CLI_REF="${ROOT}/book/src/reference/cli.md"

# Extract subcommand variant names from the Commands enum, convert to kebab-case.
# Look for lines matching "    VariantName {" inside enum Commands.
# Filter to only PascalCase names (start with uppercase, no underscores).
mapfile -t SUBCOMMANDS < <(sed -n '/^enum Commands/,/^}/p' "${CLI_MAIN}" \
    | grep -oP '^\s+([A-Z][a-zA-Z]+)\s*\{' \
    | grep -oP '[A-Z][a-zA-Z]+' \
    | sed 's/\([A-Z]\)/-\L\1/g; s/^-//' \
    | tr '[:upper:]' '[:lower:]')

for cmd in "${SUBCOMMANDS[@]}"; do
    if grep -qi "${cmd}" "${CLI_REF}"; then
        info "${cmd} ... found"
    else
        fail "CLI subcommand '${cmd}' missing from book/src/reference/cli.md"
    fi
done
echo ""

# ---------------------------------------------------------------------------
# 4. Gateway count consistency
# ---------------------------------------------------------------------------
echo "--- Check 4: Gateway count consistency ---"

# The source of truth: CLAUDE.md Oracle Integration section lists G0-G4
EXPECTED_PATTERN="G0-G4"

check_gateway_ref() {
    local file="$1"
    local label="$2"
    if grep -q "${EXPECTED_PATTERN}" "${file}"; then
        info "${label} ... references ${EXPECTED_PATTERN}"
    elif grep -q "G1-G4" "${file}"; then
        fail "${label} still references G1-G4 (should be ${EXPECTED_PATTERN})"
    else
        info "${label} ... no gateway range reference (OK)"
    fi
}

check_gateway_ref "${ROOT}/README.md" "README.md"
check_gateway_ref "${CLAUDE_MD}" "CLAUDE.md"
check_gateway_ref "${ROOT}/book/src/philosophy/mqs.md" "book/philosophy/mqs.md"
check_gateway_ref "${ROOT}/book/src/reference/gateways.md" "book/reference/gateways.md"
check_gateway_ref "${ROOT}/book/src/reference/certified-testing.md" "book/reference/certified-testing.md"
check_gateway_ref "${ROOT}/book/src/introduction.md" "book/introduction.md"

# Check that gateways.md has all G0-G4 sections
GATEWAY_DEFS="${ROOT}/book/src/reference/gateways.md"
for g in G0 G1 G2 G3 G4; do
    if grep -q "## ${g}:" "${GATEWAY_DEFS}"; then
        info "gateways.md has ${g} section"
    else
        fail "gateways.md missing ## ${g}: section"
    fi
done
echo ""

# ---------------------------------------------------------------------------
# 5. README certification table freshness
# ---------------------------------------------------------------------------
echo "--- Check 5: README certification table freshness ---"

TABLE_DATE="$(grep -oP 'updated: \K[0-9]{4}-[0-9]{2}-[0-9]{2}' "${ROOT}/README.md" | head -1)"
if [ -n "${TABLE_DATE}" ]; then
    TODAY="$(date +%Y-%m-%d)"
    TABLE_EPOCH="$(date -d "${TABLE_DATE}" +%s 2>/dev/null || echo 0)"
    TODAY_EPOCH="$(date -d "${TODAY}" +%s)"
    DAYS_OLD=$(( (TODAY_EPOCH - TABLE_EPOCH) / 86400 ))

    if [ "${DAYS_OLD}" -le 7 ]; then
        pass "Certification table is ${DAYS_OLD} day(s) old (updated: ${TABLE_DATE})"
    else
        warn "Certification table is ${DAYS_OLD} day(s) old (updated: ${TABLE_DATE}). Run: make update-certifications"
    fi
else
    fail "Could not parse certification table date from README.md"
fi
echo ""

# ---------------------------------------------------------------------------
# 6. README model count matches models.csv
# ---------------------------------------------------------------------------
echo "--- Check 6: README model count vs models.csv ---"

CSV_FILE="${ROOT}/docs/certifications/models.csv"
if [ -f "${CSV_FILE}" ]; then
    # CSV line count minus header
    CSV_COUNT=$(( $(wc -l < "${CSV_FILE}") - 1 ))

    # README summary count (e.g., "Certified | 9/92")
    README_TOTAL="$(grep -oP '\d+/\K\d+' "${ROOT}/README.md" | head -1)"

    if [ -n "${README_TOTAL}" ]; then
        if [ "${CSV_COUNT}" -eq "${README_TOTAL}" ]; then
            pass "README shows ${README_TOTAL} models, CSV has ${CSV_COUNT} (match)"
        else
            fail "README shows ${README_TOTAL} models but CSV has ${CSV_COUNT}. Run: make update-certifications"
        fi
    else
        warn "Could not parse model count from README.md"
    fi
else
    warn "models.csv not found at ${CSV_FILE}"
fi
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "=== Summary ==="
if [ "${ERRORS}" -eq 0 ] && [ "${WARNINGS}" -eq 0 ]; then
    green "All documentation consistency checks passed."
    exit 0
elif [ "${ERRORS}" -eq 0 ]; then
    yellow "${WARNINGS} warning(s), 0 errors."
    exit 0
else
    red "${ERRORS} error(s), ${WARNINGS} warning(s)."
    echo ""
    echo "Fix documentation drift and re-run: make docs-check"
    exit 1
fi
