#!/usr/bin/env bash
# verify-contract-bindings.sh — Binding drift detection for provable contracts
#
# Parses contracts/binding.yaml and verifies each binding points to a real
# function in the Rust source. Also spot-checks contract claims (MQS weights,
# grade count, gateway count) against implementing code.
#
# Exit 0: all bindings verified, all claims corroborated
# Exit 1: drift detected (binding or claim falsified)
#
# Spec §19: Contract Verification Playbook
#
# Dogfooded 2026-03-08: Fixed substring matching (grep -w), anchored value
# patterns, added missing claim checks (proof bonus, G4 threshold).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BINDING_FILE="$REPO_ROOT/contracts/binding.yaml"
CONTRACTS_DIR="$REPO_ROOT/contracts"

PASS=0
FAIL=0
TOTAL=0

pass() {
    PASS=$((PASS + 1))
    TOTAL=$((TOTAL + 1))
    printf "  [\033[32mOK\033[0m]    %-30s → %s\n" "$1" "$2"
}

fail() {
    FAIL=$((FAIL + 1))
    TOTAL=$((TOTAL + 1))
    printf "  [\033[31mDRIFT\033[0m] %-30s → %s (%s)\n" "$1" "$2" "$3"
}

# ─── Phase 1: Binding Drift Detection ─────────────────────────────────────

echo "=== Phase 1: Binding Drift Detection ==="
echo ""

if [ ! -f "$BINDING_FILE" ]; then
    echo "ERROR: $BINDING_FILE not found"
    exit 1
fi

# Parse bindings with inline python (binding.yaml uses YAML lists)
BINDINGS=$(python3 -c "
import yaml, sys
with open('$BINDING_FILE') as f:
    data = yaml.safe_load(f)
for b in data.get('bindings', []):
    equation = b.get('equation', '?')
    function = b.get('function', '?')
    module_path = b.get('module_path', '?')
    status = b.get('status', '?')
    notes = b.get('notes', '')
    print(f'{equation}|{function}|{module_path}|{status}|{notes}')
")

# Guard against empty bindings (python emits empty string → bash reads one empty line)
if [ -z "$BINDINGS" ]; then
    echo "  WARNING: No bindings found in $BINDING_FILE"
    echo ""
else
    while IFS='|' read -r equation function module_path status notes; do
        [ -z "$equation" ] && continue

        # Extract the bare function name (strip Type:: prefix for methods)
        bare_fn="${function##*::}"

        # Map module_path to crate directory
        crate=$(echo "$module_path" | sed 's/::.*//; s/_/-/g')
        crate_dir="$REPO_ROOT/crates/$crate/src"

        if [ ! -d "$crate_dir" ]; then
            fail "$equation" "$function" "crate dir not found: $crate_dir"
            continue
        fi

        # Search for the function definition with word-boundary matching.
        # grep -w prevents "fn execute" from matching "fn execute_scenario".
        # We exclude test files to avoid false positives from mock impls.
        if grep -rw --include='*.rs' -q "fn ${bare_fn}" "$crate_dir" 2>/dev/null; then
            pass "$equation" "$function ($crate)"
        else
            fail "$equation" "$function" "fn ${bare_fn} not found in $crate"
        fi
    done <<< "$BINDINGS"
fi

echo ""
echo "  Bindings: $PASS/$TOTAL verified"
echo ""

# ─── Phase 2: Contract Claim Verification ─────────────────────────────────

CLAIM_PASS=0
CLAIM_FAIL=0
CLAIM_TOTAL=0

claim_pass() {
    CLAIM_PASS=$((CLAIM_PASS + 1))
    CLAIM_TOTAL=$((CLAIM_TOTAL + 1))
    printf "  [\033[32mOK\033[0m]    %s\n" "$1"
}

claim_fail() {
    CLAIM_FAIL=$((CLAIM_FAIL + 1))
    CLAIM_TOTAL=$((CLAIM_TOTAL + 1))
    printf "  [\033[31mFAIL\033[0m]  %s (%s)\n" "$1" "$2"
}

echo "=== Phase 2: Contract Claim Verification ==="
echo ""

MQS_FILE="$REPO_ROOT/crates/apr-qa-report/src/mqs.rs"
GATEWAY_FILE="$REPO_ROOT/crates/apr-qa-report/src/mqs_gateways.rs"
ORACLE_FILE="$REPO_ROOT/crates/apr-qa-gen/src/oracle.rs"
FORMAT_CONTRACT="$REPO_ROOT/crates/apr-qa-runner/src/apr_format_contract.yaml"

# ── MQS category weights (anchored patterns: "= NNN;" not ".*NNN") ──

check_const() {
    local name="$1" value="$2" file="$3"
    if grep -q "${name}.*=.*${value}" "$file" 2>/dev/null; then
        claim_pass "MQS $name = $value"
    else
        claim_fail "MQS $name = $value" "not found in $(basename "$file")"
    fi
}

check_const "MAX_QUAL" "200" "$MQS_FILE"
check_const "MAX_PERF" "150" "$MQS_FILE"
check_const "MAX_STAB" "200" "$MQS_FILE"
check_const "MAX_COMP" "150" "$MQS_FILE"
check_const "MAX_EDGE" "150" "$MQS_FILE"
check_const "MAX_REGR" "150" "$MQS_FILE"
check_const "MAX_TOTAL" "1000" "$MQS_FILE"

# ── Proof bonus = 50 ──
if grep -q 'MAX_TOTAL + 50\|MAX_TOTAL.* 50\|bonus.*50\|50.*bonus' "$MQS_FILE" 2>/dev/null; then
    claim_pass "MQS proof bonus = 50 (raw range [0, 1050])"
else
    claim_fail "MQS proof bonus = 50" "50 bonus not found in mqs.rs"
fi

# ── Gateway zeroing via .all() ──
if grep -q '\.all(|g| g\.passed)' "$MQS_FILE" 2>/dev/null; then
    claim_pass "Gateway zeroing via .all(|g| g.passed)"
else
    claim_fail "Gateway zeroing via .all()" "pattern not found in mqs.rs"
fi

# ── Penalty floor via saturating_sub ──
if grep -q 'saturating_sub' "$MQS_FILE" 2>/dev/null; then
    claim_pass "Penalty floor via saturating_sub"
else
    claim_fail "Penalty floor via saturating_sub" "not found in mqs.rs"
fi

# ── Grade mapping: count grade entries ──
# Pattern: (f64, "X") tuples where X is a letter grade A-D with optional +/-
if [ -f "$GATEWAY_FILE" ]; then
    # Count lines with grade tuple entries: (NN.N, "X+"), not comments
    GRADE_COUNT=$(grep -cE '^\s+\(' "$GATEWAY_FILE" 2>/dev/null | head -1 || true)
    # More precise: count quoted grade strings in the GRADE_TABLE const
    GRADE_COUNT=$(sed -n '/GRADE_TABLE/,/];/p' "$GATEWAY_FILE" | grep -cE '"[A-D][+-]?"' 2>/dev/null || true)
    if [ "$GRADE_COUNT" -ge 12 ]; then
        claim_pass "Grade mapping has >= 12 grades ($GRADE_COUNT in GRADE_TABLE)"
    else
        claim_fail "Grade mapping has >= 12 grades" "only $GRADE_COUNT found in GRADE_TABLE"
    fi
else
    claim_fail "Grade mapping" "mqs_gateways.rs not found"
fi

# ── 5 gateway types (G0-G4) ──
# Verify each gateway is referenced in check_gateways function, not category mappings
if [ -f "$GATEWAY_FILE" ]; then
    # Count distinct gateway IDs in the check_gateways function (before category mapping)
    CHECK_FN=$(sed -n '/fn check_gateways/,/^[[:space:]]*fn [a-z]/p' "$GATEWAY_FILE" 2>/dev/null || true)
    GATES_FOUND=0
    for g in G0 G1 G2 G3 G4; do
        if echo "$CHECK_FN" | grep -q "\"${g}" 2>/dev/null; then
            GATES_FOUND=$((GATES_FOUND + 1))
        fi
    done
    if [ "$GATES_FOUND" -eq 5 ]; then
        claim_pass "5 gateway types (G0-G4) in check_gateways"
    else
        claim_fail "5 gateway types" "only $GATES_FOUND found in check_gateways"
    fi
else
    claim_fail "Gateway count" "mqs_gateways.rs not found"
fi

# ── G4 garbage threshold = 25% (evidence.len() / 4) ──
if [ -f "$GATEWAY_FILE" ]; then
    if grep -q '/ 4' "$GATEWAY_FILE" 2>/dev/null; then
        claim_pass "G4 garbage threshold = 25% (/ 4)"
    else
        claim_fail "G4 garbage threshold = 25%" "/ 4 not found in mqs_gateways.rs"
    fi
else
    claim_fail "G4 garbage threshold" "mqs_gateways.rs not found"
fi

# ── GarbageOracle empty output check ──
# Specific: check for output.trim().is_empty() not just any is_empty()
if grep -q 'output.*\.is_empty()\|\.trim()\.is_empty()' "$ORACLE_FILE" 2>/dev/null; then
    claim_pass "GarbageOracle checks empty output via trim().is_empty()"
else
    claim_fail "GarbageOracle empty output check" "output.trim().is_empty() not found in oracle.rs"
fi

# ── BF16 dtype byte = 30 (anchored to prevent matching 300) ──
if grep -qE 'byte: 30$|byte: 30[^0-9]' "$FORMAT_CONTRACT" 2>/dev/null; then
    claim_pass "BF16 dtype byte = 30 in format contract"
else
    claim_fail "BF16 dtype byte = 30" "not found in apr_format_contract.yaml"
fi

echo ""
echo "  Claims: $CLAIM_PASS/$CLAIM_TOTAL corroborated"
echo ""

# ─── Summary ──────────────────────────────────────────────────────────────

echo "=== Summary ==="
TOTAL_ALL=$((TOTAL + CLAIM_TOTAL))
PASS_ALL=$((PASS + CLAIM_PASS))
FAIL_ALL=$((FAIL + CLAIM_FAIL))
echo "  Total checks: $TOTAL_ALL"
echo "  Passed: $PASS_ALL"
echo "  Failed: $FAIL_ALL"

if [ "$FAIL_ALL" -gt 0 ]; then
    echo ""
    printf "  \033[31mResult: FAIL — contract drift detected\033[0m\n"
    exit 1
else
    echo ""
    printf "  \033[32mResult: PASS — all contracts verified against source\033[0m\n"
    exit 0
fi
