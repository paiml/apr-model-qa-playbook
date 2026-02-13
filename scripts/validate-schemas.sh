#!/usr/bin/env bash
# PMAT-273: Schema Validation Script
#
# Validates:
# 1. Evidence JSON files in certifications/
# 2. models.csv format and constraints
#
# Usage: ./scripts/validate-schemas.sh
# Exit codes:
#   0 - All validations passed
#   1 - Validation errors found

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

ERRORS=0
WARNINGS=0

log_error() {
    echo -e "${RED}ERROR:${NC} $1" >&2
    ((ERRORS++)) || true
}

log_warning() {
    echo -e "${YELLOW}WARNING:${NC} $1" >&2
    ((WARNINGS++)) || true
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
}

log_info() {
    echo -e "  $1"
}

# Check if jq is available
check_dependencies() {
    if ! command -v jq &> /dev/null; then
        log_error "jq is required but not installed. Install with: sudo apt-get install jq"
        exit 1
    fi
}

# Validate a single evidence.json file
validate_evidence_json() {
    local file="$1"
    local file_errors=0

    # Check if file is valid JSON
    if ! jq empty "$file" 2>/dev/null; then
        log_error "$file: Invalid JSON syntax"
        return 1
    fi

    # Check if it's an array
    local is_array
    is_array=$(jq 'type == "array"' "$file")
    if [ "${is_array}" != "true" ]; then
        log_error "$file: Root element must be an array"
        return 1
    fi

    # Check array is not empty (warning only)
    local length
    length=$(jq 'length' "$file")
    if [ "${length}" = "0" ]; then
        log_warning "$file: Evidence array is empty"
    fi

    # Validate each entry has required fields
    local missing_count
    missing_count=$(jq '[.[] | select(.id == null or .gate_id == null or .outcome == null or .timestamp == null)] | length' "$file")
    if [ "${missing_count}" != "0" ]; then
        log_error "$file: $missing_count entries missing required fields (id, gate_id, outcome, timestamp)"
        ((file_errors++)) || true
    fi

    # Validate outcome values
    local invalid_outcome_count
    invalid_outcome_count=$(jq '[.[] | select(.outcome != "Corroborated" and .outcome != "Falsified" and .outcome != "Timeout" and .outcome != "Crashed" and .outcome != "Skipped")] | length' "$file")
    if [ "${invalid_outcome_count}" != "0" ]; then
        log_error "$file: $invalid_outcome_count entries have invalid outcome values"
        ((file_errors++)) || true
    fi

    if [ "${file_errors}" -eq 0 ]; then
        log_success "$file: Valid ($length entries)"
        return 0
    else
        return 1
    fi
}

# Validate models.csv
validate_models_csv() {
    local file="$1"
    local line_num=0
    local file_errors=0

    # Expected header
    local expected_header="model_id,family,parameters,size_category,status,mqs_score,grade,certified_tier,last_certified,g1,g2,g3,g4,tps_gguf_cpu,tps_gguf_gpu,tps_apr_cpu,tps_apr_gpu,tps_st_cpu,tps_st_gpu,provenance_verified"

    # Check header
    local actual_header
    actual_header=$(head -1 "$file")
    if [ "${actual_header}" != "${expected_header}" ]; then
        log_error "$file: Invalid header"
        log_info "Expected: $expected_header"
        log_info "Got:      $actual_header"
        return 1
    fi

    # Valid values
    local valid_statuses="CERTIFIED BLOCKED PENDING UNTESTED"
    local valid_sizes="tiny small medium large xlarge huge"
    local valid_grades="A B C D F -"

    # Validate each row (skip header)
    while IFS= read -r line || [ -n "${line}" ]; do
        ((line_num++)) || true

        # Skip header
        [ "${line_num}" -eq 1 ] && continue

        # Skip empty lines
        [ -z "${line}" ] && continue

        # Parse CSV fields
        IFS=',' read -r model_id _family _parameters size_category status mqs_score grade _certified_tier last_certified g1 g2 g3 g4 rest <<< "$line"

        # Validate model_id format (org/name)
        if ! echo "${model_id}" | grep -qE '^[^/]+/[^/]+$'; then
            log_error "$file:$line_num: Invalid model_id format (expected org/name)"
            ((file_errors++)) || true
        fi

        # Validate size_category
        case " ${valid_sizes} " in
            *" ${size_category} "*) ;;
            *) log_error "$file:$line_num: Invalid size_category '${size_category}'"
               ((file_errors++)) || true ;;
        esac

        # Validate status
        case " ${valid_statuses} " in
            *" ${status} "*) ;;
            *) log_error "$file:$line_num: Invalid status '${status}'"
               ((file_errors++)) || true ;;
        esac

        # Validate mqs_score (0-1000)
        if ! echo "${mqs_score}" | grep -qE '^[0-9]+$' || [ "${mqs_score}" -lt 0 ] || [ "${mqs_score}" -gt 1000 ]; then
            log_error "$file:$line_num: Invalid mqs_score '$mqs_score' (must be 0-1000)"
            ((file_errors++)) || true
        fi

        # Validate grade
        case " ${valid_grades} " in
            *" ${grade} "*) ;;
            *) log_error "$file:$line_num: Invalid grade '${grade}'"
               ((file_errors++)) || true ;;
        esac

        # Validate g1-g4 booleans
        for g in "${g1}" "${g2}" "${g3}" "${g4}"; do
            if [ "${g}" != "true" ] && [ "${g}" != "false" ]; then
                log_error "$file:$line_num: Invalid gateway value '$g' (must be true/false)"
                ((file_errors++)) || true
                break
            fi
        done

        # Validate last_certified timestamp format (basic ISO8601 check)
        if ! echo "${last_certified}" | grep -qE '^[0-9]{4}-[0-9]{2}-[0-9]{2}T'; then
            log_error "$file:$line_num: Invalid timestamp format"
            ((file_errors++)) || true
        fi

    done < "$file"

    local row_count=$((line_num - 1))

    if [ "${file_errors}" -eq 0 ]; then
        log_success "$file: Valid ($row_count rows)"
        return 0
    else
        return 1
    fi
}

# Main validation
main() {
    echo "=== APR QA Schema Validation (PMAT-273) ==="
    echo ""

    check_dependencies

    # Validate evidence.json files
    echo "Validating evidence.json files..."
    local evidence_count=0

    while IFS= read -r -d '' file; do
        validate_evidence_json "$file" || true
        ((evidence_count++)) || true
    done < <(find "$PROJECT_ROOT/certifications" -name "evidence.json" -print0 2>/dev/null)

    if [ "${evidence_count}" -eq 0 ]; then
        log_warning "No evidence.json files found in certifications/"
    fi

    echo ""

    # Validate models.csv
    echo "Validating models.csv..."
    local models_csv="$PROJECT_ROOT/docs/certifications/models.csv"

    if [ -f "${models_csv}" ]; then
        validate_models_csv "$models_csv" || true
    else
        log_warning "models.csv not found at $models_csv"
    fi

    echo ""
    echo "=== Validation Summary ==="
    echo "Errors:   $ERRORS"
    echo "Warnings: $WARNINGS"

    if [ "${ERRORS}" -gt 0 ]; then
        echo -e "${RED}FAILED${NC}: $ERRORS error(s) found"
        exit 1
    else
        echo -e "${GREEN}PASSED${NC}: All validations successful"
        exit 0
    fi
}

main "$@"
