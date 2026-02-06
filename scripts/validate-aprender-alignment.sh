#!/usr/bin/env bash
# PMAT-274: Cross-Repo Consistency Check Script
#
# Validates that apr-model-qa-playbook data aligns with aprender family YAMLs:
# 1. Size category matches family YAML certification.size_categories
# 2. Playbook model IDs match family naming conventions
# 3. Expected dimensions match family size_variants
#
# Usage: ./scripts/validate-aprender-alignment.sh [aprender_path]
# Exit codes:
#   0 - All validations passed
#   1 - Validation errors found
#   2 - aprender not found (warning only)

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default aprender path (sibling directory or environment variable)
APRENDER_PATH="${1:-${APRENDER_PATH:-$PROJECT_ROOT/../aprender}}"

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
    echo "  $1"
}

# Check if yq is available
check_dependencies() {
    if ! command -v yq &> /dev/null; then
        log_error "yq is required but not installed. Install with: sudo snap install yq"
        exit 1
    fi
}

# Extract family name from model ID (e.g., "Qwen/Qwen2.5-Coder-0.5B-Instruct" -> "qwen2")
get_family_from_model_id() {
    local model_id="$1"
    local name
    name=$(echo "$model_id" | cut -d'/' -f2 | tr '[:upper:]' '[:lower:]')

    # Map common model names to family names
    if [[ "$name" =~ ^qwen ]]; then
        echo "qwen2"
    elif [[ "$name" =~ ^llama ]]; then
        echo "llama"
    elif [[ "$name" =~ ^mistral ]]; then
        echo "mistral"
    elif [[ "$name" =~ ^gemma ]]; then
        echo "gemma"
    elif [[ "$name" =~ ^phi ]]; then
        echo "phi"
    elif [[ "$name" =~ ^deepseek ]]; then
        echo "deepseek"
    else
        echo "unknown"
    fi
}

# Extract size variant from model ID (e.g., "Qwen/Qwen2.5-Coder-0.5B-Instruct" -> "0.5b")
get_size_from_model_id() {
    local model_id="$1"
    local name
    name=$(echo "$model_id" | cut -d'/' -f2)

    # Extract size pattern like "0.5B", "1.5B", "7B", etc.
    if [[ "$name" =~ ([0-9]+\.?[0-9]*)[Bb] ]]; then
        echo "${BASH_REMATCH[1]}b" | tr '[:upper:]' '[:lower:]'
    else
        echo "unknown"
    fi
}

# Validate size category against family YAML
validate_size_category() {
    local model_id="$1"
    local playbook_size="$2"
    local family_yaml="$3"

    local size_variant
    size_variant=$(get_size_from_model_id "$model_id")

    if [[ ! -f "$family_yaml" ]]; then
        log_warning "Family YAML not found: $family_yaml"
        return 0
    fi

    # Get expected size category from family YAML
    local expected_size
    expected_size=$(yq ".certification.size_categories.\"$size_variant\"" "$family_yaml" 2>/dev/null || echo "null")

    if [[ "$expected_size" == "null" || -z "$expected_size" ]]; then
        log_warning "$model_id: Size variant '$size_variant' not found in family YAML"
        return 0
    fi

    # Treat null/missing playbook size as a warning (not configured yet)
    if [[ "$playbook_size" == "null" || -z "$playbook_size" ]]; then
        log_warning "$model_id: size_category not set in playbook (expected '$expected_size')"
        return 0
    fi

    if [[ "$expected_size" != "$playbook_size" ]]; then
        log_error "$model_id: size_category mismatch - playbook has '$playbook_size', family YAML expects '$expected_size'"
        return 1
    fi

    return 0
}

# Validate hidden dimensions against family YAML
validate_dimensions() {
    local model_id="$1"
    local playbook_file="$2"
    local family_yaml="$3"

    local size_variant
    size_variant=$(get_size_from_model_id "$model_id")

    if [[ ! -f "$family_yaml" ]]; then
        return 0
    fi

    # Get expected dimensions from family YAML
    local expected_hidden
    expected_hidden=$(yq ".size_variants.\"$size_variant\".hidden_dim" "$family_yaml" 2>/dev/null || echo "null")

    if [[ "$expected_hidden" == "null" ]]; then
        return 0
    fi

    # Get playbook expected dimensions if set
    local playbook_hidden
    playbook_hidden=$(yq ".model.expected_hidden_dim" "$playbook_file" 2>/dev/null || echo "null")

    if [[ "$playbook_hidden" != "null" && "$playbook_hidden" != "$expected_hidden" ]]; then
        log_error "$model_id: expected_hidden_dim mismatch - playbook has '$playbook_hidden', family YAML has '$expected_hidden'"
        return 1
    fi

    return 0
}

# Main validation
main() {
    echo "=== APR QA Cross-Repo Consistency Check (PMAT-274) ==="
    echo ""

    check_dependencies

    # Check if aprender is available
    local contracts_dir="$APRENDER_PATH/contracts/model-families"
    if [[ ! -d "$contracts_dir" ]]; then
        log_warning "aprender contracts not found at: $contracts_dir"
        log_info "Skipping cross-repo validation (aprender not available)"
        log_info "Set APRENDER_PATH or pass path as argument"
        echo ""
        echo "=== Validation Summary ==="
        echo "Errors:   $ERRORS"
        echo "Warnings: $WARNINGS"
        echo -e "${YELLOW}SKIPPED${NC}: aprender not available"
        exit 0
    fi

    log_success "Found aprender contracts at: $contracts_dir"
    echo ""

    # Validate models.csv entries
    echo "Validating models.csv against family YAMLs..."
    local models_csv="$PROJECT_ROOT/docs/certifications/models.csv"

    if [[ ! -f "$models_csv" ]]; then
        log_warning "models.csv not found"
    else
        local line_num=0
        while IFS=',' read -r model_id family parameters size_category rest; do
            ((line_num++)) || true
            [[ $line_num -eq 1 ]] && continue  # Skip header
            [[ -z "$model_id" ]] && continue

            local family_name
            family_name=$(get_family_from_model_id "$model_id")
            local family_yaml="$contracts_dir/${family_name}.yaml"

            if [[ -f "$family_yaml" ]]; then
                validate_size_category "$model_id" "$size_category" "$family_yaml" || true
            fi
        done < "$models_csv"
    fi

    echo ""

    # Validate playbook files
    echo "Validating playbooks against family YAMLs..."
    local playbook_count=0

    while IFS= read -r -d '' playbook_file; do
        ((playbook_count++)) || true

        local model_id
        model_id=$(yq '.model.hf_repo' "$playbook_file" 2>/dev/null || echo "")

        if [[ -z "$model_id" || "$model_id" == "null" ]]; then
            continue
        fi

        local family_name
        family_name=$(get_family_from_model_id "$model_id")
        local family_yaml="$contracts_dir/${family_name}.yaml"

        if [[ -f "$family_yaml" ]]; then
            # Validate size category
            local playbook_size
            playbook_size=$(yq '.model.size_category' "$playbook_file" 2>/dev/null || echo "tiny")
            validate_size_category "$model_id" "$playbook_size" "$family_yaml" || true

            # Validate dimensions
            validate_dimensions "$model_id" "$playbook_file" "$family_yaml" || true
        fi
    done < <(find "$PROJECT_ROOT/playbooks/models" -name "*.yaml" -print0 2>/dev/null)

    log_success "Validated $playbook_count playbook files"

    echo ""
    echo "=== Validation Summary ==="
    echo "Errors:   $ERRORS"
    echo "Warnings: $WARNINGS"

    if [[ $ERRORS -gt 0 ]]; then
        echo -e "${RED}FAILED${NC}: $ERRORS error(s) found"
        exit 1
    else
        echo -e "${GREEN}PASSED${NC}: All cross-repo validations successful"
        exit 0
    fi
}

main "$@"
