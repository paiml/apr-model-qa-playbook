#!/usr/bin/env bash
# generate-manifest.sh — Bootstrap qualification-manifest.yaml from MVP playbooks
#
# Scans playbooks/models/*-mvp.playbook.yaml, determines model size from the
# playbook name, assigns machine affinity, and checks existing evidence to
# mark already-certified models.
#
# Usage:
#   ./generate-manifest.sh                    # writes qualification-manifest.yaml
#   ./generate-manifest.sh --stdout           # prints to stdout instead
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
PLAYBOOK_DIR="${REPO_DIR}/playbooks/models"
CERT_DIR="${REPO_DIR}/certifications"
OUTPUT="${REPO_DIR}/qualification-manifest.yaml"
TO_STDOUT=false

if [[ "${1:-}" == "--stdout" ]]; then
  TO_STDOUT=true
fi

# ── Size detection from playbook name ──────────────────────────────

size_from_name() {
  local name="$1"
  # Extract size pattern like "0.5b", "1.5b", "7b", "70b", "135m", "360m"
  local size
  size="$(echo "$name" | grep -oP '\d+[\._]?\d*[bm]' | tail -1)" || true

  if [[ -z "$size" ]]; then
    echo "unknown"
    return
  fi

  # Normalize: convert to lowercase, replace _ with .
  size="${size,,}"
  size="${size//_/.}"

  # Convert to approximate parameter count in billions
  local num
  if [[ "$size" == *m ]]; then
    num="$(echo "$size" | sed 's/m$//')"
    # Millions — convert to billions for comparison
    num="$(echo "$num" | awk '{printf "%.1f", $1/1000}')"
  else
    num="$(echo "$size" | sed 's/b$//')"
  fi

  # Bucket
  if awk "BEGIN {exit !($num <= 3.0)}"; then
    echo "small"
  elif awk "BEGIN {exit !($num <= 14.0)}"; then
    echo "medium"
  elif awk "BEGIN {exit !($num <= 34.0)}"; then
    echo "large"
  else
    echo "xlarge"
  fi
}

affinity_from_size() {
  case "$1" in
    small)   echo "any" ;;
    medium)  echo "any" ;;
    large)   echo "intel-only" ;;
    xlarge)  echo "intel-only" ;;
    *)       echo "any" ;;
  esac
}

ram_estimate() {
  local name="$1"
  local size
  size="$(echo "$name" | grep -oP '\d+[\._]?\d*[bm]' | tail -1)" || true

  if [[ -z "$size" ]]; then
    echo "2"
    return
  fi

  size="${size,,}"
  size="${size//_/.}"

  local num
  if [[ "$size" == *m ]]; then
    num="$(echo "$size" | sed 's/m$//')"
    num="$(echo "$num" | awk '{printf "%.1f", $1/1000}')"
  else
    num="$(echo "$size" | sed 's/b$//')"
  fi

  # Rough Q4 estimate: ~0.6 GB per billion params + 2 GB overhead
  echo "$num" | awk '{printf "%d", $1 * 0.6 + 2}'
}

# ── Check if already certified ─────────────────────────────────────

is_certified() {
  local playbook_name="$1"
  # Strip -mvp suffix to get model name
  local model_name="${playbook_name%-mvp}"

  # Check both naming conventions
  if [[ -f "${CERT_DIR}/${model_name}/evidence.json" ]] ||
     [[ -f "${CERT_DIR}/${playbook_name}/evidence.json" ]]; then
    return 0
  fi

  # Also check hyphenated variants (qwen2.5-coder-0.5b vs qwen2-5-coder-0-5b)
  local alt_name
  alt_name="$(echo "$model_name" | sed 's/\./-/g')"
  if [[ -f "${CERT_DIR}/${alt_name}/evidence.json" ]]; then
    return 0
  fi

  # Check for -instruct suffix variant
  if [[ -f "${CERT_DIR}/${model_name}-instruct/evidence.json" ]] ||
     [[ -f "${CERT_DIR}/${alt_name}-instruct/evidence.json" ]]; then
    return 0
  fi

  return 1
}

# ── Generate manifest ──────────────────────────────────────────────

generate() {
  cat <<'HEADER'
# qualification-manifest.yaml
#
# Git-based work queue for continuous model qualification.
# Both machines read this file, claim work via git commits, and
# update status on completion.
#
# Status values:
#   pending    — not yet started
#   running    — claimed by a machine (see claimed_by, claimed_at)
#   certified  — three-tier pipeline passed
#   failed     — pipeline failed (see failure_reason)
#   skipped    — intentionally skipped
#
# Affinity values:
#   any        — either machine can run this
#   intel-only — requires Intel (283 GB RAM) for large models

machines:
  local:
    max_ram_gb: 125
    has_gpu: true
    max_parallel: 2
  intel:
    max_ram_gb: 283
    has_gpu: false
    max_parallel: 4

queue:
HEADER

  # Collect all MVP playbooks, sort by size (smallest first)
  local -a small_models=()
  local -a medium_models=()
  local -a large_models=()
  local -a xlarge_models=()

  for pb in "${PLAYBOOK_DIR}"/*-mvp.playbook.yaml; do
    [[ -f "$pb" ]] || continue
    local name
    name="$(basename "$pb" .playbook.yaml)"
    local bucket
    bucket="$(size_from_name "$name")"

    case "$bucket" in
      small)   small_models+=("$name") ;;
      medium)  medium_models+=("$name") ;;
      large)   large_models+=("$name") ;;
      xlarge)  xlarge_models+=("$name") ;;
      *)       small_models+=("$name") ;;
    esac
  done

  # Output in priority order: small → medium → large → xlarge
  local section_header_printed=false

  emit_section() {
    local label="$1"
    shift
    local -a models=("$@")

    if [[ ${#models[@]} -eq 0 ]]; then
      return
    fi

    echo ""
    echo "  # ── ${label} ──"

    for name in "${models[@]}"; do
      local bucket
      bucket="$(size_from_name "$name")"
      local affinity
      affinity="$(affinity_from_size "$bucket")"
      local ram
      ram="$(ram_estimate "$name")"
      local status="pending"

      if is_certified "$name"; then
        status="certified"
      fi

      echo "  - playbook: ${name}"
      echo "    affinity: ${affinity}"
      echo "    ram_gb: ${ram}"
      echo "    status: ${status}"
    done
  }

  emit_section "Small models (≤3B) — either machine" "${small_models[@]}"
  emit_section "Medium models (4B-14B) — either machine" "${medium_models[@]}"
  emit_section "Large models (15B-34B) — Intel only" "${large_models[@]}"
  emit_section "XLarge models (35B+) — Intel only" "${xlarge_models[@]}"
}

if $TO_STDOUT; then
  generate
else
  generate > "$OUTPUT"
  echo "Generated ${OUTPUT}"
  echo ""
  # Summary
  total="$(grep -c 'playbook:' "$OUTPUT")"
  certified="$(grep -c 'status: certified' "$OUTPUT")"
  pending="$(grep -c 'status: pending' "$OUTPUT")"
  echo "Total: ${total} models"
  echo "  Certified: ${certified}"
  echo "  Pending:   ${pending}"
fi
