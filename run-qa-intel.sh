#!/usr/bin/env bash
# run-qa-intel.sh — Three-tier fail-fast QA pipeline for Intel
#
# Usage:
#   ./run-qa-intel.sh <playbook-name>                  # single model, full pipeline
#   ./run-qa-intel.sh --batch "<glob>"                  # batch: parallel per tier
#   ./run-qa-intel.sh --tier <tier> <playbook-name>     # single tier only
#
# Tiers (executed in order, each gates the next):
#   dim-smoke  — metadata-only, <30s, no inference
#   smoke      — 3 tests, SafeTensors + CPU only, ~2-5 min
#   mvp        — full test matrix (CPU-only, --no-gpu), ~15-60 min
#
# Flags always applied:
#   --no-gpu              (CPU-only machine, no ROCm)
#   --failure-policy stop-on-p0  (Jidoka: stop on gateway failures)
set -euo pipefail

REPO_DIR="$HOME/src/apr-model-qa-playbook"
EVIDENCE_DIR="$HOME/data/qa-evidence"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
MAX_PARALLEL_SMALL=4    # ≤3B models
MAX_PARALLEL_MEDIUM=2   # 7B-14B models
MAX_PARALLEL_LARGE=1    # >14B models

# ── Colors ──────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log()  { echo -e "${CYAN}[qa]${NC} $*"; }
ok()   { echo -e "${GREEN}[ok]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; }
die()  { fail "$@"; exit 1; }

# ── Helpers ─────────────────────────────────────────────────────

playbook_file() {
  local name="$1"
  echo "${REPO_DIR}/playbooks/models/${name}.playbook.yaml"
}

# Derive model short name from playbook name (strip tier suffix)
model_short_name() {
  local name="$1"
  echo "$name" | sed -E 's/-(mvp|smoke|quick|standard|deep|dim-smoke)$//'
}

# Estimate model size bucket from name (for parallelism budget)
size_bucket() {
  local name="$1"
  case "$name" in
    *-0.5b-*|*-0.6b-*|*-1b-*|*-1.1b-*|*-1.3b-*|*-1.5b-*|*-1.6b-*|*-1.7b-*) echo "small" ;;
    *-2b-*|*-3b-*|*-4b-*)                                                      echo "small" ;;
    *-6b-*|*-7b-*|*-8b-*|*-9b-*)                                               echo "medium" ;;
    *-12b-*|*-13b-*|*-14b-*|*-15b-*)                                           echo "medium" ;;
    *)                                                                          echo "large" ;;
  esac
}

max_parallel_for() {
  case "$(size_bucket "$1")" in
    small)  echo "$MAX_PARALLEL_SMALL" ;;
    medium) echo "$MAX_PARALLEL_MEDIUM" ;;
    large)  echo "$MAX_PARALLEL_LARGE" ;;
  esac
}

# Extract HF repo from playbook YAML (for certify --tier dim-smoke)
hf_repo_from_playbook() {
  local pb="$1"
  grep 'hf_repo:' "$pb" | head -1 | sed 's/.*hf_repo:\s*["'\'']\?\([^"'\'']*\)["'\'']\?.*/\1/'
}

# ── Preparation ────────────────────────────────────────────────

prepare() {
  log "Preparing workspace..."
  cd "$REPO_DIR"
  git pull --rebase origin main 2>/dev/null || true

  # Clean stale output dir (causes "File exists" errors)
  rm -rf output

  # Build apr-qa
  log "Building apr-qa..."
  cargo build --release --bin apr-qa 2>&1 | tail -1
  ok "apr-qa built"

  # Regenerate lock file
  cargo run --release --bin apr-qa -- lock-playbooks 2>/dev/null || true
}

# ── Tier execution ──────────────────────────────────────────────

run_dim_smoke() {
  local name="$1"
  local pb
  pb="$(playbook_file "$name")"
  local model_name
  model_name="$(model_short_name "$name")"

  log "Tier 1 (dim-smoke): ${BOLD}${model_name}${NC}"

  local hf_repo
  hf_repo="$(hf_repo_from_playbook "$pb")"
  if [ -z "$hf_repo" ]; then
    die "Cannot extract hf_repo from $pb"
  fi

  rm -rf output
  if cargo run --release --bin apr-qa -- certify \
      --tier dim-smoke \
      --fail-fast \
      "$hf_repo" 2>&1; then
    ok "dim-smoke PASS: ${model_name}"
    return 0
  else
    fail "dim-smoke FAIL: ${model_name}"
    return 1
  fi
}

run_smoke() {
  local name="$1"
  local model_name
  model_name="$(model_short_name "$name")"
  local smoke_name="${model_name}-smoke"
  local smoke_pb
  smoke_pb="$(playbook_file "$smoke_name")"

  log "Tier 2 (smoke): ${BOLD}${model_name}${NC}"

  rm -rf output

  if [ -f "$smoke_pb" ]; then
    # Smoke playbook exists — use it
    if cargo run --release --bin apr-qa -- run "$smoke_pb" \
        --no-gpu \
        --failure-policy stop-on-p0 \
        --workers 1 \
        --timeout 120000 2>&1; then
      ok "smoke PASS: ${model_name}"
      return 0
    else
      fail "smoke FAIL: ${model_name}"
      return 1
    fi
  else
    # No smoke playbook — run MVP playbook with smoke-tier settings
    # (single worker, shorter timeout, fail-fast)
    local mvp_pb
    mvp_pb="$(playbook_file "$name")"
    if [ ! -f "$mvp_pb" ]; then
      fail "smoke FAIL: no playbook for ${model_name}"
      return 1
    fi
    if cargo run --release --bin apr-qa -- run "$mvp_pb" \
        --no-gpu \
        --failure-policy fail-fast \
        --workers 1 \
        --timeout 120000 2>&1; then
      ok "smoke PASS: ${model_name}"
      return 0
    else
      fail "smoke FAIL: ${model_name}"
      return 1
    fi
  fi
}

run_mvp() {
  local name="$1"
  local pb
  pb="$(playbook_file "$name")"
  local model_name
  model_name="$(model_short_name "$name")"
  local evidence_out="${EVIDENCE_DIR}/${model_name}/${TIMESTAMP}"

  log "Tier 3 (mvp): ${BOLD}${model_name}${NC}"

  if [ ! -f "$pb" ]; then
    die "Playbook not found: $pb"
  fi

  rm -rf output
  mkdir -p "$evidence_out"

  if cargo run --release --bin apr-qa -- run "$pb" \
      --no-gpu \
      --failure-policy stop-on-p0 \
      --workers 4 \
      --timeout 300000 2>&1 | tee "${evidence_out}/run.log"; then
    ok "mvp PASS: ${model_name}"
  else
    fail "mvp FAIL: ${model_name}"
  fi

  # Copy evidence artifacts
  if [ -d output ]; then
    cp -r output/* "$evidence_out/" 2>/dev/null || true
  fi
  # Copy evidence JSON if present
  for f in output/evidence.json certifications/*/evidence.json; do
    [ -f "$f" ] && cp "$f" "$evidence_out/" 2>/dev/null || true
  done

  log "Evidence saved to ${evidence_out}"
}

# ── Full pipeline (single model) ───────────────────────────────

run_full_pipeline() {
  local name="$1"
  local start_time
  start_time="$(date +%s)"
  local model_name
  model_name="$(model_short_name "$name")"

  log "=== Full pipeline: ${BOLD}${model_name}${NC} ==="

  # Tier 1: dim-smoke
  if ! run_dim_smoke "$name"; then
    die "Pipeline stopped: dim-smoke failed for ${model_name}"
  fi

  # Tier 2: smoke
  if ! run_smoke "$name"; then
    die "Pipeline stopped: smoke failed for ${model_name}"
  fi

  # Tier 3: mvp
  run_mvp "$name"

  local elapsed=$(( $(date +%s) - start_time ))
  local min=$(( elapsed / 60 ))
  local sec=$(( elapsed % 60 ))
  ok "=== Pipeline complete: ${model_name} (${min}m ${sec}s) ==="
}

# ── Batch mode ──────────────────────────────────────────────────

run_batch() {
  local glob_pattern="$1"
  local start_time
  start_time="$(date +%s)"

  # Find matching playbook names
  local playbooks=()
  for pb in ${REPO_DIR}/playbooks/models/${glob_pattern}.playbook.yaml; do
    [ -f "$pb" ] || continue
    local base
    base="$(basename "$pb" .playbook.yaml)"
    playbooks+=("$base")
  done

  if [ ${#playbooks[@]} -eq 0 ]; then
    die "No playbooks match pattern: ${glob_pattern}"
  fi

  log "Batch: ${#playbooks[@]} models matching '${glob_pattern}'"
  for p in "${playbooks[@]}"; do
    log "  - $p"
  done

  # ── Tier 1: dim-smoke (all in parallel) ──────────────────────
  log ""
  log "=== ${BOLD}Tier 1: dim-smoke${NC} (${#playbooks[@]} models) ==="
  local failed=()
  local pids=()
  local names=()

  for name in "${playbooks[@]}"; do
    (
      run_dim_smoke "$name" > "/tmp/qa-dim-smoke-${name}.log" 2>&1
    ) &
    pids+=($!)
    names+=("$name")
  done

  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      failed+=("${names[$i]}")
      fail "dim-smoke FAIL: ${names[$i]}"
      cat "/tmp/qa-dim-smoke-${names[$i]}.log" | tail -20
    else
      ok "dim-smoke PASS: ${names[$i]}"
    fi
  done

  if [ ${#failed[@]} -gt 0 ]; then
    fail "=== dim-smoke gate failed for ${#failed[@]} model(s) ==="
    for f in "${failed[@]}"; do fail "  - $f"; done
    die "Stopping batch: fix dim-smoke failures before proceeding"
  fi
  ok "=== Tier 1 complete: all ${#playbooks[@]} models passed dim-smoke ==="

  # ── Tier 2: smoke (parallel, respecting size budget) ─────────
  log ""
  log "=== ${BOLD}Tier 2: smoke${NC} (${#playbooks[@]} models) ==="
  failed=()
  local running=0
  local max_par
  # Use the largest model's budget for the batch
  max_par="$(max_parallel_for "${playbooks[0]}")"

  for name in "${playbooks[@]}"; do
    max_par="$(max_parallel_for "$name")"
    while [ "$running" -ge "$max_par" ]; do
      wait -n 2>/dev/null || true
      running=$((running - 1))
    done

    (
      run_smoke "$name" > "/tmp/qa-smoke-${name}.log" 2>&1
    ) &
    running=$((running + 1))
  done
  wait

  # Check results
  for name in "${playbooks[@]}"; do
    if ! grep -q '\[ok\]' "/tmp/qa-smoke-${name}.log" 2>/dev/null; then
      failed+=("$name")
      fail "smoke FAIL: ${name}"
      cat "/tmp/qa-smoke-${name}.log" | tail -20
    else
      ok "smoke PASS: ${name}"
    fi
  done

  if [ ${#failed[@]} -gt 0 ]; then
    fail "=== smoke gate failed for ${#failed[@]} model(s) ==="
    for f in "${failed[@]}"; do fail "  - $f"; done
    die "Stopping batch: fix smoke failures before proceeding"
  fi
  ok "=== Tier 2 complete: all ${#playbooks[@]} models passed smoke ==="

  # ── Tier 3: mvp (parallel, respecting size budget) ───────────
  log ""
  log "=== ${BOLD}Tier 3: mvp${NC} (${#playbooks[@]} models) ==="
  running=0

  for name in "${playbooks[@]}"; do
    max_par="$(max_parallel_for "$name")"
    while [ "$running" -ge "$max_par" ]; do
      wait -n 2>/dev/null || true
      running=$((running - 1))
    done

    (
      run_mvp "$name" > "/tmp/qa-mvp-${name}.log" 2>&1
    ) &
    running=$((running + 1))
  done
  wait

  # Report results
  local passed=0
  local total=${#playbooks[@]}
  for name in "${playbooks[@]}"; do
    if grep -q '\[ok\] mvp PASS' "/tmp/qa-mvp-${name}.log" 2>/dev/null; then
      ok "mvp PASS: ${name}"
      passed=$((passed + 1))
    else
      fail "mvp FAIL: ${name}"
      tail -20 "/tmp/qa-mvp-${name}.log" 2>/dev/null
    fi
  done

  local elapsed=$(( $(date +%s) - start_time ))
  local min=$(( elapsed / 60 ))
  local sec=$(( elapsed % 60 ))
  log ""
  log "=== ${BOLD}Batch complete${NC}: ${passed}/${total} passed (${min}m ${sec}s) ==="
}

# ── Main ────────────────────────────────────────────────────────

usage() {
  echo "Usage:"
  echo "  $0 <playbook-name>                   Full pipeline (dim-smoke → smoke → mvp)"
  echo "  $0 --batch \"<glob>\"                   Batch: all matching, parallel per tier"
  echo "  $0 --tier <tier> <playbook-name>      Single tier only (dim-smoke|smoke|mvp)"
  echo "  $0 --dry-run <playbook-name>          Show what would run without executing"
  echo ""
  echo "Examples:"
  echo "  $0 qwen2.5-coder-0.5b-mvp"
  echo "  $0 --batch \"qwen2.5-*-mvp\""
  echo "  $0 --tier dim-smoke qwen2.5-coder-0.5b-mvp"
  exit 1
}

main() {
  [ $# -eq 0 ] && usage

  local mode="full"
  local tier=""
  local target=""
  local dry_run=false

  while [ $# -gt 0 ]; do
    case "$1" in
      --batch)
        mode="batch"
        target="$2"
        shift 2
        ;;
      --tier)
        mode="tier"
        tier="$2"
        shift 2
        ;;
      --dry-run)
        dry_run=true
        shift
        ;;
      --help|-h)
        usage
        ;;
      *)
        target="$1"
        shift
        ;;
    esac
  done

  [ -z "$target" ] && usage

  if $dry_run; then
    log "DRY RUN — would execute:"
    log "  mode:   $mode"
    log "  tier:   ${tier:-all (dim-smoke → smoke → mvp)}"
    log "  target: $target"
    exit 0
  fi

  prepare

  case "$mode" in
    batch)
      run_batch "$target"
      ;;
    tier)
      case "$tier" in
        dim-smoke) run_dim_smoke "$target" ;;
        smoke)     run_smoke "$target" ;;
        mvp)       run_mvp "$target" ;;
        *)         die "Unknown tier: $tier (use dim-smoke, smoke, mvp)" ;;
      esac
      ;;
    full)
      run_full_pipeline "$target"
      ;;
  esac
}

main "$@"

