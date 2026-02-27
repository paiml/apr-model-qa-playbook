#!/usr/bin/env bash
# run-qa-loop.sh — Continuous model qualification loop
#
# Reads qualification-manifest.yaml, claims the next pending model,
# runs the three-tier fail-fast pipeline, commits evidence, and loops.
# Coordinates across machines via git push/pull.
#
# Usage:
#   ./run-qa-loop.sh local              # run on local workstation (GPU enabled)
#   ./run-qa-loop.sh intel              # run on Intel Mac Pro (CPU only)
#   ./run-qa-loop.sh local --once       # single model, then exit
#   ./run-qa-loop.sh local --retry-failed  # re-queue failed models, then loop
#   ./run-qa-loop.sh local --dry-run    # show what would run without executing
#
# Designed to run in tmux for persistence:
#   tmux new -d -s qa './run-qa-loop.sh local'
#   ssh intel "cd ~/src/apr-model-qa-playbook && tmux new -d -s qa './run-qa-loop.sh intel'"
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
MANIFEST="${REPO_DIR}/qualification-manifest.yaml"
EVIDENCE_BASE="${HOME}/data/qa-evidence"
COOLDOWN=30                 # seconds between models
MAX_PUSH_RETRIES=5
LOCK_TIMEOUT_MIN=120        # stale "running" claim timeout

# ── Colors ────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

log()  { echo -e "${CYAN}[loop]${NC} $(date +%H:%M:%S) $*"; }
ok()   { echo -e "${GREEN}[ok]${NC}   $(date +%H:%M:%S) $*"; }
warn() { echo -e "${YELLOW}[warn]${NC} $(date +%H:%M:%S) $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $(date +%H:%M:%S) $*"; }

# ── Argument parsing ──────────────────────────────────────────────

MACHINE=""
RUN_ONCE=false
RETRY_FAILED=false
DRY_RUN=false

parse_args() {
  [[ $# -eq 0 ]] && {
    echo "Usage: $0 <local|intel> [--once] [--retry-failed] [--dry-run]"
    exit 1
  }

  MACHINE="$1"; shift

  case "$MACHINE" in
    local|intel) ;;
    *) echo "Unknown machine: $MACHINE (use 'local' or 'intel')"; exit 1 ;;
  esac

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --once)         RUN_ONCE=true; shift ;;
      --retry-failed) RETRY_FAILED=true; shift ;;
      --dry-run)      DRY_RUN=true; shift ;;
      *) echo "Unknown flag: $1"; exit 1 ;;
    esac
  done
}

# ── GPU flag ──────────────────────────────────────────────────────

gpu_flag() {
  if [[ "$MACHINE" == "intel" ]]; then
    echo "--no-gpu"
  else
    echo ""
  fi
}

# ── Git coordination ─────────────────────────────────────────────

git_sync() {
  cd "$REPO_DIR"
  # Stash any local changes before pulling
  local stashed=false
  if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
    git stash push -q -m "qa-loop-autostash" 2>/dev/null && stashed=true
  fi

  git pull --rebase origin main 2>/dev/null || {
    warn "git pull failed, retrying..."
    sleep 2
    git pull --rebase origin main 2>/dev/null || warn "pull still failed, continuing with local state"
  }

  if $stashed; then
    git stash pop -q 2>/dev/null || warn "stash pop conflict — manual resolution may be needed"
  fi
}

git_push_with_retry() {
  local retries=0
  while [[ $retries -lt $MAX_PUSH_RETRIES ]]; do
    if git push origin main 2>/dev/null; then
      return 0
    fi
    retries=$((retries + 1))
    warn "Push failed (attempt ${retries}/${MAX_PUSH_RETRIES}), rebasing..."
    sleep $((retries * 2))
    git pull --rebase origin main 2>/dev/null || true
  done
  fail "Push failed after ${MAX_PUSH_RETRIES} attempts"
  return 1
}

# ── Manifest operations ──────────────────────────────────────────

# Find next pending model for this machine.
# Returns playbook name or empty string.
next_pending() {
  local machine="$1"

  # Parse manifest: find first entry with status: pending and compatible affinity
  python3 -c "
import yaml, sys

with open('$MANIFEST') as f:
    data = yaml.safe_load(f)

for entry in data.get('queue', []):
    if entry.get('status') != 'pending':
        continue
    affinity = entry.get('affinity', 'any')
    machine = '$machine'
    if affinity == 'intel-only' and machine != 'intel':
        continue
    # 'any' and 'prefer-intel' work on both machines
    print(entry['playbook'])
    sys.exit(0)
" 2>/dev/null || true
}

# Update a model's status in the manifest
update_status() {
  local playbook="$1"
  local new_status="$2"
  local extra_fields="${3:-}"  # optional: "claimed_by: intel, claimed_at: 2026..."

  python3 -c "
import yaml

with open('$MANIFEST') as f:
    data = yaml.safe_load(f)

for entry in data.get('queue', []):
    if entry.get('playbook') == '$playbook':
        entry['status'] = '$new_status'
        # Remove transient fields when completing
        if '$new_status' in ('certified', 'failed', 'pending'):
            entry.pop('claimed_by', None)
            entry.pop('claimed_at', None)
        break

# Write back preserving comments (use block style)
with open('$MANIFEST', 'w') as f:
    yaml.dump(data, f, default_flow_style=False, sort_keys=False, width=120)
"
}

# Claim a model: set status to running with machine info
claim_model() {
  local playbook="$1"
  local timestamp
  timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  python3 -c "
import yaml

with open('$MANIFEST') as f:
    data = yaml.safe_load(f)

for entry in data.get('queue', []):
    if entry.get('playbook') == '$playbook':
        entry['status'] = 'running'
        entry['claimed_by'] = '$MACHINE'
        entry['claimed_at'] = '$timestamp'
        break

with open('$MANIFEST', 'w') as f:
    yaml.dump(data, f, default_flow_style=False, sort_keys=False, width=120)
"
}

# Mark model as failed with reason
mark_failed() {
  local playbook="$1"
  local reason="$2"

  python3 -c "
import yaml

with open('$MANIFEST') as f:
    data = yaml.safe_load(f)

for entry in data.get('queue', []):
    if entry.get('playbook') == '$playbook':
        entry['status'] = 'failed'
        entry['failure_reason'] = '''$reason'''
        entry.pop('claimed_by', None)
        entry.pop('claimed_at', None)
        break

with open('$MANIFEST', 'w') as f:
    yaml.dump(data, f, default_flow_style=False, sort_keys=False, width=120)
"
}

# Reset all failed models to pending
retry_failed_models() {
  python3 -c "
import yaml

with open('$MANIFEST') as f:
    data = yaml.safe_load(f)

count = 0
for entry in data.get('queue', []):
    if entry.get('status') == 'failed':
        entry['status'] = 'pending'
        entry.pop('failure_reason', None)
        entry.pop('claimed_by', None)
        entry.pop('claimed_at', None)
        count += 1

with open('$MANIFEST', 'w') as f:
    yaml.dump(data, f, default_flow_style=False, sort_keys=False, width=120)

print(f'{count} models reset from failed to pending')
"
}

# Count models by status
manifest_summary() {
  python3 -c "
import yaml
from collections import Counter

with open('$MANIFEST') as f:
    data = yaml.safe_load(f)

counts = Counter(e.get('status', 'unknown') for e in data.get('queue', []))
total = sum(counts.values())
print(f\"  Total: {total}\")
for status in ['certified', 'running', 'pending', 'failed', 'skipped']:
    if counts.get(status, 0) > 0:
        print(f\"  {status.capitalize():12s} {counts[status]}\")
"
}

# ── Three-tier pipeline ──────────────────────────────────────────

run_pipeline() {
  local playbook_name="$1"
  local pb_file="${REPO_DIR}/playbooks/models/${playbook_name}.playbook.yaml"
  local model_name="${playbook_name%-mvp}"
  local gpu_opt
  gpu_opt="$(gpu_flag)"
  local timestamp
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  local evidence_dir="${EVIDENCE_BASE}/${model_name}/${timestamp}"
  local start_time
  start_time="$(date +%s)"

  if [[ ! -f "$pb_file" ]]; then
    fail "Playbook not found: ${pb_file}"
    return 1
  fi

  mkdir -p "$evidence_dir"

  # Extract HF repo for dim-smoke
  local hf_repo
  hf_repo="$(grep 'hf_repo:' "$pb_file" | head -1 | sed 's/.*hf_repo:\s*["'\'']\?\([^"'\'']*\)["'\'']\?.*/\1/')"

  cd "$REPO_DIR"

  # ── Tier 1: dim-smoke ──────────────────────────────────────────
  log "Tier 1 (dim-smoke): ${BOLD}${model_name}${NC}"
  rm -rf output

  if [[ -n "$hf_repo" ]]; then
    if ! cargo run --release --bin apr-qa -- certify \
        --tier dim-smoke \
        --fail-fast \
        "$hf_repo" 2>&1 | tee "${evidence_dir}/dim-smoke.log"; then
      fail "dim-smoke FAIL: ${model_name}"
      return 1
    fi
  else
    warn "No hf_repo in playbook, skipping dim-smoke"
  fi
  ok "dim-smoke PASS: ${model_name}"

  # ── Tier 2: smoke ──────────────────────────────────────────────
  log "Tier 2 (smoke): ${BOLD}${model_name}${NC}"
  rm -rf output

  local smoke_pb="${REPO_DIR}/playbooks/models/${model_name}-smoke.playbook.yaml"
  if [[ -f "$smoke_pb" ]]; then
    if ! cargo run --release --bin apr-qa -- run "$smoke_pb" \
        ${gpu_opt} \
        --failure-policy stop-on-p0 \
        --workers 1 \
        --timeout 120000 2>&1 | tee "${evidence_dir}/smoke.log"; then
      fail "smoke FAIL: ${model_name}"
      return 1
    fi
  elif [[ -n "$hf_repo" ]]; then
    if ! cargo run --release --bin apr-qa -- certify \
        --tier smoke \
        --fail-fast \
        "$hf_repo" 2>&1 | tee "${evidence_dir}/smoke.log"; then
      fail "smoke FAIL: ${model_name}"
      return 1
    fi
  else
    warn "No smoke playbook and no hf_repo, skipping smoke"
  fi
  ok "smoke PASS: ${model_name}"

  # ── Tier 3: mvp ────────────────────────────────────────────────
  log "Tier 3 (mvp): ${BOLD}${model_name}${NC}"
  rm -rf output

  if ! cargo run --release --bin apr-qa -- run "$pb_file" \
      ${gpu_opt} \
      --failure-policy stop-on-p0 \
      --workers 4 \
      --timeout 300000 2>&1 | tee "${evidence_dir}/mvp.log"; then
    fail "mvp FAIL: ${model_name}"
    return 1
  fi

  # Copy evidence artifacts
  if [[ -d output ]]; then
    cp -r output/* "$evidence_dir/" 2>/dev/null || true
  fi

  # Also copy into repo certifications/ for git commit
  local cert_dir="${REPO_DIR}/certifications/${model_name}"
  mkdir -p "$cert_dir"
  for f in output/evidence.json "${evidence_dir}/evidence.json"; do
    if [[ -f "$f" ]]; then
      cp "$f" "$cert_dir/evidence.json"
      break
    fi
  done

  local elapsed=$(( $(date +%s) - start_time ))
  local min=$(( elapsed / 60 ))
  local sec=$(( elapsed % 60 ))
  ok "Pipeline complete: ${BOLD}${model_name}${NC} (${min}m ${sec}s)"
  return 0
}

# ── Build step ────────────────────────────────────────────────────

ensure_built() {
  cd "$REPO_DIR"
  log "Building apr-qa..."
  cargo build --release --bin apr-qa 2>&1 | tail -3
  ok "apr-qa built"

  # Regenerate lock file (suppress errors)
  cargo run --release --bin apr-qa -- lock-playbooks 2>/dev/null || true
}

# ── Main loop ─────────────────────────────────────────────────────

run_loop() {
  local loop_count=0
  local total_certified=0

  log "=== Qualification loop starting on ${BOLD}${MACHINE}${NC} ==="
  log ""
  manifest_summary
  log ""

  if $RETRY_FAILED; then
    log "Resetting failed models to pending..."
    retry_failed_models
    git add "$MANIFEST"
    git commit -m "reset failed models to pending for retry" 2>/dev/null || true
    git_push_with_retry || true
  fi

  # Initial sync + build
  git_sync
  if ! $DRY_RUN; then
    ensure_built
  fi

  while true; do
    loop_count=$((loop_count + 1))

    # Sync manifest
    git_sync

    # Pick next model
    local next
    next="$(next_pending "$MACHINE")"

    if [[ -z "$next" ]]; then
      ok "=== No pending models for ${MACHINE}. Queue exhausted! ==="
      manifest_summary
      break
    fi

    log ""
    log "=== [${loop_count}] Next: ${BOLD}${next}${NC} ==="

    if $DRY_RUN; then
      log "DRY RUN — would run: ${next}"
      if $RUN_ONCE; then break; fi
      continue
    fi

    # Claim the model
    claim_model "$next"
    git add "$MANIFEST"
    git commit -m "qa: claim ${next} on ${MACHINE}" 2>/dev/null || true
    if ! git_push_with_retry; then
      warn "Could not push claim for ${next}, skipping to avoid conflict"
      git_sync  # reset to remote state
      sleep 5
      continue
    fi

    # Run the pipeline
    local pipeline_ok=true
    if ! run_pipeline "$next"; then
      pipeline_ok=false
    fi

    # Update manifest with result
    if $pipeline_ok; then
      update_status "$next" "certified"
      total_certified=$((total_certified + 1))

      # Commit evidence + manifest
      local model_name="${next%-mvp}"
      git add "$MANIFEST"
      git add "certifications/${model_name}/" 2>/dev/null || true
      git commit -m "qa: certify ${next} on ${MACHINE}

Three-tier pipeline (dim-smoke → smoke → mvp) passed.
Machine: ${MACHINE}" 2>/dev/null || true

    else
      # Capture failure reason from last log
      local reason="pipeline failed"
      local evidence_dir
      evidence_dir="$(ls -td "${EVIDENCE_BASE}/${next%-mvp}/"* 2>/dev/null | head -1)"
      if [[ -n "$evidence_dir" ]]; then
        # Get last non-empty log line
        for logfile in "${evidence_dir}/mvp.log" "${evidence_dir}/smoke.log" "${evidence_dir}/dim-smoke.log"; do
          if [[ -f "$logfile" ]]; then
            reason="$(tail -5 "$logfile" | grep -v '^$' | tail -1 | head -c 200)"
            break
          fi
        done
      fi

      mark_failed "$next" "$reason"
      git add "$MANIFEST"
      git commit -m "qa: FAIL ${next} on ${MACHINE}

${reason}" 2>/dev/null || true
    fi

    git_push_with_retry || warn "Could not push results for ${next}"

    # Status report
    log ""
    log "── Progress ──"
    manifest_summary
    log "  This session: ${total_certified} certified"

    if $RUN_ONCE; then
      log "Single-run mode, exiting."
      break
    fi

    # Cooldown between models
    log "Cooling down ${COOLDOWN}s..."
    sleep "$COOLDOWN"

    # Rebuild if source changed (git pull may have updated code)
    ensure_built
  done

  log ""
  ok "=== Loop finished: ${total_certified} models certified this session ==="
}

# ── Entry point ───────────────────────────────────────────────────

parse_args "$@"
run_loop
