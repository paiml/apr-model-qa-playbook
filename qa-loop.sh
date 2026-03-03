#!/bin/sh
# QA Continuous Loop — systemd-managed service that claims and runs
# model qualification jobs from the SQLite queue.
# Design source: qa-loop.rs (rash transpilation target when rash matures)
#
# Usage:
#   QA_MACHINE=intel qa-loop.sh     # Direct (testing)
#   systemctl --user start qa-loop  # Via systemd (production)
#
# Environment:
#   QA_MACHINE  "intel" or "local" (default: intel)
#   QA_DB       SQLite DB path (default: ~/data/qa-jobs/qa-jobs.db)
#   QA_REPO     Playbook repo path (default: ~/src/apr-model-qa-playbook)
set -eu

QA_MACHINE="${QA_MACHINE:-intel}"
QA_DB="${QA_DB:-/home/noah/data/qa-jobs/qa-jobs.db}"
QA_REPO="${QA_REPO:-/home/noah/src/apr-model-qa-playbook}"
COOLDOWN=30
DISK_THRESHOLD=90   # Refuse new jobs above this % usage
CURRENT_PLAYBOOK="" # Tracks in-flight job for trap cleanup

# Route DB commands through SSH when running on local machine
if [ "$QA_MACHINE" = "local" ]; then
    JOBS_CMD="ssh intel /home/noah/src/apr-model-qa-playbook/qa-jobs.sh"
else
    JOBS_CMD="$QA_REPO/qa-jobs.sh"
fi

echo "=== QA loop starting on $QA_MACHINE ==="
echo "DB: $QA_DB"
echo "Repo: $QA_REPO"
echo "Jobs CMD: $JOBS_CMD"

# ── Ensure apr is built ─────────────────────────────────────────
ensure_built() {
    build_script="/home/noah/src/aprender/build-apr.sh"
    if [ -f "$build_script" ]; then
        echo "Building apr from aprender..."
        bash "$build_script" || echo "WARNING: apr build failed (non-fatal, using existing binary)"
    fi

    if ! command -v apr >/dev/null 2>&1; then
        echo "ERROR: apr binary not found. Run build-apr.sh first."
        exit 1
    fi

    echo "apr version: $(apr --version 2>&1 || echo unknown)"
}

# ── Extract pass rate from evidence JSON ────────────────────────
extract_pass_rate() {
    evidence_file="$1"
    python3 -c "
import json, sys
data = json.load(open(sys.argv[1]))
if isinstance(data, list):
    total = len(data)
    passed = sum(1 for e in data if e.get('outcome') == 'Corroborated')
    print(int(100 * passed / total) if total > 0 else 0)
else:
    print(0)
" "$evidence_file" 2>/dev/null || echo "0"
}

# ── Clean up heavy artifacts after EVERY run ─────────────────────
# CONTRACT: evidence.json + run.log are preserved. Everything else
# (conversions/, workspace/, *.apr, *.gguf) is deleted.
# This prevents disk exhaustion — Jidoka: clean as you go.
cleanup_evidence() {
    playbook="$1"
    model_short=$(echo "$playbook" | sed 's/-mvp$//')
    evidence_dir="$HOME/data/qa-evidence/$model_short"
    [ -d "$evidence_dir" ] || return 0
    freed=0
    for ts_dir in "$evidence_dir"/*/; do
        [ -d "$ts_dir" ] || continue
        for subdir in "${ts_dir}conversions" "${ts_dir}workspace"; do
            if [ -d "$subdir" ]; then
                size=$(du -sm "$subdir" 2>/dev/null | cut -f1 || echo 0)
                freed=$((freed + size))
                rm -rf "$subdir"
            fi
        done
    done
    if [ "$freed" -gt 0 ]; then
        echo "[cleanup] Freed ${freed}MB from $model_short artifacts"
    fi
}

# ── Trap: clean up on kill (SIGTERM/SIGINT/EXIT) ──────────────────
# If the loop is killed mid-run, cleanup_evidence never runs.
# This trap ensures heavy artifacts are always removed.
cleanup_on_exit() {
    echo "[trap] Loop interrupted — cleaning up..."
    if [ -n "$CURRENT_PLAYBOOK" ]; then
        cleanup_evidence "$CURRENT_PLAYBOOK"
        # Reset the job so it can be re-claimed
        $JOBS_CMD reset-stale "$QA_MACHINE" 2>/dev/null || true
    fi
    rm -rf "$QA_REPO/output"
    echo "[trap] Cleanup complete."
}
trap cleanup_on_exit EXIT INT TERM

# ── Disk guard: refuse new jobs if disk too full ──────────────────
disk_usage_pct() {
    df --output=pcent /home/noah 2>/dev/null | tail -1 | tr -d ' %'
}

check_disk() {
    usage=$(disk_usage_pct)
    if [ "$usage" -ge "$DISK_THRESHOLD" ]; then
        echo "[disk] ${usage}% used (threshold: ${DISK_THRESHOLD}%) — pausing"
        return 1
    fi
    return 0
}

# ── Evict HF cache for certified models ───────────────────────────
# After a model is certified, its weights are no longer needed.
# Delete cached weights to reclaim disk space.
evict_hf_cache() {
    playbook="$1"
    model_short=$(echo "$playbook" | sed 's/-mvp$//')

    # Find the HF repo from the playbook YAML
    pb_file="$QA_REPO/playbooks/models/${playbook}.playbook.yaml"
    [ -f "$pb_file" ] || return 0

    hf_repo=$(grep 'hf_repo:' "$pb_file" | head -1 | sed 's/.*hf_repo:\s*["'\'']\?\([^"'\'']*\)["'\'']\?.*/\1/')
    [ -n "$hf_repo" ] || return 0

    cache_dir="$HOME/.apr/cache/hf/$hf_repo"
    if [ -d "$cache_dir" ]; then
        size=$(du -sm "$cache_dir" 2>/dev/null | cut -f1 || echo 0)
        rm -rf "$cache_dir"
        echo "[evict] Freed ${size}MB from HF cache: $hf_repo"
    fi
}

# ── Commit evidence to git (best-effort) ────────────────────────
commit_evidence() {
    playbook="$1"
    cd "$QA_REPO"
    if ! git diff --quiet certifications/ 2>/dev/null || \
       [ -n "$(git ls-files --others --exclude-standard certifications/ 2>/dev/null | head -1)" ]; then
        git add certifications/ && \
        git commit -m "qa: certify $playbook" && \
        git push origin main 2>/dev/null || true
    fi
}

# ── Main ────────────────────────────────────────────────────────
ensure_built

# Ensure DB is initialized (only on Intel where DB lives)
if [ "$QA_MACHINE" != "local" ]; then
    $JOBS_CMD init
fi

# Reset stale jobs from previous instance of this machine
stale=$($JOBS_CMD reset-stale "$QA_MACHINE" 2>/dev/null || true)
[ -n "$stale" ] && echo "$stale"

iteration=0
while [ "$iteration" -lt 999999 ]; do
    iteration=$((iteration + 1))

    # Sync repo (non-fatal)
    cd "$QA_REPO" && git pull --rebase origin main 2>/dev/null || true

    # Disk guard: skip claiming if disk too full
    if ! check_disk; then
        sleep 60
        continue
    fi

    # Claim next job
    playbook=$($JOBS_CMD claim "$QA_MACHINE" || true)

    if [ -z "$playbook" ]; then
        echo "[$iteration] No pending jobs for $QA_MACHINE. Sleeping 60s..."
        sleep 60
        continue
    fi

    echo "[$iteration] Claimed: $playbook"
    CURRENT_PLAYBOOK="$playbook"

    start_time=$(date +%s)

    # Clean stale output directory (causes "File exists" errors)
    rm -rf "$QA_REPO/output"

    # Run the three-tier pipeline
    pipeline_ok=true
    cd "$QA_REPO" && ./run-qa-intel.sh "$playbook" || pipeline_ok=false

    end_time=$(date +%s)
    duration=$((end_time - start_time))

    if [ "$pipeline_ok" = true ]; then
        # Find evidence file (only from the MVP run output)
        evidence_file=""
        if [ -f "$QA_REPO/output/evidence.json" ]; then
            evidence_file="$QA_REPO/output/evidence.json"
        fi

        if [ -n "$evidence_file" ] && [ -f "$evidence_file" ]; then
            pass_rate=$(extract_pass_rate "$evidence_file")
            if [ "$pass_rate" -ge 50 ] 2>/dev/null; then
                $JOBS_CMD complete "$playbook" "$pass_rate" "$duration"
                echo "[$iteration] Certified: $playbook (${pass_rate}%, ${duration}s)"
            else
                $JOBS_CMD fail "$playbook" "${pass_rate}% pass rate (below 50% threshold)"
                echo "[$iteration] Failed: $playbook (${pass_rate}% < 50% threshold)"
            fi
        else
            $JOBS_CMD fail "$playbook" "No evidence produced"
            echo "[$iteration] Failed: $playbook (no evidence)"
        fi

        commit_evidence "$playbook"
    else
        # Pipeline failed
        reason=$(tail -1 "$QA_REPO"/output/*.log 2>/dev/null | head -c 200 || echo "Pipeline failed")
        $JOBS_CMD fail "$playbook" "$reason"
        echo "[$iteration] Failed: $playbook ($reason)"

        # Still commit any partial evidence
        commit_evidence "$playbook"
    fi

    # AUTO-CLEAN: Always remove heavy artifacts (conversions/, workspace/)
    # Keeps evidence.json + run.log. Prevents disk exhaustion.
    cleanup_evidence "$playbook"
    rm -rf "$QA_REPO/output"

    # Evict HF cache for certified models (weights no longer needed)
    if [ "$pipeline_ok" = true ]; then
        evict_hf_cache "$playbook"
    fi

    CURRENT_PLAYBOOK=""
    echo "[$iteration] Cooldown ${COOLDOWN}s..."
    sleep "$COOLDOWN"
done

echo "=== QA loop finished after $iteration iterations ==="
