#!/bin/sh
# QA Job Queue CLI — SQLite-backed job queue for model qualification
# Design source: qa-jobs.rs (rash transpilation target when rash matures)
#
# Usage:
#   qa-jobs.sh init                                      Create DB + schema
#   qa-jobs.sh populate                                  Scan playbooks, upsert jobs
#   qa-jobs.sh status                                    Print summary table
#   qa-jobs.sh claim <machine>                           Atomic claim, print playbook
#   qa-jobs.sh complete <playbook> <pass_rate> <duration> Mark certified
#   qa-jobs.sh fail <playbook> <reason>                  Mark failed
#   qa-jobs.sh retry-failed                              Reset failed -> pending
#   qa-jobs.sh next <machine>                            Print next pending (no claim)
set -eu

QA_DB="${QA_DB:-/home/noah/data/qa-jobs/qa-jobs.db}"
QA_REPO="${QA_REPO:-/home/noah/src/apr-model-qa-playbook}"

init_db() {
    mkdir -p "$(dirname "$QA_DB")"
    sqlite3 "$QA_DB" <<'SQL'
CREATE TABLE IF NOT EXISTS jobs (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  playbook      TEXT NOT NULL UNIQUE,
  affinity      TEXT NOT NULL DEFAULT 'any'
                CHECK(affinity IN ('any', 'intel-only')),
  ram_gb        INTEGER NOT NULL DEFAULT 2,
  status        TEXT NOT NULL DEFAULT 'pending'
                CHECK(status IN ('pending','running','certified','failed','skipped')),
  claimed_by    TEXT,
  claimed_at    TEXT,
  finished_at   TEXT,
  failure_reason TEXT,
  pass_rate     INTEGER,
  duration_secs INTEGER
);
CREATE INDEX IF NOT EXISTS idx_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_affinity ON jobs(affinity);
SQL
    echo "Job queue initialized: $QA_DB"
}

# detect_ram <playbook-name> -> prints RAM in GB
detect_ram() {
    name="$1"
    # Explicit overrides for models without -Xb- in name
    case "$name" in
        deepseek-coder-v2-lite-*) echo 12;  return ;;
        openchat-3.5-*)           echo 6;   return ;;
        phi-3-medium-*)           echo 10;  return ;;
        phi-3-small-*)            echo 6;   return ;;
        qwen3-coder-next-*)      echo 160; return ;;
    esac
    # Pattern-based detection
    case "$name" in
        *-70b-*|*-72b-*) echo 44 ;;
        *-40b-*)         echo 26 ;;
        *-34b-*|*-33b-*|*-32b-*) echo 21 ;;
        *-30b-*)         echo 20 ;;
        *-27b-*|*-28b-*) echo 18 ;;
        *-24b-*|*-22b-*) echo 16 ;;
        *-20b-*)         echo 14 ;;
        *-15b-*)         echo 11 ;;
        *-14b-*)         echo 10 ;;
        *-13b-*|*-12b-*) echo 9 ;;
        *-9b-*)          echo 7 ;;
        *-8b-*|*-7b-*|*-6.7b-*|*-6b-*) echo 6 ;;
        *-5b-*)          echo 5 ;;
        *-4b-*)          echo 4 ;;
        *-3b-*)          echo 3 ;;
        *)               echo 2 ;;
    esac
}

# detect_affinity <playbook-name> <ram_gb> -> prints affinity
detect_affinity() {
    name="$1"
    ram="$2"
    # MoE models: small active params but full weights must be loaded (~20 GB)
    # Intel-only since total model size exceeds Lambda's safe budget
    case "$name" in
        *-30b-a3b-*) echo "intel-only"; return ;;
    esac
    # Models needing >3 GB RAM are intel-only
    if [ "$ram" -gt 3 ]; then
        echo "intel-only"
    else
        echo "any"
    fi
}

populate() {
    playbook_dir="$QA_REPO/playbooks/models"
    inserted=0
    skipped=0
    auto_certified=0

    for file in "$playbook_dir"/*-mvp.playbook.yaml; do
        [ -f "$file" ] || continue
        basename=$(basename "$file" .playbook.yaml)
        ram=$(detect_ram "$basename")
        affinity=$(detect_affinity "$basename" "$ram")

        existing=$(sqlite3 "$QA_DB" "SELECT status FROM jobs WHERE playbook='$basename'" 2>/dev/null || true)

        if [ -z "$existing" ]; then
            # Check if already certified (evidence exists)
            model_short=$(echo "$basename" | sed 's/-mvp$//')
            if find "$QA_REPO/certifications" -path "*${model_short}*" -name 'evidence.json' 2>/dev/null | grep -q .; then
                sqlite3 "$QA_DB" "INSERT INTO jobs (playbook, affinity, ram_gb, status) VALUES ('$basename', '$affinity', $ram, 'certified')"
                auto_certified=$((auto_certified + 1))
            else
                sqlite3 "$QA_DB" "INSERT INTO jobs (playbook, affinity, ram_gb) VALUES ('$basename', '$affinity', $ram)"
                inserted=$((inserted + 1))
            fi
        else
            skipped=$((skipped + 1))
        fi
    done

    total=$(sqlite3 "$QA_DB" 'SELECT COUNT(*) FROM jobs')
    echo "Done. Inserted: $inserted  Auto-certified: $auto_certified  Skipped: $skipped  Total: $total"
}

show_status() {
    echo "=== QA Job Queue Status ==="
    echo ""

    certified=$(sqlite3 "$QA_DB" "SELECT COUNT(*) FROM jobs WHERE status='certified'")
    running=$(sqlite3 "$QA_DB" "SELECT COUNT(*) FROM jobs WHERE status='running'")
    pending=$(sqlite3 "$QA_DB" "SELECT COUNT(*) FROM jobs WHERE status='pending'")
    failed=$(sqlite3 "$QA_DB" "SELECT COUNT(*) FROM jobs WHERE status='failed'")
    skipped=$(sqlite3 "$QA_DB" "SELECT COUNT(*) FROM jobs WHERE status='skipped'")
    total=$(sqlite3 "$QA_DB" "SELECT COUNT(*) FROM jobs")

    echo "Certified: $certified  Running: $running  Pending: $pending  Failed: $failed  Skipped: $skipped  Total: $total"
    echo ""

    running_list=$(sqlite3 -separator '  ' "$QA_DB" "SELECT playbook, claimed_by, claimed_at FROM jobs WHERE status='running' ORDER BY claimed_at")
    if [ -n "$running_list" ]; then
        echo "Running:"
        echo "$running_list"
        echo ""
    fi

    failed_list=$(sqlite3 -separator '  ' "$QA_DB" "SELECT playbook, failure_reason FROM jobs WHERE status='failed' ORDER BY playbook")
    if [ -n "$failed_list" ]; then
        echo "Failed:"
        echo "$failed_list"
        echo ""
    fi

    pending_list=$(sqlite3 -separator '  ' "$QA_DB" "SELECT playbook, affinity, ram_gb FROM jobs WHERE status='pending' ORDER BY ram_gb ASC LIMIT 5")
    if [ -n "$pending_list" ]; then
        echo "Next pending (smallest RAM first):"
        echo "$pending_list"
    fi
}

claim_next() {
    machine="${1:-intel}"

    if [ "$machine" = "local" ]; then
        affinity_filter="AND affinity = 'any' AND ram_gb <= 3"
    else
        affinity_filter="AND affinity IN ('any', 'intel-only')"
    fi

    # Atomic claim using BEGIN EXCLUSIVE — no races
    result=$(sqlite3 "$QA_DB" "
BEGIN EXCLUSIVE;
UPDATE jobs SET status='running', claimed_by='$machine',
  claimed_at=strftime('%Y-%m-%dT%H:%M:%SZ','now')
WHERE playbook = (
  SELECT playbook FROM jobs
  WHERE status='pending'
    $affinity_filter
  ORDER BY ram_gb ASC
  LIMIT 1
) AND status='pending';
SELECT changes();
COMMIT;
")

    if [ "$result" = "1" ]; then
        sqlite3 "$QA_DB" "SELECT playbook FROM jobs WHERE status='running' AND claimed_by='$machine' ORDER BY claimed_at DESC LIMIT 1"
    fi
    # If no claim, print nothing — caller checks empty output
}

complete_job() {
    playbook="$1"
    pass_rate="$2"
    duration="$3"
    sqlite3 "$QA_DB" "UPDATE jobs SET status='certified', pass_rate=$pass_rate, duration_secs=$duration, finished_at=strftime('%Y-%m-%dT%H:%M:%SZ','now') WHERE playbook='$playbook'"
    echo "Certified: $playbook (pass_rate=${pass_rate}%, duration=${duration}s)"
}

fail_job() {
    playbook="$1"
    reason="$2"
    # Escape single quotes in reason
    safe_reason=$(echo "$reason" | sed "s/'/''/g")
    sqlite3 "$QA_DB" "UPDATE jobs SET status='failed', failure_reason='$safe_reason', finished_at=strftime('%Y-%m-%dT%H:%M:%SZ','now') WHERE playbook='$playbook'"
    echo "Failed: $playbook ($reason)"
}

retry_failed() {
    count=$(sqlite3 "$QA_DB" "SELECT COUNT(*) FROM jobs WHERE status='failed'")
    sqlite3 "$QA_DB" "UPDATE jobs SET status='pending', claimed_by=NULL, claimed_at=NULL, finished_at=NULL, failure_reason=NULL WHERE status='failed'"
    echo "Reset $count failed jobs to pending"
}

reset_all() {
    running=$(sqlite3 "$QA_DB" "SELECT COUNT(*) FROM jobs WHERE status='running'")
    failed=$(sqlite3 "$QA_DB" "SELECT COUNT(*) FROM jobs WHERE status='failed'")
    sqlite3 "$QA_DB" "UPDATE jobs SET status='pending', claimed_by=NULL, claimed_at=NULL, finished_at=NULL, failure_reason=NULL, pass_rate=NULL, duration_secs=NULL WHERE status IN ('running','failed')"
    echo "Reset $running running + $failed failed jobs to pending"
}

next_job() {
    machine="${1:-intel}"

    if [ "$machine" = "local" ]; then
        affinity_filter="AND affinity = 'any' AND ram_gb <= 3"
    else
        affinity_filter="AND affinity IN ('any', 'intel-only')"
    fi

    sqlite3 "$QA_DB" "SELECT playbook FROM jobs WHERE status='pending' $affinity_filter ORDER BY ram_gb ASC LIMIT 1"
}

reset_stale() {
    machine="${1:-intel}"
    count=$(sqlite3 "$QA_DB" "SELECT COUNT(*) FROM jobs WHERE status='running' AND claimed_by='$machine'")
    if [ "$count" -gt 0 ]; then
        sqlite3 "$QA_DB" "UPDATE jobs SET status='pending', claimed_by=NULL, claimed_at=NULL, finished_at=NULL WHERE status='running' AND claimed_by='$machine'"
        echo "Reset $count stale running job(s) from $machine to pending"
    fi
}

# ── Main dispatch ───────────────────────────────────────────────
cmd="${1:-status}"
case "$cmd" in
    init)         init_db ;;
    populate)     populate ;;
    status)       show_status ;;
    claim)        claim_next "${2:-intel}" ;;
    complete)     complete_job "$2" "${3:-0}" "${4:-0}" ;;
    fail)         fail_job "$2" "${3:-unknown}" ;;
    retry-failed) retry_failed ;;
    reset-all)    reset_all ;;
    reset-stale)  reset_stale "${2:-intel}" ;;
    next)         next_job "${2:-intel}" ;;
    *)
        echo "Usage: qa-jobs.sh <init|populate|status|claim|complete|fail|retry-failed|reset-stale|reset-all|next>"
        exit 1
        ;;
esac
