#!/bin/sh
# Post-dim-smoke cleanup: commit evidence, evict HF cache if no further tiers.
# Usage: post-dim-smoke.sh <model_short> <qa_repo>
set -eu

model="$1"
qa_repo="$2"

cd "$qa_repo" || exit 1

# Copy evidence to certifications
if [ -f output/evidence.json ]; then
    mkdir -p "certifications/$model"
    cp output/evidence.json "certifications/$model/evidence.json"
fi

# Commit evidence
git add "certifications/$model/evidence.json" 2>/dev/null || true
git commit -m "qa: certify $model (dim-smoke) (Refs #10)" 2>/dev/null || true

# Always evict HF cache after dim-smoke. apr pull will re-download
# when needed for smoke/mvp. This prevents disk exhaustion on large
# model queues (Jidoka: clean as you go).
for pb in "playbooks/models/${model}-dim-smoke.playbook.yaml" "playbooks/models/${model}-mvp.playbook.yaml"; do
    [ -f "$pb" ] || continue
    hf_repo=$(grep 'hf_repo:' "$pb" | head -1 | sed "s/.*hf_repo:\s*[\"']\?\\([^\"']*\\)[\"']\?.*/\1/")
    cache="$HOME/.apr/cache/hf/$hf_repo"
    if [ -d "$cache" ] && [ -n "$hf_repo" ]; then
        size=$(du -sm "$cache" 2>/dev/null | cut -f1 || echo 0)
        rm -rf "$cache"
        echo "evict: Freed ${size}MB from HF cache: $hf_repo"
    fi
    break
done

rm -rf "$qa_repo/output"
