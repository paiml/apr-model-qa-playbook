#!/bin/sh
# Post-certification cleanup: commit evidence, evict HF cache, remove artifacts.
# Usage: post-certify.sh <model_short> <qa_repo> <evidence_dir>
set -eu

model="$1"
qa_repo="$2"
evidence_dir="$3"

cd "$qa_repo" || exit 1

# Commit evidence
git add "certifications/$model/evidence.json" 2>/dev/null || true
git commit -m "qa: certify $model (mvp) (Refs #10)" 2>/dev/null || true
git push origin main 2>/dev/null || true

# Evict HF cache
pb_file="playbooks/models/${model}-mvp.playbook.yaml"
if [ -f "$pb_file" ]; then
    hf_repo=$(grep 'hf_repo:' "$pb_file" | head -1 | sed "s/.*hf_repo:\s*[\"']\?\\([^\"']*\\)[\"']\?.*/\1/")
    cache_dir="$HOME/.apr/cache/hf/$hf_repo"
    if [ -d "$cache_dir" ] && [ -n "$hf_repo" ]; then
        size=$(du -sm "$cache_dir" 2>/dev/null | cut -f1 || echo 0)
        rm -rf "$cache_dir"
        echo "evict: Freed ${size}MB from HF cache: $hf_repo"
    fi
fi

# Clean heavy artifacts (workspace, conversions)
for subdir in "$evidence_dir/$model/workspace" "$evidence_dir/$model/conversions"; do
    if [ -d "$subdir" ]; then
        rm -rf "$subdir"
    fi
done

rm -rf "$qa_repo/output"
