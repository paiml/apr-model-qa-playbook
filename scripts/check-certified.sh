#!/bin/sh
# Check if a model has passing certification evidence.
# Usage: check-certified.sh <evidence_file> <threshold>
# Exit 0 if certified (pass rate >= threshold), exit 1 otherwise.
evidence="$1"
threshold="${2:-50}"

test -f "$evidence" || exit 1

rate=$(python3 -c "
import json, sys
data = json.load(open(sys.argv[1]))
total = len(data) if isinstance(data, list) else 0
passed = sum(1 for e in data if e.get('outcome') == 'Corroborated') if total else 0
print(int(100 * passed / total) if total > 0 else 0)
" "$evidence" 2>/dev/null || echo "0")

[ "$rate" -ge "$threshold" ] 2>/dev/null
