#!/bin/sh
# Extract pass rate (0-100) from evidence.json
# Usage: pass-rate.sh <evidence_file>
# Returns: integer percentage, or "0" on any error
python3 -c "
import json, sys
data = json.load(open(sys.argv[1]))
total = len(data) if isinstance(data, list) else 0
passed = sum(1 for e in data if e.get('outcome') == 'Corroborated') if total else 0
print(int(100 * passed / total) if total > 0 else 0)
" "$1" 2>/dev/null || echo "0"
