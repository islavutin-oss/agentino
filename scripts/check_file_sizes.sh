#!/usr/bin/env bash
# Enforce max file size policy: 600 lines per .py file
# Run in CI or as pre-commit hook

MAX_LINES=650
FAILED=0

for f in $(find src/agentino -name "*.py" -not -path "*/__pycache__/*"); do
    lines=$(wc -l < "$f")
    if [ "$lines" -gt "$MAX_LINES" ]; then
        echo "FAIL: $f ($lines lines > $MAX_LINES max)"
        FAILED=1
    fi
done

if [ "$FAILED" -eq 0 ]; then
    echo "OK: all files under $MAX_LINES lines"
else
    echo ""
    echo "Split large files into focused modules. See CLAUDE.md for patterns."
    exit 1
fi
