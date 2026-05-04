#!/usr/bin/env bash
# Force-commit anything new under a results path. Idempotent — safe to run
# repeatedly. The first arg is the path (default: results/), the second is
# the commit message (must be a single argument; quote it).
#
# Usage:
#   ./scripts/git_save_results.sh results/p5 "results: p5 sweep done"

set -euo pipefail

PATH_TO_SAVE="${1:-results}"
MSG="${2:-results: snapshot $(date -u +%Y-%m-%dT%H:%M:%SZ)}"

if [[ ! -d "$PATH_TO_SAVE" ]]; then
    echo "no $PATH_TO_SAVE/ to commit"
    exit 0
fi

# Anything in there?
if [[ -z "$(ls -A "$PATH_TO_SAVE" 2>/dev/null)" ]]; then
    echo "$PATH_TO_SAVE/ is empty; nothing to commit"
    exit 0
fi

# Force-add despite gitignore.
git add -f "$PATH_TO_SAVE"

# Bail if nothing staged (e.g., results unchanged since last commit).
if git diff --cached --quiet; then
    echo "no result changes under $PATH_TO_SAVE/ to commit"
    exit 0
fi

git commit -m "$MSG"
echo "committed: $MSG"
