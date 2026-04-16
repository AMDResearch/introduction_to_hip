#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
MANIFEST="$REPO_ROOT/ai_academy.manifest"
DEST="$REPO_ROOT/ai_academy"

if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: $MANIFEST not found." >&2
    exit 1
fi

# Clean previous output to remove stale files
rm -rf "$DEST"

count=0
while IFS= read -r line || [[ -n "$line" ]]; do
    # Skip blank lines and comments
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue

    src="$REPO_ROOT/$line"
    if [[ ! -e "$src" ]]; then
        echo "WARNING: $line does not exist, skipping." >&2
        continue
    fi

    target="$DEST/$line"
    mkdir -p "$(dirname "$target")"
    cp -a "$src" "$target"
    count=$((count + 1))
done < "$MANIFEST"

echo "Synced $count entries into $DEST"
