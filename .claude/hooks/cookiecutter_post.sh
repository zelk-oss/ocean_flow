#!/usr/bin/env bash
# Remove the most recent rendered cookiecutter temp directory for this
# session (LIFO pop from the stack written by the pre hook).
set -euo pipefail

INPUT="$(cat)"
SESSION_ID="$(printf '%s' "$INPUT" | jq -r '.session_id // "nosession"')"
STACK_FILE="/tmp/surrogate-template-render.${SESSION_ID}.stack"

[[ -f "$STACK_FILE" ]] || exit 0

RENDER_ROOT="$(tail -n1 "$STACK_FILE")"
# Pop last line.
sed -i '' -e '$d' "$STACK_FILE" 2>/dev/null || sed -i -e '$d' "$STACK_FILE"
[[ -s "$STACK_FILE" ]] || rm -f "$STACK_FILE"

case "$RENDER_ROOT" in
    /tmp/surrogate-template-render-*) rm -rf "$RENDER_ROOT" ;;
esac
