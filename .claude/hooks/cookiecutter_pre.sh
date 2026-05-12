#!/usr/bin/env bash
# Render the cookiecutter template into a timestamped temp directory,
# then rewrite the incoming Bash command so it runs inside it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)-$$"
RENDER_ROOT="/tmp/surrogate-template-render-${STAMP}"

INPUT="$(cat)"
ORIG_CMD="$(printf '%s' "$INPUT" | jq -r '.tool_input.command')"
SESSION_ID="$(printf '%s' "$INPUT" | jq -r '.session_id // "nosession"')"

mkdir -p "$RENDER_ROOT"

if ! cookiecutter "$TEMPLATE_DIR" --no-input \
        -o "$RENDER_ROOT" >/tmp/cookiecutter_pre.log 2>&1; then
    rm -rf "$RENDER_ROOT"
    jq -nc --arg r "cookiecutter render failed; see /tmp/cookiecutter_pre.log" '{
      hookSpecificOutput: {
        hookEventName: "PreToolUse",
        permissionDecision: "deny",
        permissionDecisionReason: $r
      }
    }'
    exit 0
fi

RENDERED_DIR="$(find "$RENDER_ROOT" -mindepth 1 -maxdepth 1 -type d | head -n1)"

# Track render roots per session so the post hook can clean up LIFO.
STACK_FILE="/tmp/surrogate-template-render.${SESSION_ID}.stack"
printf '%s\n' "$RENDER_ROOT" >> "$STACK_FILE"

NEW_CMD="cd \"$RENDERED_DIR\" && $ORIG_CMD"

jq -nc --arg cmd "$NEW_CMD" --arg dir "$RENDERED_DIR" '{
  systemMessage: ("cookiecutter rendered to " + $dir),
  hookSpecificOutput: {
    hookEventName: "PreToolUse",
    updatedInput: { command: $cmd }
  }
}'
