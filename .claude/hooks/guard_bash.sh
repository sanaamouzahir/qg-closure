#!/usr/bin/env bash
# PreToolUse guard for Bash tool calls. Receives the tool-call JSON on stdin.
# Exit 2 = block the command and tell the agent why. Exit 0 = allow.
input="$(cat)"

# HARD RULE: forbidden SGE flags, never allowed anywhere.
if grep -Eq 'ibamd\.q|h_vmem' <<<"$input"; then
  echo "BLOCKED: forbidden SGE flag (ibamd.q or h_vmem detected). GPU jobs use '-q ibgpu.q -l gpu=1' ONLY." >&2
  exit 2
fi

# HARD RULE: no direct push to main from a branch session (global supervisor merges via PR).
if grep -Eq 'git[[:space:]]+push' <<<"$input" && grep -Eq '(^|[^a-zA-Z])main([^a-zA-Z]|$)' <<<"$input"; then
  echo "BLOCKED: direct push to main. Push your branch and open a PR; the global supervisor handles merges." >&2
  exit 2
fi

# HARD RULE: no float32 in the closure data/train path (advisory catch for obvious cases).
if grep -Eq 'dtype[= ]+.?float32|--compute-dtype[= ]+float32' <<<"$input"; then
  echo "BLOCKED: float32 in a closure command. float64 is mandatory through data-build and training." >&2
  exit 2
fi

exit 0
