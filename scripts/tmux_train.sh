#!/usr/bin/env bash
# Wisdom Dogah — create/attach tmux session for long ASR training
set -euo pipefail
cd "$(dirname "$0")/.."
SESSION="${SESSION_NAME:-ghana-asr}"
CONFIG="${1:-configs/whisper_akan_ewe.yaml}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Attaching to existing session: $SESSION"
  exec tmux attach -t "$SESSION"
fi

tmux new-session -d -s "$SESSION" "bash scripts/train.sh '$CONFIG'; bash"
echo "Started tmux session '$SESSION'. Attach with: tmux attach -t $SESSION"
