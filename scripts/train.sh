#!/usr/bin/env bash
# Wisdom Dogah — launch Whisper training (prefer running inside tmux)
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-configs/whisper_akan_ewe.yaml}"
echo "Author: Wisdom Dogah"
echo "Config: $CONFIG"
echo "CUDA check:"
python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
python -m ghana_asr.cli.train --config "$CONFIG"
