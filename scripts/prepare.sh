#!/usr/bin/env bash
# Wisdom Dogah — prepare manifests on H200
set -euo pipefail
cd "$(dirname "$0")/.."
python -m ghana_asr.cli.prepare --config "${1:-configs/whisper_akan_ewe.yaml}"
