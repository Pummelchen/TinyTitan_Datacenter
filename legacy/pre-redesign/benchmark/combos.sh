#!/usr/bin/env bash
# Compatibility entry point for the current benchmark harness: the coder round
# (coding clients), the features round (direct OpenAI requests) and the clients
# round (the launcher's client list and this harness's agreeing; no model).
#
# Examples:
#   benchmark/combos.sh
#   benchmark/combos.sh --round coder
#   benchmark/combos.sh --round features
#   benchmark/combos.sh --round clients
#   benchmark/combos.sh --round all --output .build/benchmark-rounds/my-run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$SCRIPT_DIR/coder_cli_benchmark.py" "$@"
