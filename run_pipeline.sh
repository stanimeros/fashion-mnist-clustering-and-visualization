#!/usr/bin/env bash
# Fashion-MNIST DR + clustering pipeline
#
# Usage:
#   ./run_pipeline.sh quick    # smoke test (few epochs, small subsamples)
#   ./run_pipeline.sh full       # production run (default)
#
# Optional: create a venv first —  python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
# First-time on a server you may run:  INSTALL_DEPS=1 ./run_pipeline.sh full

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODE="${1:-full}"
case "$MODE" in
  quick)
    export FASHION_MNIST_QUICK_RUN=1
    ;;
  full)
    export FASHION_MNIST_QUICK_RUN=0
    ;;
  *)
    echo "Usage: $0 [quick|full]" >&2
    exit 1
    ;;
esac

if [[ -d .venv ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi

if [[ "${INSTALL_DEPS:-0}" == "1" ]]; then
  python3 -m pip install -r requirements.txt
fi

exec python3 main.py
