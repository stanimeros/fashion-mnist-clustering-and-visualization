#!/usr/bin/env bash
# Fashion-MNIST DR + clustering pipeline
#
# Creates .venv if missing, installs requirements, then runs main.py.
#
# Usage:
#   ./run_pipeline.sh        # full run (default)
#   ./run_pipeline.sh full
#   ./run_pipeline.sh quick  # smoke test
#
# Optional: PYTHON=python3.11 ./run_pipeline.sh   (default: python3.11 if present, else python3)

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

if [[ -n "${PYTHON:-}" ]]; then
  VENV_PY="$PYTHON"
elif command -v python3.11 &>/dev/null; then
  VENV_PY=python3.11
else
  VENV_PY=python3
fi

if [[ ! -d .venv ]]; then
  echo "Creating .venv with ${VENV_PY} …"
  "${VENV_PY}" -m venv .venv
fi

# shellcheck source=/dev/null
source .venv/bin/activate

echo "Installing dependencies …"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Running main.py (QUICK_RUN=${FASHION_MNIST_QUICK_RUN}) …"
exec python main.py
