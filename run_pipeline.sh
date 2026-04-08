#!/usr/bin/env bash
# Fashion-MNIST DR + clustering pipeline
#
# Creates .venv only if missing. Runs pip only when requirements.txt changes
# (hash in .venv/.requirements.sha256) or on first run; override with FORCE_PIP_INSTALL=1.
#
# Usage:
#   ./run_pipeline.sh              # full run (foreground)
#   ./run_pipeline.sh full|quick
#
# Background (close terminal; output only in log):
#   ./run_pipeline.sh --background
#   ./run_pipeline.sh -b quick
#
# Optional: PYTHON=python3.11 ./run_pipeline.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DETACH=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    -b|--background|--detach)
      DETACH=true
      shift
      ;;
    *)
      break
      ;;
  esac
done

MODE="${1:-full}"

if [[ "$DETACH" == true ]]; then
  mkdir -p "$ROOT/logs"
  LOGFILE="$ROOT/logs/pipeline-$(date +%Y%m%d-%H%M%S).log"
  nohup "$ROOT/run_pipeline.sh" "$MODE" >>"$LOGFILE" 2>&1 &
  echo "$!" >"$ROOT/logs/last.pid"
  printf '%s\n' \
    "Background job started (safe to close this terminal)." \
    "  Log:  $LOGFILE" \
    "  PID:  $(cat "$ROOT/logs/last.pid")" \
    "  tail -f \"$LOGFILE\""
  exit 0
fi

case "$MODE" in
  quick)
    export FASHION_MNIST_QUICK_RUN=1
    ;;
  full)
    export FASHION_MNIST_QUICK_RUN=0
    ;;
  *)
    echo "Usage: $0 [-b|--background] [quick|full]" >&2
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

REQ_HASH="$(
  python -c "import hashlib, pathlib; print(hashlib.sha256(pathlib.Path('requirements.txt').read_bytes()).hexdigest())"
)"
HASH_FILE=".venv/.requirements.sha256"
NEED_INSTALL=1
if [[ "${FORCE_PIP_INSTALL:-0}" == "1" ]]; then
  :
elif [[ -f "$HASH_FILE" ]] && [[ "$(cat "$HASH_FILE")" == "$REQ_HASH" ]]; then
  NEED_INSTALL=0
fi

if [[ "$NEED_INSTALL" == 1 ]]; then
  echo "Installing / updating dependencies …"
  export PIP_DISABLE_PIP_VERSION_CHECK=1
  python -m pip install -q --upgrade pip
  python -m pip install -r requirements.txt
  printf '%s' "$REQ_HASH" >"$HASH_FILE"
else
  echo "Skipping pip: .venv OK and requirements.txt unchanged (use FORCE_PIP_INSTALL=1 to reinstall)."
fi

echo "Running main.py (QUICK_RUN=${FASHION_MNIST_QUICK_RUN}) …"
exec python main.py
