#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$HOME/NSSON}"
cd "$ROOT"
source .venv_nsson_gpu/bin/activate
export PYTHONPATH=.
exec python apps/remote_worker.py --config configs/remote_worker.yaml
