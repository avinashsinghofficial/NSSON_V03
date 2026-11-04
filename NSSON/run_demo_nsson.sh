#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/avi/NSSON"
cd "$ROOT" || exit 1

source .venv_nsson_gpu/bin/activate
export PYTHONPATH=.

exec python apps/run_demo_nsson.py --config configs/demo_simple.yaml "$@"
