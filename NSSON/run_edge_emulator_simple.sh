#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/avi/NSSON"
cd "$ROOT" || exit 1

source .venv_nsson_gpu/bin/activate
export PYTHONPATH=.

exec python apps/jetson_edge_worker.py --config configs/edge_emulator_simple.yaml
