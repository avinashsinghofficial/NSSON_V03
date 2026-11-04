#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$HOME/NSSON}"
cd "$ROOT"
source .venv_nsson_gpu/bin/activate
export PYTHONPATH=.
exec python apps/jetson_node_agent.py --config configs/jetson_agent.yaml
