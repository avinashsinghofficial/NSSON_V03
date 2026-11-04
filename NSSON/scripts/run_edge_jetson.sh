#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.
python apps/edge_worker.py --config configs/edge_jetson.yaml
