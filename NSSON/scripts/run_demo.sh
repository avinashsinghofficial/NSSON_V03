#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.
python apps/run_demo.py --config configs/demo.yaml
