#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.
python apps/remote_worker.py --config configs/remote_worker.yaml
