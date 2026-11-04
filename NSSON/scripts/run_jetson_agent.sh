#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.
python apps/jetson_node_agent.py --config configs/jetson_agent.yaml
