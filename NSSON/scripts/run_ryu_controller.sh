#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export EVENTLET_NO_GREENDNS=yes
python -m ryu.cmd.manager apps/ryu_camnsson_app.py --wsapi-port 8080
