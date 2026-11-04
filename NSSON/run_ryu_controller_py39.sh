#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$HOME/NSSON}"
cd "$ROOT"
source .venv_ryu39/bin/activate
export PYTHONPATH=.
export EVENTLET_NO_GREENDNS=yes
exec python -m ryu.cmd.manager apps/ryu_camnsson_app.py --wsapi-port 8080
