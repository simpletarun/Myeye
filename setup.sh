#!/usr/bin/env bash
# Myeye - one-click setup + run (Linux / macOS)
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
# a moved/copied project leaves a .venv that cannot be activated
[ -f .venv/bin/activate ] || rm -rf .venv
$PY -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip -q
pip install -r requirements.txt
python run.py "$@"
