#!/usr/bin/env bash
# Local-network diagnosis service on macOS / Linux - no internet connection needed.
#   bash lan/start_lan.sh --prepare      # once, while online: create lan/.venv and install packages
#   bash lan/start_lan.sh                # start: http://<address>:8000
#   bash lan/start_lan.sh --https        # start with a local HTTPS certificate
# Extra arguments are passed to server/lan.py (e.g. --port 8080 --models-dir /data/models).
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."   # project folder
if [ "${1:-}" = "--prepare" ]; then
    python3 -m venv lan/.venv
    lan/.venv/bin/python -m pip install --upgrade pip
    lan/.venv/bin/python -m pip install -r server/requirements.txt
    mkdir -p models
    [ -f web/dist/index.html ] || { (cd web && npm ci && npm run build); }
    echo "ready - put the *_package.zip files into $(pwd)/models and run: bash lan/start_lan.sh"
    exit 0
fi
[ -x lan/.venv/bin/python ] || { echo "not prepared yet - run once while online: bash lan/start_lan.sh --prepare"; exit 1; }
exec lan/.venv/bin/python server/lan.py "$@"
