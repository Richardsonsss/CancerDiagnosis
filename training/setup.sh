#!/usr/bin/env bash
# One-time environment setup on the training VM (Ubuntu / Debian; NVIDIA GPU recommended).
#   bash ~/CancerDiagnosisSystem/training/setup.sh
#
# Everything is installed into ./.venv with .venv/bin/python -m pip, never into the
# system Python (Ubuntu blocks that: "error: externally-managed-environment").
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"   # the folder this script is in

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "WARNING: nvidia-smi not found - no NVIDIA driver, training will run on CPU (very slow)."
    echo "         Use a GPU instance (g4dn/g5/p3...) with the Deep Learning AMI or install the driver."
else
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
fi

PYVER=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
echo "system python: $PYVER"

venv_ok() { [ -x .venv/bin/python ] && .venv/bin/python -m pip --version >/dev/null 2>&1; }

if ! venv_ok; then
    # A .venv without pip (created while python3-venv was missing) makes `pip`
    # fall back to the system pip -> externally-managed-environment. Rebuild it.
    rm -rf .venv
    if ! python3 -m venv .venv 2>/dev/null || ! venv_ok; then
        echo "installing python${PYVER}-venv (needed to create virtual environments)"
        sudo apt-get update
        sudo apt-get install -y python3-venv "python${PYVER}-venv" || sudo apt-get install -y python3-venv python3-full
        rm -rf .venv
        python3 -m venv .venv
    fi
fi
venv_ok || { echo "ERROR: could not create a working .venv"; exit 1; }

PY=.venv/bin/python
$PY -m pip install --upgrade pip
# The default Linux wheels of torch ship with CUDA.
$PY -m pip install -r requirements.txt

$PY - <<'EOF'
import sys, torch, timm
print("python", sys.executable)
print("torch", torch.__version__, "| timm", timm.__version__)
print("CUDA available:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
EOF

$PY ../common/voting.py
