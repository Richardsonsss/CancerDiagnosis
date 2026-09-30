#!/usr/bin/env bash
# Train a cancer model from a prompt (any language) on the training VM and build its model package.
#
#   bash run_train.sh "recognise skin cancer"
#   bash run_train.sh "Hirntumor im MRT erkennen" --pool quick --top_k 3   # any language
# Long run, keep it alive after SSH disconnects:
#   nohup bash ~/CancerDiagnosisSystem/training/run_train.sh "recognise skin cancer" > /dev/null 2>&1 &
# Progress: tail -f ~/CancerDiagnosisSystem/logs/train_*.log
# Result:   ~/CancerDiagnosisSystem/artifacts/<task>_package.zip  (copy to the diagnosis service)
set -uo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"   # the folder this script is in
[ -x .venv/bin/python ] || { echo "ERROR: no .venv here - run: bash $PWD/setup.sh"; exit 1; }
source .venv/bin/activate
export PYTHONUNBUFFERED=1   # show progress immediately in the log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # less fragmentation -> fewer OOMs

mkdir -p ../logs
python train_system.py "$@" 2>&1 | tee -a "../logs/train_$(date +%Y%m%d_%H%M%S).log"
