#!/bin/bash

set -euo pipefail

LOG_DIR="/workspace/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/provision-$(date +'%Y%m%d-%H%M%S').log"
exec > >(tee -a "$LOG_FILE") 2>&1
trap 'echo "ERROR on line $LINENO: $BASH_COMMAND"; exit 1' ERR

echo "=== Provisioning started: $(date -Is) ==="

git config --global user.name "Nicola Greco"
git config --global user.email "niccogreek@gmail.com"
cd /workspace
git clone https://github.com/nicogreeco/JurassicClass.git

cd JurassicClass
source /venv/main/bin/activate
uv pip install -r requirements.txt
uv pip install vastai huggingface_hub

mkdir -p /workspace/JurassicClass/dataset/dataset
hf download niccogreek/JurassicClass --repo-type dataset \
  --local-dir /workspace/JurassicClass/dataset/dataset

uv pip install python-dateutil

sudo apt update && sudo apt install screen -y

echo "=== Provisioning finished OK: $(date -Is) ==="
