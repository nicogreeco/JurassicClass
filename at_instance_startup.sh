#!/bin/bash

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