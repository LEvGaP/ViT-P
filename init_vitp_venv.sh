#!/bin/bash
set -e

source /opt/conda/etc/profile.d/conda.sh
conda create --name vitp python=3.11 -y

conda activate vitp

pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 xformers==0.0.24

pip3 install -r requirements.txt