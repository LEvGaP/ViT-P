#!/bin/bash

source /opt/conda/etc/profile.d/conda.sh
conda create --name vitp python=3.11 -y

conda activate vitp

pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 xformers==0.0.24

pip install numpy==1.26.4

pip install omegaconf torchmetrics==1.6.0 fvcore iopath submitit==1.5.2