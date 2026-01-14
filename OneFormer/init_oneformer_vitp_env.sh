#!/bin/bash
set -e

module load gpu/cuda-11.6

source /opt/conda/etc/profile.d/conda.sh
conda create --prefix env python=3.12 -y

conda activate ./env

pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

pip3 install numpy==1.26.4 opencv-python==4.10.0.82

python tools/setup_detectron2.py

pip3 install 'pip<24.1'

pip3 install -r requirements.txt

cd oneformer/modeling/pixel_decoder/ops
sh custom_make.sh
cd ../../../..

cd oneformer/modeling/backbone/ops_dcnv3
sh custom_make.sh
sh run_test.sh
cd ../../../..

cd detectron2
pip install -e .
cd ..