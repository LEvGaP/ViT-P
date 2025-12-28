#!/bin/sh

module load gpu/cuda-11.6

source /opt/conda/etc/profile.d/conda.sh
conda activate ./env

cd ../ViT-P
export PYTHONPATH="$PYTHONPATH:$(pwd)"
cd ../OneFormer
