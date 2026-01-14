#!/bin/bash

module load gpu/cuda-11.6

NODE_PARAMS="-p hiperf --gres=gpu:v100:1 --nodelist=tesla-v100 -t 00:15:00"

srun -n1 \
    --cpus-per-task=8 \
    --mem=45000 \
    $NODE_PARAMS \
    --job-name=vit-p \
    python setup.py build install