#!/bin/bash

module load gpu/cuda-11.6

GPUS=1
NODE_PARAMS="-p hiperf --gres=gpu:a100:$GPUS --nodelist=tesla-a101 -t 00:15:00"

srun -n1 \
    --cpus-per-task=8 \
    --mem=45000 \
    $NODE_PARAMS \
    --job-name=vit-p \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS} \
    python setup.py build install