#!/bin/sh

source ./env.sh

export DETECTRON2_DATASETS="/home6/m_imm_freedata/Segmentation/Projects/lapshin/Detectron2_datasets"

GPUS=1
NODE_PARAMS="-p hiperf --gres=gpu:a100:1 --nodelist=tesla-a101 -t 00:05:00"

CONFIG="./configs/ade20k/dinat/oneformer_dinat_large_bs16_160k_1280x1280.yaml"
CHECKPOINT_WEIGHTS="/home6/m_imm_freedata/Segmentation/Projects/lapshin/OneFormer_ViT-P/checkpoints"
MODEL_WEIGHTS="$CHECKPOINT_WEIGHTS/1280x1280_250_16_dinat_l_oneformer_ade20k_160k.pth"
CLASSIFICATION_WEIGHTS="$CHECKPOINT_WEIGHTS/model_ADE20k_base774_250point.pth"
TASK="semantic"

sbatch -n1 \
    --cpus-per-task=8 \
    --mem=45000 \
    $NODE_PARAMS \
    --job-name="vitp-eval" \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS} \
    --wrap="python train_net.py --dist-url 'tcp://127.0.0.1:50164' \
        --num-gpus ${GPUS} \
        --config-file ${CONFIG} \
        --eval-only MODEL.IS_TRAIN False MODEL.WEIGHTS ${MODEL_WEIGHTS} \
        MODEL.Classification_WEIGHTS ${CLASSIFICATION_WEIGHTS} \
        MODEL.TEST.TASK ${TASK}"
