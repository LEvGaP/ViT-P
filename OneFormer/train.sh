#!/bin/sh

source ./env.sh

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export DETECTRON2_DATASETS="/home6/m_imm_freedata/Segmentation/Projects"

GPUS=2
NODE_PARAMS="-p hiperf --gres=gpu:a100:2 --nodelist=tesla-a101 -t 00:30:00"

CONFIG="./configs/ade20k/dinat/oneformer_dinat_large_bs16_160k_1280x1280.yaml"
# MODEL_WEIGHTS=""
MODEL_WEIGHTS="/home6/m_imm_freedata/Segmentation/Projects/lapshin/OneFormer_ViT-P/checkpoints/1280x1280_250_16_dinat_l_oneformer_ade20k_160k.pth"
CLASSIFICATION_WEIGHTS="/home/s0038/_scratch2/ViT-P/OneFormer/andrey_weights/model_ADE20k_base774_250point.pth"
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
        MODEL.WEIGHTS \"${MODEL_WEIGHTS}\" \
        MODEL.Classification_WEIGHTS ${CLASSIFICATION_WEIGHTS} \
        MODEL.TEST.TASK ${TASK} \
        SOLVER.IMS_PER_BATCH 2 \
        SOLVER.BASE_LR 0.0001 \
        DATASETS.TRAIN \"('landcover_trees_train',)\" \
        DATASETS.TEST_SEMANTIC \"('landcover_trees_val',)\" \
        INPUT.DATASET_MAPPER_NAME landcover_trees \
        OUTPUT_DIR ./output_landcover_trees \
        SOLVER.AMP.ENABLED False \
        MODEL.SEM_SEG_HEAD.NUM_CLASSES 2 \
        TEST.EVAL_PERIOD 100 \
        SOLVER.CHECKPOINT_PERIOD 100 \
        MODEL.DEVICE cuda"