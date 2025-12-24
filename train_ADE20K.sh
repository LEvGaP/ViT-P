#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gpus-per-node=a100:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=00-07:00:00

DATASET_PATH="/home6/m_imm_freedata/Segmentation/Projects/lapshin/Detectron2_datasets"
VITP_PATH="/home/s0213/_scratch2/ViT-P"

GPUS=1
NODE_PARAMS="-p hiperf --gres=gpu:a100:1 --nodelist=tesla-a101 -t 00:30:00"

cp -r dinov2 $SLURM_TMPDIR
unzip -q $DATASET_PATH/ADEChallengeData2016.zip -d $SLURM_TMPDIR/ViT-P/datasets
echo "*****Images are ready*****"

source /opt/conda/etc/profile.d/conda.sh
conda activate vitp


export PYTHONPATH=$PYTHONPATH:$VITP_PATH

cd $SLURM_TMPDIR
cd ViT-P

sbatch -n1 \
    --cpus-per-task=8 \
    --mem=45000 \
    $NODE_PARAMS \
    --job-name="vitp-eval" \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS} \
    --wrap="python dinov2/train/train.py \
        --config-file dinov2/configs/train/vitb14_ADE20k.yaml \
        --output-dir ./OUTPUT_DIR \
        --no-resume"

tar -cf ~/projects/model_ADE20k_base.tar checkpoint.pth
