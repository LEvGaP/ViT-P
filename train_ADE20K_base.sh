#!/bin/bash
#SBATCH --job-name=vitp_ade20k
#SBATCH --partition=hiperf
#SBATCH --nodelist=tesla-v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:v100:8
#SBATCH --mem=0
#SBATCH --time=20:00:00
#SBATCH --output=slurm-%j.out


# cp -r dinov2 $SLURM_TMPDIR
# unzip -q ./backup/ADEChallengeData2016/ADEChallengeData2016.zip -d $SLURM_TMPDIR/ViT-P/datasets
# echo "*****Images are ready*****"


# module purge
# module load gcc arrow
# module spider opencv
# module load opencv/4.9.0
source venv311/bin/activate

export PYTHONPATH=$PWD/ViT-P

# cd $SLURM_TMPDIR
cd ViT-P

srun --unbuffered python dinov2/train/train.py --config-file dinov2/configs/OneFormer/vitb14_ADE20k.yaml --output-dir ./OUTPUT_DIR

tar -cf ~/projects/model_ADE20k_base.tar OUTPUT_DIR
