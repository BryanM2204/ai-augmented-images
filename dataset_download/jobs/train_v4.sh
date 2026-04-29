#!/bin/bash
#SBATCH --job-name=vit_finetune_v4
#SBATCH --output=logs/vit_finetune_v4_%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100"
#SBATCH --mem=128GB
#SBATCH --partition=general-gpu
#SBATCH --time=11:59:00

# activate ai-images 
module purge
module load gcc/12.2.0
module load cuda/12.2
source /home/bam20007/miniconda3/etc/profile.d/conda.sh
conda activate ai-images

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# run the training script
echo "Starting training at $(date)"
python ../../model_v4/train.py

echo "Finished training at $(date)"