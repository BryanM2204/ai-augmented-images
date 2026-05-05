#!/bin/bash
#SBATCH --job-name=vit_eval_v6
#SBATCH --output=logs/vit_eval_v6_%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100"
#SBATCH --mem=48GB
#SBATCH --partition=general-gpu
#SBATCH --time=00:30:00

# activate ai-images 
module purge
module load gcc/12.2.0
module load cuda/12.2
source /home/bam20007/miniconda3/etc/profile.d/conda.sh
conda activate ai-images

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# run the training script
echo "Starting evaluation at $(date)"
python ../../model_v6_vit_base/eval.py

echo "Finished evaluation at $(date)"