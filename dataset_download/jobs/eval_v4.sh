#!/bin/bash
#SBATCH --job-name=vit_eval_v4
#SBATCH --output=logs/vit_eval_v4_USE_THIS_PLEASAE%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100"
#SBATCH --mem=32GB
#SBATCH --partition=general-gpu
#SBATCH --time=11:59:00

# activate ai-images 
module purge
module load gcc/12.2.0
module load cuda/12.2
source /home/bam20007/miniconda3/etc/profile.d/conda.sh
conda activate ai-images

# run the training script
echo "Starting evaluation at $(date)"
python ../../model_v3/eval.py

echo "Finished evaluation at $(date)"