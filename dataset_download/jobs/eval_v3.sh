#!/bin/bash
#SBATCH --job-name=vit_eval_v3
#SBATCH --output=logs/vit_eval_v3_%j.out
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
conda activate vlm_internvl

# run the training script
echo "Starting evaluation at $(date)"
python ../../model_v2/eval_v2.py

echo "Finished evaluation at $(date)"