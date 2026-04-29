#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=logs/test_%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100"
#SBATCH --mem=64GB
#SBATCH --partition=general-gpu
#SBATCH --time=00:10:00

# activate ai-images
module purge
module load gcc/12.2.0
module load cuda/12.2
source /home/bam20007/miniconda3/etc/profile.d/conda.sh
conda activate ai-images

# run the testing script
echo "Starting testing at $(date)"
python ./predict.py
echo "Finished testing at $(date)"