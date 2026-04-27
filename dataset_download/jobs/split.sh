#!/bin/bash
#SBATCH --job-name=split # Job name
#SBATCH --output=logs/split_%j.out   # Standard output log (%j inserts JobID)
#SBATCH --partition=general             # Use a general compute partition
#SBATCH --nodes=1                       # Run on a single node
#SBATCH --cpus-per-task=8               # Use multiple cores for parallel workers
#SBATCH --mem=16G                        # Memory requirement
#SBATCH --time=4:00:00                 # Max wall time (adjust based on internet speed)

# activate conda environemnt - ai-iamges
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ai-images

echo "Creating splits for training, validation, and testing..."
python ../../preprocess/splits.py

echo "Splitting completed."