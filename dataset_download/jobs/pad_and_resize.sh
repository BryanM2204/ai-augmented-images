#!/bin/bash
#SBATCH --job-name=pad_and_resize # Job name
#SBATCH --output=logs/pad_and_resize_%j.out   # Standard output log (%j inserts JobID)
#SBATCH --partition=general             # Use a general compute partition
#SBATCH --nodes=1                       # Run on a single node
#SBATCH --cpus-per-task=8               # Use multiple cores for parallel workers
#SBATCH --mem=16G                        # Memory requirement
#SBATCH --time=1:00:00                 # Max wall time (adjust based on internet speed)

# activate conda environemnt - ai-iamges
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ai-images

# run the padding and resizing script
echo "Starting padding and resizing of images..."
python ../../preprocess/pad_and_resize.py
echo "Padding and resizing completed."
