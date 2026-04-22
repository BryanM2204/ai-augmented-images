#!/bin/bash
#SBATCH --job-name=defactify_download # Job name
#SBATCH --output=logs/defactify_%j.out   # Standard output log (%j inserts JobID)
#SBATCH --partition=general             # Use a general compute partition
#SBATCH --nodes=1                       # Run on a single node
#SBATCH --cpus-per-task=8               # Use multiple cores for parallel workers
#SBATCH --mem=16G                        # Memory requirement
#SBATCH --time=2:30:00                 # Max wall time (adjust based on internet 

# activate conda environemnt - ai-iamges
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ai-images

# run extraction
python ../../preprocess/frame_extraction.py