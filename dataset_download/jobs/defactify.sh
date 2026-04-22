#!/bin/bash
#SBATCH --job-name=defactify_download # Job name
#SBATCH --output=logs/defactify_%j.out   # Standard output log (%j inserts JobID)
#SBATCH --partition=general             # Use a general compute partition
#SBATCH --nodes=1                       # Run on a single node
#SBATCH --cpus-per-task=4               # Use multiple cores for parallel workers
#SBATCH --mem=8G                        # Memory requirement
#SBATCH --time=11:30:00                 # Max wall time (adjust based on internet speed)

# activate conda environemnt - ai-iamges
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ai-images


# run the download script
python ../defactify.py --export-images-dir ../data/defactify