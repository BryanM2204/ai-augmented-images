#!/bin/bash
#SBATCH --job-name=celebdf_download # Job name
#SBATCH --output=logs/download_%j.out   # Standard output log (%j inserts JobID)
#SBATCH --partition=general             # Use a general compute partition
#SBATCH --nodes=1                       # Run on a single node
#SBATCH --cpus-per-task=4               # Use multiple cores for parallel workers
#SBATCH --mem=8G                        # Memory requirement
#SBATCH --time=11:30:00                 # Max wall time (adjust based on internet speed)