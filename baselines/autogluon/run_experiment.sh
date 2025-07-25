#!/bin/bash
#SBATCH --job-name=autogluon_job
#SBATCH --output=logs/%x_%A/%a_output.log
#SBATCH --error=logs/%x_%A/%a_error.log
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=1-9

# Load modules or setup environment if needed
# module load cuda/11.7

# Activate conda environment
source ~/.bashrc
conda activate autogluon_baseline

echo "Starting job on $(hostname) at $(date)"

# Run the python script
python baselines/autogluon/run_autogluon.py

echo "Job finished at $(date)"
