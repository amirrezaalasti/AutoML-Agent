#!/bin/bash
#SBATCH --job-name=autogluon_job
#SBATCH --output=autogluon_output_%j.log
#SBATCH --error=autogluon_error_%j.log
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# Activate virtual environment
bash

conda activate AgentSmac_baseline_autogluon

echo "Starting job on `hostname` at `date`"

# Run the python script
python baselines/autogluon/dummy.py

echo "Job finished at `date`"
