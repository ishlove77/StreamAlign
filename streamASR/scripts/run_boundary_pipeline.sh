#!/usr/bin/env bash
# run_boundary_pipeline.sh
#
# Full BoundaryClassifier pipeline:
#   Step 1: Create boundary dataset from LibriSpeech
#   Step 2: Train BoundaryClassifier (saves best precision model)
#
# Submit with:
#   sbatch scripts/run_boundary_pipeline.sh

#SBATCH --partition=vram24
#SBATCH --oversubscribe
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -w cerny
##SBATCH --qos=<your-qos>   # uncomment and set your cluster's QoS
#SBATCH --job-name=boundary_cls
#SBATCH --output=slurm_boundary_%j.log

set -euo pipefail

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate streamASR

cd /home/streamalign/streamASR

echo "========================================================"
echo "Step 1: Creating boundary dataset"
echo "========================================================"
bash scripts/create_boundary_dataset.sh

echo ""
echo "========================================================"
echo "Step 2: Training boundary classifier"
echo "========================================================"
bash scripts/train_boundary_classifier.sh

echo ""
echo "Pipeline complete."
