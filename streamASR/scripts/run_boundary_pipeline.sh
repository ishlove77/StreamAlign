#!/usr/bin/env bash
# run_boundary_pipeline.sh
#
# Full BoundaryClassifier pipeline:
#   Step 1: Create boundary dataset from LibriSpeech
#   Step 2: Train BoundaryClassifier (saves best precision model)
#
# Run directly on a GPU node, or submit through your scheduler, e.g.:
#   sbatch --gres=gpu:1 -c 8 scripts/run_boundary_pipeline.sh

set -euo pipefail

STREAMASR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${STREAMASR_ROOT}"

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
