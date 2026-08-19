#!/usr/bin/env bash
# train_boundary_classifier.sh
#
# Step 2 (BoundaryClassifier pipeline): Train BoundaryClassifier on the
# (enc_feat, pred_feat, label) dataset produced by create_boundary_dataset.sh.
#
# Usage
# -----
#   bash train_boundary_classifier.sh                  # default paths
#   bash train_boundary_classifier.sh --no-resume      # train from scratch
#   bash train_boundary_classifier.sh --gpu 1          # select GPU

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Defaults ─────────────────────────────────────────────────────────────────
HPARAMS="${SCRIPT_DIR}/hparams/boundary_classifier.yaml"
NO_RESUME_ARG=""
GPU_ARG=""

# ── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --hparams)    HPARAMS="$2";          shift 2 ;;
        --no-resume)  NO_RESUME_ARG="--no_resume"; shift ;;
        --gpu)        GPU_ARG="CUDA_VISIBLE_DEVICES=$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

echo "========================================================"
echo "Training BoundaryClassifier"
echo "  Hparams : ${HPARAMS}"
echo "========================================================"

# shellcheck disable=SC2086
env ${GPU_ARG} python "${SCRIPT_DIR}/train/train_boundary_classifier.py" \
    "${HPARAMS}"    \
    ${NO_RESUME_ARG}

echo ""
echo "Training complete."
