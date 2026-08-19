#!/usr/bin/env bash
# ============================================================================
# Example: plain RVQ inference (mirror of scripts/rvq/run_train_rvq_example.sh).
# Reconstructs wavs from a checkpoint trained by that script, on a subset of
# LibriSpeech test-clean. Copy and adjust the env vars below for your run.
#
# Configurable knobs (must match the values the checkpoint was trained with —
# they are read by models/model_tokenizer.py at construction):
#   RVQ_R              RVQ depth (number of residual codebooks).
#   RVQ_CODEBOOK_SIZE  Per-layer codebook size.
#   D_HIDDEN           Decoder transformer hidden dim (default 512).
#
# Required:
#   EXP_NAME           Used for the output dir name.
#   STUDENT_CKPT       Absolute path to epoch_*.pt to evaluate.
#
# Optional:
#   TEST_CSV           Subset CSV. Default: data/groundtruth_test-clean_1of10.csv
#   OUTPUT_DIR         Where reconstructed wavs go. Default:
#                      results/inference/${EXP_NAME}/test-clean_1of10
#   BATCH_SIZE         Default 1 (forward_streaming is per-utterance).
#   CHUNK_SIZE         Default 4 (match training).
#   LEFT_CONTEXT       Default 32 (match training).
#
# Run:
#   sr 1 24 --qos=q-low bash scripts/inference/run_inference_rvq_example.sh
# ============================================================================

set -euo pipefail

STREAMASR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_ROOT="/home/streamalign/streamASR"

export PYTHONPATH="/home/CosyVoice:/home/CosyVoice/third_party/Matcha-TTS:${PYTHONPATH:-}"
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1

# --- knobs (must match the training run that produced STUDENT_CKPT) -------
export RVQ_R=8                  # match training (try 8, 16, 32)
export RVQ_CODEBOOK_SIZE=512    # match training (try 512, 64, 16)
export D_HIDDEN=512             # match training (default 512)
# --------------------------------------------------------------------------

: "${EXP_NAME:?EXP_NAME is required (used for output dir naming)}"
: "${STUDENT_CKPT:?STUDENT_CKPT is required (absolute path to epoch_*.pt)}"
: "${TEST_CSV:=${STREAMASR_ROOT}/data/groundtruth_test-clean_1of10.csv}"
: "${OUTPUT_DIR:=${STREAMASR_ROOT}/results/inference/${EXP_NAME}/test-clean_1of10}"
: "${BATCH_SIZE:=1}"
: "${CHUNK_SIZE:=4}"
: "${LEFT_CONTEXT:=32}"

NUM_GPUS=$(python -c "import torch; print(max(1, torch.cuda.device_count()))")

echo "============================================================"
echo "[infer-example] $(date -Iseconds)"
echo "  EXP_NAME           = ${EXP_NAME}"
echo "  STUDENT_CKPT       = ${STUDENT_CKPT}"
echo "  TEST_CSV           = ${TEST_CSV}"
echo "  OUTPUT_DIR         = ${OUTPUT_DIR}"
echo "  RVQ_R              = ${RVQ_R}"
echo "  RVQ_CODEBOOK_SIZE  = ${RVQ_CODEBOOK_SIZE}"
echo "  D_HIDDEN           = ${D_HIDDEN}"
echo "  CHUNK_SIZE         = ${CHUNK_SIZE}"
echo "  LEFT_CONTEXT       = ${LEFT_CONTEXT}"
echo "  NUM_GPUS           = ${NUM_GPUS}"
echo "============================================================"

mkdir -p "${OUTPUT_DIR}"
cd "${STREAMASR_ROOT}"

python "${STREAMASR_ROOT}/inference/inference_core.py" \
    --variant=rvq \
    --input_dir="/home/datasets/experiments/streamalign/groundtruth" \
    --output_dir="${OUTPUT_DIR}" \
    --split="test-clean" \
    --output_split="test-clean" \
    --batch_size="${BATCH_SIZE}" \
    --chunk_size="${CHUNK_SIZE}" \
    --left_context="${LEFT_CONTEXT}" \
    --hparams="${STREAMASR_ROOT}/hparams/alignment.yaml" \
    --truthmodel_checkpoint_path="${REMOTE_ROOT}/results/char_asr_ckpt" \
    --studentmodel_checkpoint_path="${STUDENT_CKPT}" \
    --test_csv="${TEST_CSV}" \
    --word_hparams="${REMOTE_ROOT}/hparams/chunk_streaming_word_fastemit.yaml" \
    --word_checkpoint="${REMOTE_ROOT}/train/results/conformer_transducer_char/word_fastemit/save/word_asr_ckpt" \
    --boundary_classifier_ckpt="${REMOTE_ROOT}/train/results/best_model.pt" \
    --world_size="${NUM_GPUS}" \
    --tokenizer="llama"

echo "[infer-example] DONE $(date -Iseconds)"
