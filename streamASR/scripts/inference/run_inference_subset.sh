#!/usr/bin/env bash
# Parameterized reconstruction inference on a data subset.
#
# Required env vars (caller must export):
#   EXP_NAME           e.g. streamalign_r16
#   VARIANT            rvq
#   STUDENT_CKPT       absolute path to epoch_*.pt
#   OUTPUT_DIR         absolute path for reconstructed wavs
#   TEST_CSV           absolute path to subset CSV
# Optional:
#   RVQ_BOT_DIM        e.g. 16            (rvq_bot variant only)
#   BATCH_SIZE         default 1
#   CHUNK_SIZE         default 4
#   LEFT_CONTEXT       default 32

set -euo pipefail

: "${EXP_NAME:?EXP_NAME is required}"
: "${VARIANT:?VARIANT is required}"
: "${STUDENT_CKPT:?STUDENT_CKPT is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
: "${TEST_CSV:?TEST_CSV is required}"
: "${BATCH_SIZE:=1}"
: "${CHUNK_SIZE:=4}"
: "${LEFT_CONTEXT:=32}"

STREAMASR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_ROOT="/home/streamalign/streamASR"

export PYTHONPATH="/home/CosyVoice:/home/CosyVoice/third_party/Matcha-TTS:${PYTHONPATH:-}"
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1

# respective variants; we don't force-default them here to avoid silently
# masking caller mistakes.

NUM_GPUS=$(python -c "import torch; print(max(1, torch.cuda.device_count()))")

echo "============================================================"
echo "[run_inference_subset] $(date -Iseconds)"
echo "  EXP_NAME      = ${EXP_NAME}"
echo "  VARIANT       = ${VARIANT}"
echo "  STUDENT_CKPT  = ${STUDENT_CKPT}"
echo "  OUTPUT_DIR    = ${OUTPUT_DIR}"
echo "  TEST_CSV      = ${TEST_CSV}"
echo "  RVQ_BOT_DIM   = ${RVQ_BOT_DIM:-<unset>}"
echo "  HEAD          = $(git -C "${STREAMASR_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "  NUM_GPUS      = ${NUM_GPUS}"
echo "============================================================"

mkdir -p "${OUTPUT_DIR}"

python "${STREAMASR_ROOT}/inference/inference_core.py" \
    --variant="${VARIANT}" \
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

echo "[run_inference_subset] DONE $(date -Iseconds)"
