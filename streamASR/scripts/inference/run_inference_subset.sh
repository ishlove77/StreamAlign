#!/usr/bin/env bash
# Parameterized reconstruction inference on a data subset.
#
# Required env vars (caller must export):
#   EXP_NAME           e.g. streamalign_r16
#   VARIANT            rvq
#   STUDENT_CKPT       absolute path to epoch_*.pt
#   OUTPUT_DIR         absolute path for reconstructed wavs
#   TEST_CSV           absolute path to subset CSV
#   INPUT_DIR          ground-truth wav root (contains <split>/ subdirs)
# Optional:
#   BATCH_SIZE         default 1
#   CHUNK_SIZE         default 4
#   LEFT_CONTEXT       default 32

set -euo pipefail

: "${EXP_NAME:?EXP_NAME is required}"
: "${VARIANT:?VARIANT is required}"
: "${STUDENT_CKPT:?STUDENT_CKPT is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
: "${TEST_CSV:?TEST_CSV is required}"
: "${INPUT_DIR:?INPUT_DIR is required (ground-truth wav root)}"
: "${BATCH_SIZE:=1}"
: "${CHUNK_SIZE:=4}"
: "${LEFT_CONTEXT:=32}"

STREAMASR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ---- environment (override via env) ----------------------------------------
PYTHON=${PYTHON:-python}
COSYVOICE_ROOT=${COSYVOICE_ROOT:-${STREAMASR_ROOT}/third_party/CosyVoice}
CHAR_ASR_CKPT=${CHAR_ASR_CKPT:-${STREAMASR_ROOT}/results/char_asr_ckpt}
WORD_ASR_CKPT=${WORD_ASR_CKPT:-${STREAMASR_ROOT}/train/results/conformer_transducer_char/word_fastemit/save/word_asr_ckpt}
BOUNDARY_CKPT=${BOUNDARY_CKPT:-${STREAMASR_ROOT}/train/results/boundary_classifier/save/best_model.pt}
# ----------------------------------------------------------------------------

export PYTHONPATH="${COSYVOICE_ROOT}:${COSYVOICE_ROOT}/third_party/Matcha-TTS:${PYTHONPATH:-}"
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1

NUM_GPUS=$("${PYTHON}" -c "import torch; print(max(1, torch.cuda.device_count()))")

echo "============================================================"
echo "[run_inference_subset] $(date -Iseconds)"
echo "  EXP_NAME      = ${EXP_NAME}"
echo "  VARIANT       = ${VARIANT}"
echo "  STUDENT_CKPT  = ${STUDENT_CKPT}"
echo "  OUTPUT_DIR    = ${OUTPUT_DIR}"
echo "  TEST_CSV      = ${TEST_CSV}"
echo "  HEAD          = $(git -C "${STREAMASR_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "  NUM_GPUS      = ${NUM_GPUS}"
echo "============================================================"

mkdir -p "${OUTPUT_DIR}"

"${PYTHON}" "${STREAMASR_ROOT}/inference/inference_core.py" \
    --variant="${VARIANT}" \
    --input_dir="${INPUT_DIR}" \
    --output_dir="${OUTPUT_DIR}" \
    --split="test-clean" \
    --output_split="test-clean" \
    --batch_size="${BATCH_SIZE}" \
    --chunk_size="${CHUNK_SIZE}" \
    --left_context="${LEFT_CONTEXT}" \
    --hparams="${STREAMASR_ROOT}/hparams/alignment.yaml" \
    --truthmodel_checkpoint_path="${CHAR_ASR_CKPT}" \
    --studentmodel_checkpoint_path="${STUDENT_CKPT}" \
    --test_csv="${TEST_CSV}" \
    --word_hparams="${STREAMASR_ROOT}/hparams/chunk_streaming_word_fastemit.yaml" \
    --word_checkpoint="${WORD_ASR_CKPT}" \
    --boundary_classifier_ckpt="${BOUNDARY_CKPT}" \
    --world_size="${NUM_GPUS}" \
    --tokenizer="llama"

echo "[run_inference_subset] DONE $(date -Iseconds)"
