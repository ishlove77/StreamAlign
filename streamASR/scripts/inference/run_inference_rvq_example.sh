#!/usr/bin/env bash
# ============================================================================
# Example: plain RVQ inference. Reconstructs wavs from a trained checkpoint
# on a subset of LibriSpeech test-clean. Copy and adjust the env vars below.
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
#   INPUT_DIR          Ground-truth wav root (contains test-clean/).
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
#   bash scripts/inference/run_inference_rvq_example.sh
# ============================================================================

set -euo pipefail

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

# --- knobs (must match the training run that produced STUDENT_CKPT) -------
export RVQ_R=${RVQ_R:-16}                       # match training
export RVQ_CODEBOOK_SIZE=${RVQ_CODEBOOK_SIZE:-512}  # match training
export D_HIDDEN=${D_HIDDEN:-512}                # match training (default 512)
# --------------------------------------------------------------------------

: "${EXP_NAME:?EXP_NAME is required (used for output dir naming)}"
: "${STUDENT_CKPT:?STUDENT_CKPT is required (absolute path to epoch_*.pt)}"
: "${INPUT_DIR:?INPUT_DIR is required (ground-truth wav root)}"
: "${TEST_CSV:=${STREAMASR_ROOT}/data/groundtruth_test-clean_1of10.csv}"
: "${OUTPUT_DIR:=${STREAMASR_ROOT}/results/inference/${EXP_NAME}/test-clean_1of10}"
: "${BATCH_SIZE:=1}"
: "${CHUNK_SIZE:=4}"
: "${LEFT_CONTEXT:=32}"

NUM_GPUS=$("${PYTHON}" -c "import torch; print(max(1, torch.cuda.device_count()))")

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

"${PYTHON}" "${STREAMASR_ROOT}/inference/inference_core.py" \
    --variant=rvq \
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

echo "[infer-example] DONE $(date -Iseconds)"
