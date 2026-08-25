#!/usr/bin/env bash
# ============================================================================
# run_inference.sh
# Run streaming inference and re-synthesis with the Data2Vec student model
# and CosyVoice flow+hift vocoder.
# ============================================================================

set -euo pipefail

STREAMASR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- environment (override via env) ----------------------------------------
PYTHON=${PYTHON:-python}
LIBRITTS_ROOT=${LIBRITTS_ROOT:?set LIBRITTS_ROOT to your LibriTTS root}
STUDENT_CKPT=${STUDENT_CKPT:?set STUDENT_CKPT to the tokenizer checkpoint (epoch_*.pt)}
OUT_DIR=${OUT_DIR:-${STREAMASR_ROOT}/results/inference/streamalign_recon}
TEST_CSV=${TEST_CSV:-${LIBRITTS_ROOT}/csv/test-clean.csv}
CHAR_ASR_CKPT=${CHAR_ASR_CKPT:-${STREAMASR_ROOT}/results/char_asr_ckpt}
WORD_ASR_CKPT=${WORD_ASR_CKPT:-${STREAMASR_ROOT}/train/results/conformer_transducer_char/word_fastemit/save/word_asr_ckpt}
BOUNDARY_CKPT=${BOUNDARY_CKPT:-${STREAMASR_ROOT}/train/results/boundary_classifier/save/best_model.pt}
# ----------------------------------------------------------------------------

NUM_GPUS=$("${PYTHON}" -c "import torch; print(max(1, torch.cuda.device_count()))")

"${PYTHON}" "${STREAMASR_ROOT}/inference/inference_core.py" \
    --input_dir="${LIBRITTS_ROOT}" \
    --output_dir="${OUT_DIR}" \
    --split="test-clean" \
    --output_split="16k-test-clean" \
    --batch_size=1 \
    --chunk_size=4 \
    --left_context=32 \
    --hparams="${STREAMASR_ROOT}/hparams/alignment.yaml" \
    --truthmodel_checkpoint_path="${CHAR_ASR_CKPT}" \
    --studentmodel_checkpoint_path="${STUDENT_CKPT}" \
    --test_csv="${TEST_CSV}" \
    --word_hparams="${STREAMASR_ROOT}/hparams/chunk_streaming_word_fastemit.yaml" \
    --word_checkpoint="${WORD_ASR_CKPT}" \
    --boundary_classifier_ckpt="${BOUNDARY_CKPT}" \
    --world_size="${NUM_GPUS}" \
    --tokenizer="llama"
