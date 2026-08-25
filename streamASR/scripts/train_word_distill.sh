#!/usr/bin/env bash
# train_word_distill.sh
#
# Step 2: Train the character-level Conformer Transducer using chunk-aligned
# RNNT loss supervised by the TextGrids produced by generate_textgrids.sh.
#
# Each TextGrid interval carries the exact text the word model emitted for
# that 160 ms chunk; the character model is trained to match that per-chunk
# output, chunk by chunk.
#
# Usage
# -----
#   bash train_word_distill.sh                   # default paths
#   bash train_word_distill.sh --tg-dir /path    # custom TextGrid directory
#   bash train_word_distill.sh --gpu 0           # select GPU

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- environment (override via env) ----------------------------------------
PYTHON=${PYTHON:-python}
LIBRITTS_ROOT=${LIBRITTS_ROOT:?set LIBRITTS_ROOT to your LibriTTS root}
LIBRISPEECH_ROOT=${LIBRISPEECH_ROOT:?set LIBRISPEECH_ROOT to your LibriSpeech root}
EMILIA_ROOT=${EMILIA_ROOT:?set EMILIA_ROOT to your Emilia dataset root}
EMILIA_CSV=${EMILIA_CSV:-${EMILIA_ROOT}/emilia_en_400h.csv}
# TextGrids from generate_textgrids.sh (LibriTTS) and its --emilia run.
TEXTGRID_ROOT=${TEXTGRID_ROOT:-${LIBRITTS_ROOT}/chunk_textgrids_word_model_final2}
EMILIA_TEXTGRID_ROOT=${EMILIA_TEXTGRID_ROOT:-${EMILIA_ROOT}/chunk_textgrids_word_model_final2}
# ----------------------------------------------------------------------------

CHAR_HPARAMS="${SCRIPT_DIR}/hparams/chunk_streaming_libritts_distill.yaml"
TG_DIR="${TEXTGRID_ROOT}"
GPU_ARG=""

# ── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tg-dir) TG_DIR="$2"; shift 2 ;;
        --gpu)    GPU_ARG="--device cuda:$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

echo "========================================================"
echo "Training character-level ASR (chunk word distillation)"
echo "  Char model : ${CHAR_HPARAMS}"
echo "  TextGrid   : ${TG_DIR}"
echo "========================================================"

# Yaml output paths are relative; run from train/ so results land under
# streamASR/train/results/ (the canonical location).
cd "${SCRIPT_DIR}/train"

# shellcheck disable=SC2086
"${PYTHON}" "${SCRIPT_DIR}/train/train_asr_word_distill.py" \
    "${CHAR_HPARAMS}"              \
    --textgrid_dir "${TG_DIR}"     \
    --data_folder "${LIBRITTS_ROOT}" \
    --valid_data_folder "${LIBRISPEECH_ROOT}" \
    --emilia_data_folder "${EMILIA_ROOT}" \
    --emilia_train_csv "${EMILIA_CSV}" \
    --emilia_textgrid_dir "${EMILIA_TEXTGRID_ROOT}" \
    ${GPU_ARG}

echo ""
echo "Training complete."
