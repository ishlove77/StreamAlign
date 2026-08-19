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

# ── Defaults ─────────────────────────────────────────────────────────────────
DATA_DIR="/home/datasets/LibriTTS"
CHAR_HPARAMS="${SCRIPT_DIR}/hparams/chunk_streaming_libritts_distill.yaml"
TG_DIR="${DATA_DIR}/chunk_textgrids_word_model_final2"
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

# shellcheck disable=SC2086
python "${SCRIPT_DIR}/train/train_asr_word_distill.py" \
    "${CHAR_HPARAMS}"              \
    --textgrid_dir "${TG_DIR}"     \
    ${GPU_ARG}

echo ""
echo "Training complete."
