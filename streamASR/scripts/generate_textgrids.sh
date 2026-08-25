#!/usr/bin/env bash
# generate_textgrids.sh
#
# Step 1: Stream LibriSpeech audio through the word-level StreamingASR model
# and save chunk-committed text as Praat TextGrid files.
#
# Each TextGrid interval stores exactly the text the word model emitted for
# that 160 ms chunk, with xmin/xmax as exact encoder output frame indices.
#
# Usage
# -----
#   bash generate_textgrids.sh                   # train-clean-100, default paths
#   bash generate_textgrids.sh --full            # + train-clean-100
#   bash generate_textgrids.sh --test 50         # quick run, 50 files per split
#   bash generate_textgrids.sh --tg-dir /path    # custom output directory
#   bash generate_textgrids.sh --overwrite       # regenerate existing TextGrids
#   bash generate_textgrids.sh --ngpus 4         # multi-GPU (4 GPUs in parallel)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- environment (override via env) ----------------------------------------
PYTHON=${PYTHON:-python}
LIBRITTS_ROOT=${LIBRITTS_ROOT:?set LIBRITTS_ROOT to your LibriTTS root}
WORD_ASR_CKPT=${WORD_ASR_CKPT:-${SCRIPT_DIR}/train/results/conformer_transducer_char/word_fastemit/save/word_asr_ckpt}
TEXTGRID_ROOT=${TEXTGRID_ROOT:-${LIBRITTS_ROOT}/chunk_textgrids_word_model_final2}
# ----------------------------------------------------------------------------

# ── Defaults ─────────────────────────────────────────────────────────────────
DATA_DIR="${LIBRITTS_ROOT}"
WORD_HPARAMS="${SCRIPT_DIR}/hparams/chunk_streaming_word_fastemit.yaml"
WORD_CKPT="${WORD_ASR_CKPT}"
TG_DIR="${TEXTGRID_ROOT}"
CHUNK_SIZE=4
LEFT_CONTEXT=32
SPLITS="${SPLITS:-dev-clean test-clean}"
MAX_FILES_ARG=""
OVERWRITE_ARG=""
EMILIA_CSV=""
DATA_ROOT_ARG=""
NGPUS=1

# ── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --full)        SPLITS="train-clean-360"; shift ;;
        --test)        MAX_FILES_ARG="--max_files $2"; shift 2 ;;
        --tg-dir)      TG_DIR="$2"; shift 2 ;;
        --overwrite)   OVERWRITE_ARG="--overwrite"; shift ;;
        --chunk-size)  CHUNK_SIZE="$2"; shift 2 ;;
        --left-context) LEFT_CONTEXT="$2"; shift 2 ;;
        --emilia)
            EMILIA_ROOT="${EMILIA_ROOT:?set EMILIA_ROOT to your Emilia dataset root}"
            EMILIA_CSV="${EMILIA_CSV:-${EMILIA_ROOT}/emilia_en_400h.csv}"
            DATA_ROOT_ARG="--data_root ${EMILIA_ROOT}"
            TG_DIR="${EMILIA_TEXTGRID_ROOT:-${EMILIA_ROOT}/chunk_textgrids_word_model_final2}"
            shift ;;
        --emilia-csv)  EMILIA_CSV="$2"; shift 2 ;;
        --data-root)   DATA_ROOT_ARG="--data_root $2"; shift 2 ;;
        --ngpus)       NGPUS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

CSV_MANIFEST_ARG=""
if [[ -n "${EMILIA_CSV}" ]]; then
    CSV_MANIFEST_ARG="--csv_manifest ${EMILIA_CSV}"
fi

echo "========================================================"
echo "Generating chunk-level TextGrids"
echo "  Word model : ${WORD_HPARAMS}"
echo "  Checkpoint : ${WORD_CKPT}"
if [[ -n "${EMILIA_CSV}" ]]; then
echo "  CSV        : ${EMILIA_CSV}"
else
echo "  Splits     : ${SPLITS}"
fi
echo "  Chunk size : ${CHUNK_SIZE} enc-frames ($(( CHUNK_SIZE * 40 )) ms)"
echo "  GPUs       : ${NGPUS}"
echo "  Output dir : ${TG_DIR}"
echo "========================================================"

if [[ "${NGPUS}" -le 1 ]]; then
    # ── Single-GPU mode ──────────────────────────────────────────────────────
    # shellcheck disable=SC2086
    "${PYTHON}" "${SCRIPT_DIR}/data/generate_chunk_textgrids.py" \
        --hparams_file  "${WORD_HPARAMS}"   \
        --checkpoint    "${WORD_CKPT}"      \
        --input_dir     "${DATA_DIR}"       \
        --splits        ${SPLITS}           \
        --chunk_size    "${CHUNK_SIZE}"     \
        --left_context  "${LEFT_CONTEXT}"   \
        --output_dir    "${TG_DIR}"         \
        ${CSV_MANIFEST_ARG}                 \
        ${DATA_ROOT_ARG}                    \
        ${MAX_FILES_ARG}                    \
        ${OVERWRITE_ARG}
else
    # ── Multi-GPU mode ───────────────────────────────────────────────────────
    PIDS=()
    for (( RANK=0; RANK<NGPUS; RANK++ )); do
        echo "Launching rank ${RANK}/${NGPUS} on GPU ${RANK} ..."
        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES="${RANK}" \
        "${PYTHON}" "${SCRIPT_DIR}/data/generate_chunk_textgrids.py" \
            --hparams_file  "${WORD_HPARAMS}"   \
            --checkpoint    "${WORD_CKPT}"      \
            --input_dir     "${DATA_DIR}"       \
            --splits        ${SPLITS}           \
            --chunk_size    "${CHUNK_SIZE}"     \
            --left_context  "${LEFT_CONTEXT}"   \
            --output_dir    "${TG_DIR}"         \
            --rank          "${RANK}"           \
            --world_size    "${NGPUS}"          \
            ${CSV_MANIFEST_ARG}                 \
            ${DATA_ROOT_ARG}                    \
            ${MAX_FILES_ARG}                    \
            ${OVERWRITE_ARG} &
        PIDS+=($!)
    done

    echo "Waiting for ${NGPUS} GPU workers to finish ..."
    FAIL=0
    for PID in "${PIDS[@]}"; do
        if ! wait "${PID}"; then
            echo "ERROR: Worker PID ${PID} failed." >&2
            FAIL=1
        fi
    done
    if [[ "${FAIL}" -ne 0 ]]; then
        echo "Some workers failed. Check logs above." >&2
        exit 1
    fi
fi

echo ""
echo "Done. TextGrids written to: ${TG_DIR}"
