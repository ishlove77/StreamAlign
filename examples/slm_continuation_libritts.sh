#!/usr/bin/env bash
# =============================================================================
# SLM example: LibriTTS speech continuation.
#
# Takes short LibriTTS prompts, encodes them into StreamAlign R=16 units with
# the tokenizer, continues them autoregressively with the StreamAlign SLM, and
# decodes the generated units back to audio. Each output is the prompt followed
# by the model's continuation, so you can hear whether the SLM carries the
# speaker and the sentence forward.
#
# Usage:
#   bash examples/slm_continuation_libritts.sh [N_PROMPTS]
#
# Environment:
#   WEIGHTS         weights dir from download_weights.sh   (default ./weights)
#   LIBRITTS        LibriTTS root
#   COSYVOICE_ROOT  CosyVoice checkout (provides the flow/HiFT decoder)
#   MOSHI_ROOT      moshi checkout (StreamingTransformer for the hier SLM)
#   OUT_DIR         where continuations are written
#   PROMPT_DIR      pre-built prompt dir; skips sampling from LibriTTS
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

N_PROMPTS="${1:-5}"
: "${WEIGHTS:=${REPO_ROOT}/weights}"
: "${LIBRITTS:=${LIBRITTS_ROOT:-/data/LibriTTS}}"
: "${COSYVOICE_ROOT:=${REPO_ROOT}/streamASR/third_party/CosyVoice}"
: "${OUT_DIR:=${REPO_ROOT}/outputs/continuation}"
: "${PROMPT_DIR:=${REPO_ROOT}/outputs/continuation_prompts}"
: "${SPLIT:=test-clean}"
# How much of each prompt to condition on, and how much to generate.
# max_new_tokens is counted in subwords.
: "${PROMPT_MAX_SUBWORDS:=20}"
: "${MAX_NEW_TOKENS:=16}"

W="${WEIGHTS}/Streamalign-R16"
SLM_CKPT="${SLM_CKPT:-${WEIGHTS}/Streamalign-SLM-R16/step_00185000.pt}"

for f in "${W}/rvq_teacher/epoch_22.pt" "${SLM_CKPT}"; do
    if [ ! -f "${f}" ]; then
        echo "Missing weights: ${f}" >&2
        echo "Run: bash examples/download_weights.sh" >&2
        exit 1
    fi
done

# ---- Build a small prompt directory from LibriTTS ---------------------------
if [ ! -d "${PROMPT_DIR}" ] || [ -z "$(ls -A "${PROMPT_DIR}" 2>/dev/null)" ]; then
    if [ ! -d "${LIBRITTS}/${SPLIT}" ]; then
        echo "LibriTTS split not found: ${LIBRITTS}/${SPLIT}" >&2
        echo "Set LIBRITTS=<root>, or point PROMPT_DIR at your own .wav files." >&2
        exit 1
    fi
    mkdir -p "${PROMPT_DIR}"
    echo "Sampling ${N_PROMPTS} prompts from ${LIBRITTS}/${SPLIT} -> ${PROMPT_DIR}"
    # Sorted order keeps the selection reproducible.
    find "${LIBRITTS}/${SPLIT}" -name "*.wav" | sort | head -n "${N_PROMPTS}" | \
        while read -r wav; do cp "${wav}" "${PROMPT_DIR}/"; done
fi
echo "prompts: $(ls "${PROMPT_DIR}" | wc -l) wav(s) in ${PROMPT_DIR}"

# ---- Run continuation -------------------------------------------------------
export PYTHONUNBUFFERED=1
# R must match both the tokenizer checkpoint and the SLM's unit vocabulary.
export RVQ_R=16 RVQ_CODEBOOK_SIZE=512
unset RVQ_CODEBOOK_DIM
# sdpa attention: the flash_attention_2 default hangs on mixed-generation GPUs.
export STREAMSLM_ATTN_IMPL=sdpa
# streamSLM imports streamASR (tokenizer) and CosyVoice (flow/HiFT decoder).
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/streamASR:${COSYVOICE_ROOT}:${COSYVOICE_ROOT}/third_party/Matcha-TTS:${PYTHONPATH:-}"
export COSYVOICE_ROOT
# The hierarchical SLM needs Moshi's StreamingTransformer (path-based import).
: "${MOSHI_ROOT:=${HOME}/moshi/moshi}"
export MOSHI_ROOT

mkdir -p "${OUT_DIR}"
cd "${REPO_ROOT}"

python -u streamSLM/scripts/examples/_run_speech_cont_sdpa.py \
    --checkpoint "${SLM_CKPT}" \
    --teacher_checkpoint "${W}/rvq_teacher/epoch_22.pt" \
    --variant rvq \
    --input_wav_dir "${PROMPT_DIR}" \
    --out_dir "${OUT_DIR}" \
    --max_utts "${N_PROMPTS}" \
    --prompt_max_subwords "${PROMPT_MAX_SUBWORDS}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --hparams "${W}/alignment.yaml" \
    --truthmodel_checkpoint_path "${W}/alignment_model" \
    --word_asr_hparams "${W}/streaming_asr/chunk_streaming_word_fastemit.yaml" \
    --word_asr_checkpoint "${W}/streaming_asr" \
    --word_asr_tokenizer "${W}/streaming_asr/tokenizer.ckpt" \
    --cosyvoice_model_dir "${COSYVOICE_ROOT}/pretrained_models/Fun-CosyVoice3-0.5B" \
    --temperature_text 0.9 --top_p_text 0.95 \
    --seed 1 --bf16

echo
echo "Continuations written to ${OUT_DIR}"
