#!/usr/bin/env bash
# Example wrapper around _infer_speech_cont.sh for StreamSLM speech-continuation
# inference on LibriSpeech test-clean. Run:
#
#   bash streamSLM/scripts/examples/infer_speech_cont_example.sh
#
# Outputs:
#   out/slm_rvq_<CKPT_TAG>/<utt>/{gt,cont}.{wav,units.pt}
#   out/slm_rvq_<CKPT_TAG>/summary.jsonl
#   logs/streamSLM_train/infer_rvq_<CKPT_TAG>_launcher.log

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

# ---- environment (override via env) ----------------------------------------
CHECKPOINTS_ROOT=${CHECKPOINTS_ROOT:-${REPO_ROOT}/weights}
CACHE_DIR=${CACHE_DIR:-${REPO_ROOT}/cache/streamSLM_units_C512_R16}
LIBRISPEECH_ROOT=${LIBRISPEECH_ROOT:?set LIBRISPEECH_ROOT to your LibriSpeech root}
# ----------------------------------------------------------------------------

# --- R16 run (the released configuration) -----------------------------------
SLM_CKPT=${SLM_CKPT:-${CHECKPOINTS_ROOT}/Streamalign-SLM-R16/step_00185000.pt}
TEACHER_CKPT=${TEACHER_CKPT:-${CHECKPOINTS_ROOT}/Streamalign-R16/rvq_teacher/epoch_22.pt}
MANIFEST_WORLD=32
RVQ_R=16
CKPT_TAG=R16_step185000
VARIANT=${VARIANT:-rvq}

# --- Prompt / output config (override on CLI if needed) ---------------------
NUM_UTTS=${NUM_UTTS:-5}        # how many test-clean utts to sample
SAMPLE_SEED=${SAMPLE_SEED:-0}  # seed for prompt sampling
PROMPT_MAX_SUBWORDS=${PROMPT_MAX_SUBWORDS:-20}

UNITS_CACHE="${CACHE_DIR}/librispeech/test-clean/test-clean"
MANIFEST_GLOB="${CACHE_DIR}/librispeech/test-clean/manifest_shard*_of${MANIFEST_WORLD}.csv"
LIBRI_TC_ROOT=${LIBRI_TC_ROOT:-${LIBRISPEECH_ROOT}/test-clean}

LOG=${REPO_ROOT}/logs/streamSLM_train/infer_rvq_${CKPT_TAG}_launcher.log
mkdir -p "$(dirname "${LOG}")"
rm -f "${LOG}" "${REPO_ROOT}/logs/streamSLM_train/infer_rvq_${CKPT_TAG}.log"
rm -rf "${REPO_ROOT}/out/slm_rvq_${CKPT_TAG}"

echo "[wrapper] VARIANT=${VARIANT} CKPT_TAG=${CKPT_TAG}"
echo "[wrapper] SLM_CKPT=${SLM_CKPT}"
echo "[wrapper] TEACHER_CKPT=${TEACHER_CKPT}"
echo "[wrapper] UNITS_CACHE=${UNITS_CACHE}"
echo "[wrapper] MANIFEST_GLOB=${MANIFEST_GLOB}"
echo "[wrapper] log -> ${LOG}"

SLM_CKPT="${SLM_CKPT}" \
TEACHER_CKPT="${TEACHER_CKPT}" \
UNITS_CACHE="${UNITS_CACHE}" \
MANIFEST_GLOB="${MANIFEST_GLOB}" \
LIBRI_TC_ROOT="${LIBRI_TC_ROOT}" \
CKPT_TAG="${CKPT_TAG}" \
VARIANT="${VARIANT}" \
RVQ_R="${RVQ_R}" RVQ_CODEBOOK_SIZE=512 \
NUM_UTTS="${NUM_UTTS}" \
SAMPLE_SEED="${SAMPLE_SEED}" \
PROMPT_MAX_SUBWORDS="${PROMPT_MAX_SUBWORDS}" \
nohup bash "${REPO_ROOT}/streamSLM/scripts/_infer_speech_cont.sh" \
  >"${LOG}" 2>&1 &

echo "[wrapper] pid=$! — tail -f ${LOG}"
