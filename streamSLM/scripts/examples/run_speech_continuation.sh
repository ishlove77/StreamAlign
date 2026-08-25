#!/usr/bin/env bash
# Speech continuation from ${INPUT_WAV_DIR}, using the released R16
# checkpoints. Outputs land under ${CONT_ROOT}/<exp-tag>/.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

# ---- environment (override via env) ----------------------------------------
CHECKPOINTS_ROOT=${CHECKPOINTS_ROOT:-${REPO_ROOT}/weights}
SLM_CKPT=${SLM_CKPT:-${CHECKPOINTS_ROOT}/Streamalign-SLM-R16/step_00185000.pt}
TEACHER_CKPT=${TEACHER_CKPT:-${CHECKPOINTS_ROOT}/Streamalign-R16/rvq_teacher/epoch_22.pt}
INPUT_WAV_DIR=${INPUT_WAV_DIR:?set INPUT_WAV_DIR to a directory of prompt wavs}
CONT_ROOT=${CONT_ROOT:-${REPO_ROOT}/out/continuation}
# Job-submission prefix; empty = run directly on this machine.
# Example (Slurm wrapper): LAUNCHER="sr 1 48 --qos=q-low"
LAUNCHER=${LAUNCHER:-}
# ----------------------------------------------------------------------------

LOG_DIR=${REPO_ROOT}/logs/streamSLM_train
mkdir -p "${LOG_DIR}"

# max_new_tokens=16 mirrors TASTE-SpokenLM's extra_words=16 (their gen length).
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-16}
# Canonical _infer_speech_cont.sh defaults prompt_max_subwords=20.
PROMPT_MAX_SUBWORDS=${PROMPT_MAX_SUBWORDS:-20}
MAX_UTTS=${MAX_UTTS:-0}

# Vocab-mask: restrict text-head decoding to the lower/space/apos subword set
# used by every streamSLM continuation run.
ALLOWED_TEXT_TOKEN_IDS=${ALLOWED_TEXT_TOKEN_IDS:-${REPO_ROOT}/cache/allowed_text_token_ids/vocab_lower_space_apos.pt}

export STREAMSLM_ATTN_IMPL=${STREAMSLM_ATTN_IMPL:-sdpa}

run_one() {
  local TAG=$1
  local SLM=$2
  local TEACHER=$3
  local R=$4
  # NOTE: we intentionally do NOT set RVQ_CODEBOOK_DIM — every existing
  # inference path (`_infer_speech_cont.sh`) defaults codebook_dim
  # to feat_dim=256. Setting RVQ_CODEBOOK_DIM=16 here produces a
  # teacher-checkpoint shape mismatch.

  local OUT_DIR="${CONT_ROOT}/${TAG}"
  local LAUNCH_LOG="${LOG_DIR}/cont_wavs_${TAG}_launcher.log"
  local RUN_LOG="${LOG_DIR}/cont_wavs_${TAG}.log"
  mkdir -p "${OUT_DIR}"
  rm -f "${LAUNCH_LOG}" "${RUN_LOG}"

  echo "[launch] ${TAG} out=${OUT_DIR} log=${RUN_LOG}"

  # NOTE: invoke through _run_speech_cont_sdpa.py rather than `-m
  # streamSLM.inference.speech_cont_from_wavs`. The wrapper pins
  # STREAMSLM_ATTN_IMPL=sdpa from inside Python so the choice survives
  # scheduler/nohup chains that can drop env vars, which would send the
  # model down the flash_attention_2 default path.
  nohup env \
    PYTHONUNBUFFERED=1 \
    RVQ_R="${R}" RVQ_CODEBOOK_SIZE=512 \
    STREAMSLM_ATTN_IMPL="${STREAMSLM_ATTN_IMPL}" \
    ${LAUNCHER} \
      python -u "${REPO_ROOT}/streamSLM/scripts/examples/_run_speech_cont_sdpa.py" \
        --checkpoint "${SLM}" \
        --teacher_checkpoint "${TEACHER}" \
        --variant rvq \
        --input_wav_dir "${INPUT_WAV_DIR}" \
        --out_dir "${OUT_DIR}" \
        --max_utts "${MAX_UTTS}" \
        --prompt_max_subwords "${PROMPT_MAX_SUBWORDS}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --allowed_text_token_ids "${ALLOWED_TEXT_TOKEN_IDS}" \
        --temperature_text 0.9 --top_p_text 0.95 \
        --seed 1 --bf16 \
    >"${LAUNCH_LOG}" 2>&1 &
  echo "[launch] ${TAG} pid=$!"
}

run_one streamalign_slm_r16 "${SLM_CKPT}" "${TEACHER_CKPT}" 16

wait
echo "[launch] submitted; tail logs/streamSLM_train/cont_wavs_*.log"
