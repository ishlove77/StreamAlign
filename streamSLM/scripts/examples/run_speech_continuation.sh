#!/usr/bin/env bash
# Speech continuation from /home/.../continuation/input_wavs/, using the
# two delay2 hier-durfirst-durreg emilia4000h checkpoints (latest).
#
# Runs the released R16 configuration on one mid-quota 48 GB GPU.
# Outputs land under
#   /home/datasets/continuation/<exp-tag>/

set -uo pipefail

REPO_ROOT=/home/streamalign
cd "${REPO_ROOT}"

INPUT_WAV_DIR=${INPUT_WAV_DIR:-/home/datasets/continuation/input_wavs}
CONT_ROOT=${CONT_ROOT:-/home/datasets/continuation}
LOG_DIR=${REPO_ROOT}/logs/streamSLM_train
mkdir -p "${LOG_DIR}"

# Inference knobs (mid quota, 48 GB GPU each).
VRAM=${VRAM:-48}
SR_EXCLUDE=${SR_EXCLUDE:-kandinsky,greco,namjune,matisse}
# max_new_tokens=16 mirrors TASTE-SpokenLM's extra_words=16 (their gen length).
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-16}
# Canonical _infer_speech_cont.sh defaults prompt_max_subwords=20.
PROMPT_MAX_SUBWORDS=${PROMPT_MAX_SUBWORDS:-20}
MAX_UTTS=${MAX_UTTS:-0}

# Vocab-mask: restrict text-head decoding to the lower/space/apos subword set
# used by every streamSLM continuation run.
ALLOWED_TEXT_TOKEN_IDS=${ALLOWED_TEXT_TOKEN_IDS:-${REPO_ROOT}/cache/allowed_text_token_ids/vocab_lower_space_apos.pt}

# SDPA is safe on 48 GB pool (Ampere+) but doesn't hurt to pin.
export STREAMSLM_ATTN_IMPL=${STREAMSLM_ATTN_IMPL:-sdpa}

run_one() {
  local TAG=$1
  local SLM=$2
  local TEACHER=$3
  local R=$4
  # NOTE: we intentionally do NOT set RVQ_CODEBOOK_DIM — every existing
  # inference path (`_infer_speech_cont.sh`) defaults codebook_dim
  # to feat_dim=256, including for the R32 "d16" cache. The "d16" in
  # the cache name is a separate knob; setting RVQ_CODEBOOK_DIM=16 here
  # produces a teacher-checkpoint shape mismatch.

  local OUT_DIR="${CONT_ROOT}/${TAG}"
  local LAUNCH_LOG="${LOG_DIR}/cont_wavs_${TAG}_launcher.log"
  local RUN_LOG="${LOG_DIR}/cont_wavs_${TAG}.log"
  mkdir -p "${OUT_DIR}"
  rm -f "${LAUNCH_LOG}" "${RUN_LOG}"

  echo "[launch] ${TAG} out=${OUT_DIR} log=${RUN_LOG}"

  # Mid quota (per CLAUDE.md default): no --qos flag.
  # Optional override via SR_QOS=q-low / <your-qos>.
  QOS_FLAG=()
  if [[ -n "${SR_QOS:-}" ]]; then
    QOS_FLAG=(--qos="${SR_QOS}")
  fi
  # NOTE: invoke through _run_speech_cont_sdpa.py rather than `-m
  # streamSLM.inference.speech_cont_from_wavs`. The wrapper pins
  # STREAMSLM_ATTN_IMPL=sdpa from inside Python so the choice survives the
  # `env(1) -> sr -> srun --pty -> nohup` chain that has silently dropped
  # env vars on us before, sending the model down the flash_attention_2
  # default path and hanging at startup on 24 GB nodes.
  nohup env \
    PYTHONUNBUFFERED=1 \
    RVQ_R="${R}" RVQ_CODEBOOK_SIZE=512 \
    STREAMSLM_ATTN_IMPL="${STREAMSLM_ATTN_IMPL}" \
    sr 1 "${VRAM}" "${QOS_FLAG[@]}" --exclude="${SR_EXCLUDE}" \
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

run_one streamalign_slm_r16 \
  "${REPO_ROOT}/checkpoints/streamSLM/streamalign_slm_r16/latest.pt" \
  "/home/streamalign/streamASR/checkpoints/streamalign_r16/epoch_22.pt" \
  16

wait
echo "[launch] submitted; tail logs/streamSLM_train/cont_wavs_*.log"
