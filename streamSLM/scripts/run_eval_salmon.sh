#!/usr/bin/env bash
# Launch StreamSLM SALMon evaluation on one 48 GB GPU under low quota.
#
# All 48 GB nodes were full at job-submit time, so q-low queues this until a
# slot frees up. Steady-state cost is ~1× streamAlign-teacher forward + 1×
# StreamSLM forward per audio; SALMon's 10×400×2 = 8000 utterances should
# run in roughly 30–60 min on rtx6000-class hardware.
#
# Override knobs via env:
#   SLM_CKPT     : streamSLM checkpoint (.pt)
#   TEACHER_CKPT : streamAlign RVQ teacher used to extract training units
#   DATASETS     : space-separated list of SALMon task folders (default = all 10)
#   OUT_DIR      : per-checkpoint results directory (auto-namespaced inside)
#   SCORING_MODE : "loss" (default, TASTE-style sum of per-stream mean CE)
#                  or "likelihood" (sum-reduction joint log-likelihood).
#                  Results land under OUT_DIR/<ckpt_tag>/<mode>/.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# Defaults align with the current canonical hier-durfirst-durreg sweep
# (RVQ teacher + R16 SLM). Override SLM_CKPT / TEACHER_CKPT / VARIANT etc.
# via env vars to evaluate other configs.
CHECKPOINTS_ROOT=${CHECKPOINTS_ROOT:-${REPO_ROOT}/weights}
SLM_CKPT=${SLM_CKPT:-${CHECKPOINTS_ROOT}/Streamalign-SLM-R16/step_00185000.pt}
TEACHER_CKPT=${TEACHER_CKPT:-${CHECKPOINTS_ROOT}/Streamalign-R16/rvq_teacher/epoch_22.pt}
DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to your SALMon dataset root}
ASR_HPARAMS=${ASR_HPARAMS:-${REPO_ROOT}/streamASR/hparams/alignment.yaml}
TRUTH_MODEL_CKPT=${TRUTH_MODEL_CKPT:-${REPO_ROOT}/streamASR/results/char_asr_ckpt}
COSYVOICE_ROOT=${COSYVOICE_ROOT:-${REPO_ROOT}/streamASR/third_party/CosyVoice}
export COSYVOICE_ROOT
export PYTHONPATH="${COSYVOICE_ROOT}:${COSYVOICE_ROOT}/third_party/Matcha-TTS:${PYTHONPATH:-}"
# Job-submission prefix; empty = run directly. E.g. LAUNCHER="sr 1 24 --qos=q-low"
LAUNCHER=${LAUNCHER:-}
# TGs are written alongside the wavs (see run_eval_tg_gen.sh); only set
# TG_ROOT if they live in a separate mirror.
TG_ROOT=${TG_ROOT:-${DATA_ROOT}}
OUT_DIR=${OUT_DIR:-${REPO_ROOT}/results/salmon}
LOG_DIR=${LOG_DIR:-${REPO_ROOT}/logs/streamSLM_eval}
mkdir -p "${LOG_DIR}"

DATASETS=${DATASETS:-"energy_consistency gender_consistency pitch_consistency sentiment_consistency speaker_consistency"}

VARIANT=${VARIANT:-rvq}
# RVQ knobs.
RVQ_R=${RVQ_R:-16}
RVQ_CODEBOOK_SIZE=${RVQ_CODEBOOK_SIZE:-512}
SCORING_MODE=${SCORING_MODE:-loss}

CKPT_TAG="$(basename "$(dirname "${SLM_CKPT}")")_$(basename "${SLM_CKPT}" .pt)"
LOG_FILE="${LOG_DIR}/salmon_${CKPT_TAG}_${SCORING_MODE}.log"

echo "[launch] SLM_CKPT=${SLM_CKPT}"
echo "[launch] TEACHER_CKPT=${TEACHER_CKPT}"
echo "[launch] DATASETS=${DATASETS}"
echo "[launch] OUT_DIR=${OUT_DIR}"
echo "[launch] SCORING_MODE=${SCORING_MODE}"
echo "[launch] LOG_FILE=${LOG_FILE}"

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export RVQ_R="${RVQ_R}"
export RVQ_CODEBOOK_SIZE="${RVQ_CODEBOOK_SIZE}"
# Force SDPA — the 24 GB pool is mostly Turing/Volta (rtx6000-class), and
# FlashAttention-2 requires Ampere+. SDPA works everywhere.
export STREAMSLM_ATTN_IMPL="${STREAMSLM_ATTN_IMPL:-sdpa}"

${LAUNCHER} \
  python -m streamSLM.eval.salmon \
    --hparams "${ASR_HPARAMS}" \
    --truthmodel_checkpoint "${TRUTH_MODEL_CKPT}" \
    --slm_checkpoint "${SLM_CKPT}" \
    --teacher_checkpoint "${TEACHER_CKPT}" \
    --data_root "${DATA_ROOT}" \
    --tg_root "${TG_ROOT}" \
    --datasets ${DATASETS} \
    --output_dir "${OUT_DIR}" \
    --variant "${VARIANT}" \
    --rvq_num_quantizers "${RVQ_R}" \
    --rvq_codebook_size "${RVQ_CODEBOOK_SIZE}" \
    --scoring_mode "${SCORING_MODE}" \
    2>&1 | tee "${LOG_FILE}"
