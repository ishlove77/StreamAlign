#!/usr/bin/env bash
# Launch StreamSLM StoryCloze evaluation on one 48 GB GPU under low quota.
#
# Same protocol as run_eval_salmon.sh; sSC has ~3.7k pairs and tSC has
# ~3.7k. End-to-end runtime should be similar to SALMon.
#
# Override knobs via env:
#   SLM_CKPT     : streamSLM checkpoint (.pt)
#   TEACHER_CKPT : streamAlign RVQ teacher
#   DATASETS     : "sSC tSC" (default) or a single subset
#   OUT_DIR      : results dir (auto-namespaced inside)
#   SCORING_MODE : "loss" (default, TASTE-style mean-CE) or "likelihood".
#                  Results land under OUT_DIR/<ckpt_tag>/<mode>/.
#   VARIANT      : teacher variant — "rvq".
#   RVQ_R        : per-RVQ residual quantizers (used when VARIANT=rvq*).
#   RVQ_CODEBOOK_SIZE : per-RVQ codebook size (used when VARIANT=rvq*).

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# Defaults align with the current canonical hier-durfirst-durreg sweep
# (RVQ teacher + R16 SLM). Override SLM_CKPT / TEACHER_CKPT / VARIANT etc.
# via env vars to evaluate other configs.
CHECKPOINTS_ROOT=${CHECKPOINTS_ROOT:-${REPO_ROOT}/weights}
SLM_CKPT=${SLM_CKPT:-${CHECKPOINTS_ROOT}/Streamalign-SLM-R16/step_00185000.pt}
TEACHER_CKPT=${TEACHER_CKPT:-${CHECKPOINTS_ROOT}/Streamalign-R16/rvq_teacher/epoch_22.pt}
DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to your StoryCloze dataset root}
ASR_HPARAMS=${ASR_HPARAMS:-${REPO_ROOT}/streamASR/hparams/alignment.yaml}
TRUTH_MODEL_CKPT=${TRUTH_MODEL_CKPT:-${REPO_ROOT}/streamASR/results/char_asr_ckpt}
COSYVOICE_ROOT=${COSYVOICE_ROOT:-${REPO_ROOT}/streamASR/third_party/CosyVoice}
export COSYVOICE_ROOT
export PYTHONPATH="${COSYVOICE_ROOT}:${COSYVOICE_ROOT}/third_party/Matcha-TTS:${PYTHONPATH:-}"
# Job-submission prefix; empty = run directly. E.g. LAUNCHER="sr 1 24 --qos=q-low"
LAUNCHER=${LAUNCHER:-}
OUT_DIR=${OUT_DIR:-${REPO_ROOT}/results/storycloze}
LOG_DIR=${LOG_DIR:-${REPO_ROOT}/logs/streamSLM_eval}
mkdir -p "${LOG_DIR}"

DATASETS=${DATASETS:-"sSC tSC"}

VARIANT=${VARIANT:-rvq}
# RVQ knobs.
RVQ_R=${RVQ_R:-16}
RVQ_CODEBOOK_SIZE=${RVQ_CODEBOOK_SIZE:-512}
SCORING_MODE=${SCORING_MODE:-loss}

CKPT_TAG="$(basename "$(dirname "${SLM_CKPT}")")_$(basename "${SLM_CKPT}" .pt)"
LOG_FILE="${LOG_DIR}/storycloze_${CKPT_TAG}_${SCORING_MODE}.log"

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
# Force SDPA on 24 GB pool — FlashAttention-2 needs Ampere+ and the
# pool includes Turing/Volta nodes (basquiat, botticelli, rtx6000-class).
export STREAMSLM_ATTN_IMPL="${STREAMSLM_ATTN_IMPL:-sdpa}"

${LAUNCHER} \
  python -m streamSLM.eval.storycloze \
    --hparams "${ASR_HPARAMS}" \
    --truthmodel_checkpoint "${TRUTH_MODEL_CKPT}" \
    --slm_checkpoint "${SLM_CKPT}" \
    --teacher_checkpoint "${TEACHER_CKPT}" \
    --data_root "${DATA_ROOT}" \
    --datasets ${DATASETS} \
    --output_dir "${OUT_DIR}" \
    --variant "${VARIANT}" \
    --rvq_num_quantizers "${RVQ_R}" \
    --rvq_codebook_size "${RVQ_CODEBOOK_SIZE}" \
    --scoring_mode "${SCORING_MODE}" \
    2>&1 | tee "${LOG_FILE}"
