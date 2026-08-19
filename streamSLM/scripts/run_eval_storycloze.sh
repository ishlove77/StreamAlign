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
SLM_CKPT=${SLM_CKPT:-${REPO_ROOT}/checkpoints/streamSLM/streamalign_slm_r16/step_00185000.pt}
TEACHER_CKPT=${TEACHER_CKPT:-/home/streamalign/streamASR/checkpoints/streamalign_r16/epoch_22.pt}
DATA_ROOT=${DATA_ROOT:-/home/datasets/StoryCloze}
OUT_DIR=${OUT_DIR:-${REPO_ROOT}/results/storycloze}
LOG_DIR=${LOG_DIR:-${REPO_ROOT}/logs/streamSLM_eval}
mkdir -p "${LOG_DIR}"

DATASETS=${DATASETS:-"sSC tSC"}

VARIANT=${VARIANT:-rvq}
# RVQ knobs.
RVQ_R=${RVQ_R:-16}
RVQ_CODEBOOK_SIZE=${RVQ_CODEBOOK_SIZE:-512}
SCORING_MODE=${SCORING_MODE:-loss}
SR_EXCLUDE=${SR_EXCLUDE:-kandinsky,greco,namjune,matisse}

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

sr 1 24 --qos=q-low --exclude="${SR_EXCLUDE}" \
  python -m streamSLM.eval.storycloze \
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
