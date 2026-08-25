#!/usr/bin/env bash
# SALMon + StoryCloze eval on the R=16 hier-durfirst-durreg ckpt.
#
# Single-GPU SALMon (~30-60 min) followed by 8-way parallel StoryCloze
# (~30 min). Both write to results/{salmon,storycloze}/<ckpt_tag>/loss/.
# Set LAUNCHER to submit through your scheduler (e.g. "sr 1 24 --qos=q-low").

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

# --- Checkpoints --------------------------------------------------------- #
CHECKPOINTS_ROOT=${CHECKPOINTS_ROOT:-${REPO_ROOT}/weights}
SLM_CKPT=${SLM_CKPT:-${CHECKPOINTS_ROOT}/Streamalign-SLM-R16/step_00185000.pt}
TEACHER_CKPT=${TEACHER_CKPT:-${CHECKPOINTS_ROOT}/Streamalign-R16/rvq_teacher/epoch_22.pt}

# --- Quantizer knobs ----------------------------------------------------- #
VARIANT=rvq
RVQ_R=16
RVQ_CODEBOOK_SIZE=512

# --- StoryCloze parallelism --------------------------------------------- #
WORLD=${WORLD:-8}

LOG_DIR=${LOG_DIR:-${REPO_ROOT}/logs/streamSLM_eval}
mkdir -p "${LOG_DIR}"

export SLM_CKPT TEACHER_CKPT VARIANT RVQ_R RVQ_CODEBOOK_SIZE WORLD

echo "[hier_R16] SLM_CKPT=${SLM_CKPT}"
echo "[hier_R16] TEACHER_CKPT=${TEACHER_CKPT}"
echo "[hier_R16] VARIANT=${VARIANT} RVQ_R=${RVQ_R}"

echo "[hier_R16] launching SALMon (single-GPU)"
nohup bash streamSLM/scripts/run_eval_salmon.sh \
  > "${LOG_DIR}/hier_R16_salmon_rvq.log" 2>&1 &
SALMON_PID=$!

echo "[hier_R16] launching StoryCloze (${WORLD}-way parallel)"
nohup bash streamSLM/scripts/run_eval_storycloze_parallel.sh \
  > "${LOG_DIR}/hier_R16_storycloze_rvq.log" 2>&1 &
STORYCLOZE_PID=$!

echo "[hier_R16] SALMon PID=${SALMON_PID}  StoryCloze PID=${STORYCLOZE_PID}"
echo "[hier_R16] tail logs/streamSLM_eval/hier_R16_*.log to follow"
