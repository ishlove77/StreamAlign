#!/usr/bin/env bash
# =============================================================================
# StreamAlign-SLM (R16) — unified two-phase training.
#
#   Phase 1  (pretrain) : base training from scratch.
#                         loss_w_text=1.0, text_kl_weight=0.9, max_steps=300k.
#   Phase 2  (finetune) : resume from the phase-1 checkpoint at RESUME_STEP
#                         (180k) with the text-emphasis recipe
#                         loss_w_text=5.0, text_kl_weight=0.5, max_steps=210k.
#                         The step ~185k checkpoint of this phase is the
#                         reported StreamAlign-SLM (SALMon 69.1 / StoryCloze 72.1).
#
#   PHASE=pretrain | finetune | all   (default: all — runs both, chained)
#
# Both phases share the R16-final recipe: plain RVQ R=16 / C=512 units,
# hierarchical AR (duration-first), delay=2, audioboost (weighted acoustic
# layer mix + per-codebook RVQ loss weights), Llama-3.2-1B backbone.
#
# Data: per-utterance RVQ units cached by streamSLM/extract/extract_tokens.py
# (see run_extract_full_seq.sh) under CACHE_ROOT. Set CACHE_ROOT before running.
# =============================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PHASE=${PHASE:-all}

# ---- data -------------------------------------------------------------------
# RVQ-unit cache produced by the extractor (LibriSpeech tc100/360/other-500 +
# Emilia full). Manifests are sharded <split>/manifest_shard*_of<WORLD>.csv.
export CACHE_ROOT=${CACHE_ROOT:?set CACHE_ROOT to the streamSLM_units_C512_R16 cache dir}
export MANIFEST_WORLD=${MANIFEST_WORLD:-48}
export EMI_WORLD=${EMI_WORLD:-48}
export EMI_GLOB=${EMI_GLOB-}                                   # no curated subset
export EMI_FULL_GLOB=${EMI_FULL_GLOB:-${CACHE_ROOT}/emilia/full/manifest_shard*_of48.csv}
export EMI_FULL_WORLD=${EMI_FULL_WORLD:-48}
export VAL_MANIFEST_WORLD=${VAL_MANIFEST_WORLD:-4}
export VAL_MANIFEST_GLOB=${VAL_MANIFEST_GLOB:-${CACHE_ROOT}/librispeech/dev-clean/manifest_shard*_of${VAL_MANIFEST_WORLD}.csv}

# ---- tokenizer / model (R16 final) -----------------------------------------
export TOKEN_TYPE=${TOKEN_TYPE:-rvq}
export RVQ_NQ=${RVQ_NQ:-16}
export RVQ_C=${RVQ_C:-512}
export MODEL_ARCH=${MODEL_ARCH:-streamslm_hier}
export HIER_AR_ORDER=${HIER_AR_ORDER:-duration_first}
export DURATION_LOSS_TYPE=${DURATION_LOSS_TYPE:-regression}
export DELAY=${DELAY:-2}
export TEXT_TOKENIZER=${TEXT_TOKENIZER:-llama}
export BACKBONE=${BACKBONE:-meta-llama/Llama-3.2-1B}
export REF_MODEL=${REF_MODEL:-meta-llama/Llama-3.2-1B}

# ---- audioboost -------------------------------------------------------------
export ACOUSTIC_LAYER_MIX=${ACOUSTIC_LAYER_MIX:-weighted}
export AUDIO_EMB_INIT_SCALE=${AUDIO_EMB_INIT_SCALE:-0.01}
# per-codebook RVQ loss weights: l1=x10, l2..l8=x3, l9..l16=x1 (sum 39)
export RVQ_LOSS_WEIGHTS=${RVQ_LOSS_WEIGHTS:-10,3,3,3,3,3,3,3,1,1,1,1,1,1,1,1}

# ---- shared schedule / optimization ----------------------------------------
export LR=${LR:-2e-4}
export LR_STEP_AT=${LR_STEP_AT:-10000}
export LR_STEP_TO=${LR_STEP_TO:-5e-5}
export WARMUP=${WARMUP:-200}
export BATCH_SIZE=${BATCH_SIZE:-8}
export GRAD_ACCUM=${GRAD_ACCUM:-4}
export GRAD_CLIP=${GRAD_CLIP:-1.0}
export LOSS_W_ACOUSTIC=${LOSS_W_ACOUSTIC:-1.0}
export LOSS_W_DURATION=${LOSS_W_DURATION:-0.1}
export VAL_EVERY=${VAL_EVERY:-500}
export LOG_EVERY=${LOG_EVERY:-50}

# ---- output / resources -----------------------------------------------------
export OUT_ROOT=${OUT_ROOT:-${REPO_ROOT}/checkpoints/streamSLM}
export LOG_DIR=${LOG_DIR:-${REPO_ROOT}/logs/streamSLM_train}
export PRETRAIN_NAME=${PRETRAIN_NAME:-slm_r16_pretrain}
export FINETUNE_NAME=${FINETUNE_NAME:-slm_r16_finetune}
export RESUME_STEP=${RESUME_STEP:-180000}                     # phase-1 ckpt to finetune from
export PRETRAIN_MAX_STEPS=${PRETRAIN_MAX_STEPS:-300000}
export FINETUNE_MAX_STEPS=${FINETUNE_MAX_STEPS:-210000}

export NGPU=${NGPU:-1}
# Job-submission prefix; empty = run directly. E.g. LAUNCHER="sr 1 48 --qos=q-low"
export LAUNCHER=${LAUNCHER:-}
export STREAMSLM_ATTN_IMPL=${STREAMSLM_ATTN_IMPL:-sdpa}
export STREAMSLM_LIGER=${STREAMSLM_LIGER:-0}
export MOSHI_ROOT=${MOSHI_ROOT:?set MOSHI_ROOT to a moshi 0.2.13 checkout (hier depformer)}

_COMMON="${REPO_ROOT}/streamSLM/scripts/_abl_common.sh"

run_pretrain() {
  export RUN_NAME="${PRETRAIN_NAME}"
  export OUT_DIR="${OUT_ROOT}/${RUN_NAME}"
  export LOSS_W_TEXT=1.0
  export TEXT_KL_WEIGHT=0.9
  export MAX_STEPS="${PRETRAIN_MAX_STEPS}"
  export SAVE_EVERY=10000
  export RESUME=""
  export RESUME_MODEL_ONLY=0
  echo "[phase1/pretrain] ${RUN_NAME} -> ${OUT_DIR} (max ${MAX_STEPS}, text_kl 0.9, loss_w_text 1.0)"
  bash "${_COMMON}"
}

run_finetune() {
  export RUN_NAME="${FINETUNE_NAME}"
  export OUT_DIR="${OUT_ROOT}/${RUN_NAME}"
  export LOSS_W_TEXT=5.0
  export TEXT_KL_WEIGHT=0.5
  export MAX_STEPS="${FINETUNE_MAX_STEPS}"
  export SAVE_EVERY=2500
  local step_tag; step_tag=$(printf "step_%08d.pt" "${RESUME_STEP}")
  export RESUME=${RESUME:-${OUT_ROOT}/${PRETRAIN_NAME}/${step_tag}}
  export RESUME_MODEL_ONLY=0
  echo "[phase2/finetune] ${RUN_NAME} -> ${OUT_DIR} (resume ${RESUME}, max ${MAX_STEPS}, text_kl 0.5, loss_w_text 5.0)"
  bash "${_COMMON}"
}

case "${PHASE}" in
  pretrain) run_pretrain ;;
  finetune) run_finetune ;;
  all)
    # chain: run phase 1 synchronously, then phase 2 resuming from its ckpt.
    export LAUNCH_FG=1
    run_pretrain
    run_finetune
    ;;
  *) echo "PHASE must be pretrain|finetune|all (got '${PHASE}')" >&2; exit 2 ;;
esac
