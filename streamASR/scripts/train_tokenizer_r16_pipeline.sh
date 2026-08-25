#!/usr/bin/env bash
# =============================================================================
# train_tokenizer_r16_pipeline.sh
#
# Full StreamAlign R=16 tokenizer training pipeline, in three stages:
#
#   Stage 1  (continuous)  : learn the continuous acoustic embedding z_m with
#                            the quantizer bypassed (RVQ_BYPASS=1), so the
#                            encoder / char-aggregator are trained before any
#                            hard quantization is introduced.
#   Stage 2  (R=16 RVQ)    : resume from Stage 1 and train ResidualVQ with
#                            R=16 codebook layers (plain RVQ, EMA-updated
#                            codebook), matching the R=32 pipeline.
#   Stage 3  (cosine)      : resume from Stage 2 and fine-tune with a cosine
#                            learning-rate decay to 0.
#
# Prerequisites (see scripts/README.md for the full pipeline order):
#   1. ASR trained            -> run_train_asr_word_fastemit.sh
#   2. TextGrids generated     -> generate_textgrids.sh
#   3. Boundary dataset + clf  -> create_boundary_dataset.sh, train_boundary_classifier.sh
#   4. Stage-1 word distill    -> train_word_distill.sh
#   5. CosyVoice features      -> scripts/precompute/
#
# Usage:
#   bash scripts/train_tokenizer_r16_pipeline.sh <stage>
#     <stage> = continuous | r16 | cosine
#
# Override any of the env vars below on the command line, e.g.
#   RESUME_PATH=.../epoch_13.pt bash scripts/train_tokenizer_r16_pipeline.sh r16
# =============================================================================
set -euo pipefail

STAGE="${1:?usage: train_tokenizer_r16_pipeline.sh <continuous|r16|cosine>}"

# Repo-relative paths (this script lives in streamASR/scripts/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STREAMASR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Vendored SpeechBrain lives under streamASR/speechbrain; keep it on the path.
export PYTHONPATH="${STREAMASR_ROOT}:${PYTHONPATH:-}"

# ---- Tokenizer / quantizer config -------------------------------------------
: "${RVQ_R:=16}"                 # R=16 codebook layers
: "${RVQ_CODEBOOK_SIZE:=512}"
export RVQ_R RVQ_CODEBOOK_SIZE
unset RVQ_CODEBOOK_DIM           # codebook_dim = feat_dim (256)

# ---- Data / checkpoint paths (override via env) -----------------------------
: "${PYTHON:=python}"
# TRUTH_CKPT: point (or symlink results/char_asr_ckpt) at your best char-ASR
# SpeechBrain checkpoint dir (train/results/conformer_transducer_char/char_asr/save/CKPT+...).
: "${TRUTH_CKPT:=${STREAMASR_ROOT}/results/char_asr_ckpt}"
: "${HPARAMS:=${STREAMASR_ROOT}/hparams/alignment.yaml}"
: "${LIBRISPEECH_ROOT:?set LIBRISPEECH_ROOT to your LibriSpeech root}"
: "${LIBRITTS_ROOT:?set LIBRITTS_ROOT to your LibriTTS root}"
: "${EMILIA_ROOT:?set EMILIA_ROOT to your Emilia dataset root}"
: "${EMILIA_CSV:=${EMILIA_ROOT}/emilia_en_400h.csv}"
# Training reads LibriTTS; TextGrids default to the generate_textgrids.sh
# output alongside the LibriTTS root.
: "${TEXTGRID_ROOT:=${LIBRITTS_ROOT}/chunk_textgrids_word_model_final2}"
: "${COSYVOICE_ROOT:=${STREAMASR_ROOT}/third_party/CosyVoice}"
: "${NUM_EPOCHS:=400}"
: "${MASTER_PORT:=29662}"
# Exported: the trainers / utils read these names from the environment.
export LIBRISPEECH_ROOT LIBRITTS_ROOT EMILIA_ROOT TEXTGRID_ROOT COSYVOICE_ROOT

NUM_GPUS=$("${PYTHON}" -c "import torch; print(max(1, torch.cuda.device_count()))")
cd "${STREAMASR_ROOT}"

common_args=(
    --num_epochs="${NUM_EPOCHS}"
    --batch_size=16
    --weight_decay=1e-5
    --mask_prob=0.08
    --train_sample_ratio=0.5
    --chunk_size=4
    --left_context=32
    --hparams="${HPARAMS}"
    --checkpoint_path="${TRUTH_CKPT}"
    --emilia_sample_ratio=0.1
    --emilia_csv="${EMILIA_CSV}"
    --emilia_data_root="${EMILIA_ROOT}"
    --data_root="${LIBRITTS_ROOT}"
    --use_precomputed_features
)

case "${STAGE}" in
  continuous)
    # Stage 1: continuous representation. RVQ_BYPASS=1 skips quantization
    # entirely (see models/model_tokenizer.py::_rvq_quantize), so the
    # encoder and char-aggregator are trained on continuous seg_z. The
    # quantizer is still constructed, so this checkpoint initializes Stage 2.
    export RVQ_BYPASS=1
    : "${LEARNING_RATE:=1e-4}"
    : "${EXP_NAME:=streamalign_r16_stage1_continuous}"
    TRAINER="${STREAMASR_ROOT}/train/train_tokenizer.py"
    RESUME_ARG=()
    ;;
  r16)
    # Stage 2: RVQ R=16, plain quantizer (EMA-updated codebook).
    export RVQ_BYPASS=0    # explicit: quantize (never inherit Stage-1 bypass)
    : "${RESUME_PATH:?set RESUME_PATH to the Stage-1 (continuous) checkpoint}"
    : "${LEARNING_RATE:=1e-4}"
    : "${EXP_NAME:=streamalign_r16_stage2_rvq}"
    TRAINER="${STREAMASR_ROOT}/train/train_tokenizer.py"
    RESUME_ARG=(--resume_path="${RESUME_PATH}")
    ;;
  cosine)
    # Stage 3: cosine LR decay fine-tune (peak -> 0).
    export RVQ_BYPASS=0    # explicit: quantize (never inherit Stage-1 bypass)
    : "${RESUME_PATH:?set RESUME_PATH to the Stage-2 (R=16) checkpoint}"
    : "${COSINE_TOTAL_STEPS:=30000}"       # decay horizon in steps (~5 epochs)
    : "${COSINE_MIN_LR:=0}"
    : "${LEARNING_RATE:=1e-5}"   # lower peak LR for the decay stage
    export COSINE_TOTAL_STEPS COSINE_MIN_LR
    : "${EXP_NAME:=streamalign_r16_stage3_cosine}"
    TRAINER="${STREAMASR_ROOT}/train/train_tokenizer_cosine.py"
    RESUME_ARG=(--resume_path="${RESUME_PATH}")
    ;;
  *)
    echo "unknown stage: ${STAGE} (expected continuous|r16|cosine)" >&2; exit 1 ;;
esac

echo "[r16-pipeline:${STAGE}] R=${RVQ_R} C=${RVQ_CODEBOOK_SIZE} lr=${LEARNING_RATE} exp=${EXP_NAME}"

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    "${TRAINER}" \
    --exp_name="${EXP_NAME}" \
    --learning_rate="${LEARNING_RATE}" \
    "${common_args[@]}" \
    "${RESUME_ARG[@]}"
