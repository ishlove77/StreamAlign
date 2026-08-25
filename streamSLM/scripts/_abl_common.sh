#!/usr/bin/env bash
# Shared bits for the 10k-step ablation launchers (sourced, not run).
# Sets the Llama-3.2-1B + RVQ C=512 R=16 + depth_transformer baseline. Each
# variant launcher overrides the relevant knob via env var before sourcing.

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"

# ---- shared baseline (matches streamalign_slm_r16)
RUN_TAG=${RUN_TAG:?RUN_TAG required (e.g. fuse_moshi_10k)}
# Default: prefix with "abl_" for ablation launchers. Main launchers can
# override RUN_NAME directly to skip the prefix.
RUN_NAME=${RUN_NAME:-abl_${RUN_TAG}}
OUT_DIR=${OUT_DIR:-${REPO_ROOT}/checkpoints/streamSLM/${RUN_NAME}}
LOG_DIR=${LOG_DIR:-${REPO_ROOT}/logs/streamSLM_train}
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

CACHE_ROOT=${CACHE_ROOT:-${REPO_ROOT}/cache/streamSLM_units_C512_R16}
# Default manifest sharding (default 16 for the legacy discrete-only cache;
# the C512_prequant cache is 44-way for train splits and 6-way for dev/test).
MANIFEST_WORLD=${MANIFEST_WORLD:-16}
# Per-split shard counts. Default to MANIFEST_WORLD; override if the cache
# uses heterogeneous shard counts across splits.
TC100_WORLD=${TC100_WORLD:-${MANIFEST_WORLD}}
TC360_WORLD=${TC360_WORLD:-${MANIFEST_WORLD}}
TC500_WORLD=${TC500_WORLD:-${MANIFEST_WORLD}}
EMI_WORLD=${EMI_WORLD:-${MANIFEST_WORLD}}
# LibriSpeech globs are env-overridable; set to "" to skip a split entirely
# (e.g. distill cache that only contains emilia).
TC100_GLOB=${TC100_GLOB-${CACHE_ROOT}/librispeech/train-clean-100/manifest_shard*_of${TC100_WORLD}.csv}
TC360_GLOB=${TC360_GLOB-${CACHE_ROOT}/librispeech/train-clean-360/manifest_shard*_of${TC360_WORLD}.csv}
TC500_GLOB=${TC500_GLOB-${CACHE_ROOT}/librispeech/train-other-500/manifest_shard*_of${TC500_WORLD}.csv}
# Curated Emilia 400h subset. Override with EMI_GLOB="" to drop it (e.g. when
# emilia/full is in use, since full subsumes 400h).
EMI_GLOB=${EMI_GLOB-${CACHE_ROOT}/emilia/400h/manifest_shard*_of${EMI_WORLD}.csv}

# Optional Emilia 40k-hr "full" set. Sharding world differs from the four core
# splits (48-way in the C512_prequant cache), so it gets its own knob. Empty
# glob = skip. Manifest count must match EMI_FULL_WORLD when present.
EMI_FULL_GLOB=${EMI_FULL_GLOB:-}
EMI_FULL_WORLD=${EMI_FULL_WORLD:-48}

# Optional fixed validation manifest set (e.g. dev-clean). Empty = legacy
# random val_fraction split of the training manifests.
VAL_MANIFEST_GLOB=${VAL_MANIFEST_GLOB:-}
VAL_MANIFEST_WORLD=${VAL_MANIFEST_WORLD:-6}

# TASTE stage-2 loss weights (mirror config.py). The phase-2 finetune bumps
# LOSS_W_TEXT to push the text head over the acoustic heads.
LOSS_W_TEXT=${LOSS_W_TEXT:-1.0}
LOSS_W_ACOUSTIC=${LOSS_W_ACOUSTIC:-1.0}
LOSS_W_DURATION=${LOSS_W_DURATION:-0.1}

BACKBONE=${BACKBONE:-meta-llama/Llama-3.2-1B}
# Frozen text-only LLM used for KL distillation on text logits (TASTE stage-2).
# Empty string disables KL — text loss collapses to pure CE. Single-dash form
# preserves an explicit empty override from launchers.
REF_MODEL=${REF_MODEL-meta-llama/Llama-3.2-1B}
TEXT_TOKENIZER=${TEXT_TOKENIZER:-llama}
TOKEN_TYPE=${TOKEN_TYPE:-rvq}
# RVQ knobs (used when TOKEN_TYPE=rvq).
# train.py reads rvq_* only when token_type=rvq.
RVQ_NQ=${RVQ_NQ:-16}
RVQ_C=${RVQ_C:-512}
DELAY=${DELAY:-1}

PREDICTOR_TYPE=${PREDICTOR_TYPE:-depth_transformer}
DEPFORMER_DIM=${DEPFORMER_DIM:-1024}
DEPFORMER_HEADS=${DEPFORMER_HEADS:-16}
DEPFORMER_LAYERS=${DEPFORMER_LAYERS:-6}
DEPFORMER_FF=${DEPFORMER_FF:-4096}
# Dropout on the previous-codebook embedding fed into the next depth-layer
# prediction (AR coupling branch only — does NOT touch dep_in(h) or BOS).
# 0.0 = off, preserving the original audioboost recipe.
PREV_DEPTH_OUTPUT_DROPOUT=${PREV_DEPTH_OUTPUT_DROPOUT:-0.0}
# MOSS-Local depth predictor knobs (only consumed when
# PREDICTOR_TYPE=moss_local_depth; defaults match MossTTSLocalConfig).
MOSS_LOCAL_HIDDEN=${MOSS_LOCAL_HIDDEN:-1536}
MOSS_LOCAL_HEADS=${MOSS_LOCAL_HEADS:-16}
MOSS_LOCAL_LAYERS=${MOSS_LOCAL_LAYERS:-4}
MOSS_LOCAL_FFN=${MOSS_LOCAL_FFN:-8960}
MOSS_LOCAL_BRIDGE_FFN=${MOSS_LOCAL_BRIDGE_FFN:-2048}

BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-4}
GRAD_CLIP=${GRAD_CLIP:-1.0}
NUM_WORKERS=${NUM_WORKERS:-4}
# Sequence packing: 1 = enable, forces train loader to B=1 packed sequences of
# up to PACK_MAX_TOKENS subwords. Validation always runs unpacked.
PACKING=${PACKING:-0}
PACK_MAX_TOKENS=${PACK_MAX_TOKENS:-4096}
# Text-loss KL blend (TASTE stage-2). Default 0.9. Setting to 0 drops KL and
# skips the ref_model load (~5 GB + half the per-step FLOPs recovered).
TEXT_KL_WEIGHT=${TEXT_KL_WEIGHT:-0.9}
LR=${LR:-2e-4}
# Optional step-decay: switch lr to LR_STEP_TO once step >= LR_STEP_AT (-1 disables).
LR_STEP_AT=${LR_STEP_AT:--1}
LR_STEP_TO=${LR_STEP_TO:-0.0}
WARMUP=${WARMUP:-200}        # short for 10k-step ablation
MAX_STEPS=${MAX_STEPS:-10000}
SAVE_EVERY=${SAVE_EVERY:-2500}
LOG_EVERY=${LOG_EVERY:-50}
VAL_EVERY=${VAL_EVERY:-500}

# Ablation knobs (default = legacy behavior).
FUSION_MODE=${FUSION_MODE:-gated_softmax}
AUDIO_EMB_INIT_SCALE=${AUDIO_EMB_INIT_SCALE:-auto}
RVQ_LOSS_WEIGHTS=${RVQ_LOSS_WEIGHTS:-}
# Optional resume from a step_*.pt or latest.pt checkpoint. Loads model+optim
# state and seeds start_step from ckpt['step']. Architecture knobs (delay,
# fusion_mode, predictor type, etc.) come from the *current* CLI args, so the
# resumed checkpoint must be shape-compatible with whatever the launcher sets.
RESUME=${RESUME:-}
# When set to 1, resume loads only model weights (strict=False) and starts a
# fresh optimizer with start_step=0 — fork-from-weights semantics. Use for
# warm-starting a different-arch run from a baseline ckpt (e.g. R=16 fork
# from a deeper baseline; layers 0..R-1 load, layers R..R_src-1 reset).
RESUME_MODEL_ONLY=${RESUME_MODEL_ONLY:-0}
# Duration ↔ acoustic coupling knobs (Exp A / Exp B).
DURATION_HEAD_INPUT=${DURATION_HEAD_INPUT:-llm}
DURATION_LOSS_TYPE=${DURATION_LOSS_TYPE:-regression}
DURATION_NUM_BUCKETS=${DURATION_NUM_BUCKETS:-256}
FEED_DURATION_TO_DEPFORMER=${FEED_DURATION_TO_DEPFORMER:-0}
# Continuous-acoustic head (replaces per-codebook CE with regression to a
# teacher pre-quantizer feature). Default 'rvq' = discrete codes.
ACOUSTIC_TARGET=${ACOUSTIC_TARGET:-rvq}
ACOUSTIC_FEAT_DIM=${ACOUSTIC_FEAT_DIM:-256}
ACOUSTIC_FEAT_LOSS=${ACOUSTIC_FEAT_LOSS:-l1}
# Hidden-state source for acoustic + duration heads.
#   "last" (default, legacy): final transformer layer only.
#   "weighted": TASTE-style softmax mix over all hidden_states.
ACOUSTIC_LAYER_MIX=${ACOUSTIC_LAYER_MIX:-last}
# Model architecture variant.
#   "streamslm" (default, legacy): parallel text / acoustic / duration heads off h_n.
#   "streamslm_hier": Moshi-style step-internal depth-AR; lm_head(h_n) -> w_{n+1}
#     stays on the backbone, (q_n, d_n) come from a depth transformer that
#     conditions on w_{n+1}. Ordering controlled by HIER_AR_ORDER.
MODEL_ARCH=${MODEL_ARCH:-streamslm}
HIER_AR_ORDER=${HIER_AR_ORDER:-duration_last}

NGPU=${NGPU:-1}               # >1 = DDP via torchrun on a single multi-GPU alloc
# Job-submission prefix; empty = run directly on this machine. Any scheduler
# flags (GPU count, VRAM, qos, node excludes, ...) belong in the prefix, e.g.:
#   LAUNCHER="sr 1 48 --qos=q-low --exclude=nodeA"
LAUNCHER=${LAUNCHER:-}
export LAUNCHER
STREAMSLM_ATTN_IMPL=${STREAMSLM_ATTN_IMPL:-flash_attention_2}
export STREAMSLM_ATTN_IMPL
# Toggle HF gradient checkpointing on the backbone. Default off; flip to 1 for
# big-backbone runs (e.g. Llama-3.2-3B) to drop activation memory at the cost
# of an extra forward per layer.
STREAMSLM_GRADIENT_CHECKPOINTING=${STREAMSLM_GRADIENT_CHECKPOINTING:-0}
export STREAMSLM_GRADIENT_CHECKPOINTING
# Toggle bitsandbytes AdamW8bit (saves ~21 GB of fp32 optimizer state at 3.5B
# trainable). Default off; flip to 1 for big-backbone runs that overflow plain
# AdamW state on a single 48GB GPU.
STREAMSLM_OPT_8BIT=${STREAMSLM_OPT_8BIT:-0}
export STREAMSLM_OPT_8BIT

# launch <cmd...>: run through ${LAUNCHER} when set, else directly.
launch() {
  if [[ -n "${LAUNCHER}" ]]; then ${LAUNCHER} "$@"; else "$@"; fi
}

export WANDB_MODE=${WANDB_MODE:-disabled}
export WANDB_DISABLED=${WANDB_DISABLED:-true}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

shopt -s nullglob
TC100_FILES=( ${TC100_GLOB} )
TC360_FILES=( ${TC360_GLOB} )
TC500_FILES=( ${TC500_GLOB} )
EMI_FILES=( ${EMI_GLOB} )
EMI_FULL_FILES=()
if [[ -n "${EMI_FULL_GLOB}" ]]; then
  EMI_FULL_FILES=( ${EMI_FULL_GLOB} )
fi
VAL_FILES=()
if [[ -n "${VAL_MANIFEST_GLOB}" ]]; then
  VAL_FILES=( ${VAL_MANIFEST_GLOB} )
fi
shopt -u nullglob

if [[ -n "${TC100_GLOB}" ]] && (( ${#TC100_FILES[@]} != TC100_WORLD )); then
  echo "[fatal] tc100 manifests incomplete (expect ${TC100_WORLD}, got ${#TC100_FILES[@]}). Glob=${TC100_GLOB}" >&2
  exit 2
fi
if [[ -n "${TC360_GLOB}" ]] && (( ${#TC360_FILES[@]} != TC360_WORLD )); then
  echo "[fatal] tc360 manifests incomplete (expect ${TC360_WORLD}, got ${#TC360_FILES[@]}). Glob=${TC360_GLOB}" >&2
  exit 2
fi
if [[ -n "${TC500_GLOB}" ]] && (( ${#TC500_FILES[@]} != TC500_WORLD )); then
  echo "[fatal] tc500 manifests incomplete (expect ${TC500_WORLD}, got ${#TC500_FILES[@]}). Glob=${TC500_GLOB}" >&2
  exit 2
fi
if [[ -n "${EMI_GLOB}" ]] && (( ${#EMI_FILES[@]} != EMI_WORLD )); then
  echo "[fatal] emilia/400h manifests incomplete (expect ${EMI_WORLD}, got ${#EMI_FILES[@]}). Glob=${EMI_GLOB}" >&2
  exit 2
fi
if [[ -n "${EMI_FULL_GLOB}" ]] && (( ${#EMI_FULL_FILES[@]} != EMI_FULL_WORLD )); then
  echo "[fatal] emilia/full manifests incomplete (expect ${EMI_FULL_WORLD}, got ${#EMI_FULL_FILES[@]}). Glob=${EMI_FULL_GLOB}" >&2
  exit 2
fi
if [[ -n "${VAL_MANIFEST_GLOB}" ]] && (( ${#VAL_FILES[@]} != VAL_MANIFEST_WORLD )); then
  echo "[fatal] val manifests incomplete (expect ${VAL_MANIFEST_WORLD}, got ${#VAL_FILES[@]}). Glob=${VAL_MANIFEST_GLOB}" >&2
  exit 2
fi

echo "[plan] ${RUN_NAME}"
echo "[plan]   cache_root=${CACHE_ROOT} manifest_world=${MANIFEST_WORLD}"
echo "[plan]   emi_400h=${EMI_GLOB:-<off>} (n=${#EMI_FILES[@]})"
echo "[plan]   emi_full=${EMI_FULL_GLOB:-<off>} (n=${#EMI_FULL_FILES[@]}, world=${EMI_FULL_WORLD})"
echo "[plan]   val_manifest=${VAL_MANIFEST_GLOB:-<random val_fraction split>} (n=${#VAL_FILES[@]})"
echo "[plan]   acoustic_target=${ACOUSTIC_TARGET} feat_dim=${ACOUSTIC_FEAT_DIM} feat_loss=${ACOUSTIC_FEAT_LOSS}"
echo "[plan]   teacher_rvq_ckpt=${TEACHER_RVQ_CKPT:-<off>} acc_layers=${TEACHER_RVQ_ACC_LAYERS:-<default>}"
echo "[plan]   ref_model=${REF_MODEL:-<off: text-CE only>}"
echo "[plan]   fusion_mode=${FUSION_MODE} audio_emb_init_scale=${AUDIO_EMB_INIT_SCALE} acoustic_layer_mix=${ACOUSTIC_LAYER_MIX}"
echo "[plan]   rvq_loss_weights=${RVQ_LOSS_WEIGHTS:-<uniform>}"
echo "[plan]   duration_head_input=${DURATION_HEAD_INPUT} duration_loss_type=${DURATION_LOSS_TYPE} buckets=${DURATION_NUM_BUCKETS} feed_dur=${FEED_DURATION_TO_DEPFORMER}"
echo "[plan]   model_arch=${MODEL_ARCH} hier_ar_order=${HIER_AR_ORDER}"
echo "[plan]   depformer dim=${DEPFORMER_DIM} heads=${DEPFORMER_HEADS} layers=${DEPFORMER_LAYERS} ff=${DEPFORMER_FF} prev_depth_dropout=${PREV_DEPTH_OUTPUT_DROPOUT}"
echo "[plan]   moss_local hidden=${MOSS_LOCAL_HIDDEN} heads=${MOSS_LOCAL_HEADS} layers=${MOSS_LOCAL_LAYERS} ffn=${MOSS_LOCAL_FFN} bridge_ffn=${MOSS_LOCAL_BRIDGE_FFN}"
echo "[plan]   packing=${PACKING} pack_max_tokens=${PACK_MAX_TOKENS} text_kl_weight=${TEXT_KL_WEIGHT}"
echo "[plan]   lr=${LR} warmup=${WARMUP} max_steps=${MAX_STEPS} save=${SAVE_EVERY} val=${VAL_EVERY} grad_clip=${GRAD_CLIP} lr_step_at=${LR_STEP_AT} lr_step_to=${LR_STEP_TO}"
echo "[plan]   delay=${DELAY} resume=${RESUME:-<fresh>} resume_model_only=${RESUME_MODEL_ONLY}"
echo "[plan]   ngpu=${NGPU} launcher=${LAUNCHER:-<direct>} attn_impl=${STREAMSLM_ATTN_IMPL} grad_ckpt=${STREAMSLM_GRADIENT_CHECKPOINTING} opt_8bit=${STREAMSLM_OPT_8BIT}"
echo "[plan]   OUT_DIR=${OUT_DIR}"
echo "[plan]   LOG_FILE=${LOG_FILE}"

# Build the command. NGPU>1 -> DDP via torchrun on a single multi-GPU alloc.
if (( NGPU > 1 )); then
  CMD=(torchrun --standalone --nproc_per_node="${NGPU}" -m streamSLM.train.train)
else
  CMD=("${PYTHON:-python}" -u -m streamSLM.train.train)
fi
CMD+=(
  --manifest "${TC100_FILES[@]}" "${TC360_FILES[@]}" "${TC500_FILES[@]}" "${EMI_FILES[@]}" "${EMI_FULL_FILES[@]}"
  --out_dir "${OUT_DIR}"
  --backbone "${BACKBONE}"
  --ref_model "${REF_MODEL}"
  --text_tokenizer "${TEXT_TOKENIZER}"
  --token_type "${TOKEN_TYPE}"
  --rvq_num_quantizers "${RVQ_NQ}"
  --rvq_codebook_size "${RVQ_C}"
  --delay "${DELAY}"
  --acoustic_predictor_type "${PREDICTOR_TYPE}"
  --depformer_dim "${DEPFORMER_DIM}"
  --depformer_num_heads "${DEPFORMER_HEADS}"
  --depformer_num_layers "${DEPFORMER_LAYERS}"
  --depformer_dim_feedforward "${DEPFORMER_FF}"
  --prev_depth_output_dropout "${PREV_DEPTH_OUTPUT_DROPOUT}"
  --moss_local_hidden "${MOSS_LOCAL_HIDDEN}"
  --moss_local_num_heads "${MOSS_LOCAL_HEADS}"
  --moss_local_num_layers "${MOSS_LOCAL_LAYERS}"
  --moss_local_ffn "${MOSS_LOCAL_FFN}"
  --moss_local_bridge_ffn "${MOSS_LOCAL_BRIDGE_FFN}"
  --batch_size "${BATCH_SIZE}"
  --grad_accum "${GRAD_ACCUM}"
  --grad_clip "${GRAD_CLIP}"
  --num_workers "${NUM_WORKERS}"
  --lr "${LR}"
  --lr_step_at "${LR_STEP_AT}"
  --lr_step_to "${LR_STEP_TO}"
  --warmup_steps "${WARMUP}"
  --max_steps "${MAX_STEPS}"
  --save_every "${SAVE_EVERY}"
  --log_every "${LOG_EVERY}"
  --val_every "${VAL_EVERY}"
  --bf16
  --fusion_mode "${FUSION_MODE}"
  --audio_emb_init_scale "${AUDIO_EMB_INIT_SCALE}"
  --duration_head_input "${DURATION_HEAD_INPUT}"
  --duration_loss_type "${DURATION_LOSS_TYPE}"
  --duration_num_buckets "${DURATION_NUM_BUCKETS}"
  --acoustic_target "${ACOUSTIC_TARGET}"
  --acoustic_feat_dim "${ACOUSTIC_FEAT_DIM}"
  --acoustic_feat_loss "${ACOUSTIC_FEAT_LOSS}"
  --acoustic_layer_mix "${ACOUSTIC_LAYER_MIX}"
  --loss_w_text "${LOSS_W_TEXT}"
  --loss_w_acoustic "${LOSS_W_ACOUSTIC}"
  --loss_w_duration "${LOSS_W_DURATION}"
  --text_kl_weight "${TEXT_KL_WEIGHT}"
  --model_arch "${MODEL_ARCH}"
  --hier_ar_order "${HIER_AR_ORDER}"
)
if [[ "${PACKING}" == "1" ]]; then
  CMD+=(--packing --pack_max_tokens "${PACK_MAX_TOKENS}")
fi
if [[ -n "${RVQ_LOSS_WEIGHTS}" ]]; then
  CMD+=(--rvq_loss_weights "${RVQ_LOSS_WEIGHTS}")
fi
if [[ "${FEED_DURATION_TO_DEPFORMER}" == "1" ]]; then
  CMD+=(--feed_duration_to_depformer)
fi
if (( ${#VAL_FILES[@]} > 0 )); then
  CMD+=(--val_manifest "${VAL_FILES[@]}")
fi
if [[ -n "${RESUME}" ]]; then
  CMD+=(--resume "${RESUME}")
fi
if [[ -n "${TEACHER_RVQ_CKPT:-}" ]]; then
  CMD+=(--teacher_rvq_ckpt "${TEACHER_RVQ_CKPT}")
fi
if [[ -n "${TEACHER_RVQ_ACC_LAYERS:-}" ]]; then
  CMD+=(--teacher_rvq_acc_layers "${TEACHER_RVQ_ACC_LAYERS}")
fi
if [[ "${RESUME_MODEL_ONLY}" == "1" ]]; then
  CMD+=(--resume_model_only)
fi

LAUNCH_FG=${LAUNCH_FG:-0}    # 1 = run synchronously (used by watchdog chain)
if [[ "${LAUNCH_FG}" == "1" ]]; then
  launch "${CMD[@]}" > "${LOG_FILE}" 2>&1
  rc=$?
  echo "[done] rc=${rc}  ${RUN_NAME}  log=${LOG_FILE}"
  exit "${rc}"
fi

nohup bash -c 'if [[ -n "${LAUNCHER}" ]]; then exec ${LAUNCHER} "$@"; else exec "$@"; fi' _ "${CMD[@]}" > "${LOG_FILE}" 2>&1 &
PID=$!
echo "[submitted] ${RUN_NAME} pid=${PID}  log=${LOG_FILE}"
disown "${PID}" 2>/dev/null || true
