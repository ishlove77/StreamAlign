#!/usr/bin/env bash
# Full re-extraction with the fixed (word-boundary-preserving) path.
#
# Splits run sequentially in this order; within each split, 8 GPU shards
# run in parallel (--world_size=8, ranks 0..7). Each shard is one
# `sr 1 24` call (mid QoS, default).
#
#   1) LibriSpeech train-clean-100
#   2) LibriSpeech train-clean-360
#   3) LibriSpeech train-other-500
#   4) Emilia 400h (driven by emilia_en_400h.csv)
#
# Cache layout:
#   ${CACHE_ROOT}/librispeech/<split>/manifest_shard{R}_of8.csv
#   ${CACHE_ROOT}/librispeech/<split>/<rel>.units.pt
#   ${CACHE_ROOT}/emilia/400h/manifest_shard{R}_of8.csv
#   ${CACHE_ROOT}/emilia/400h/<rel>.units.pt
#
# Logs:
#   ${LOG_DIR}/<dataset>/<split>/shard{R}_of8.log
#   ${LOG_DIR}/orchestrator.log    <- this script's own stdout

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

CACHE_ROOT=${CACHE_ROOT:-${REPO_ROOT}/cache/streamSLM_units_C2048_fixed}
LOG_DIR=${LOG_DIR:-${REPO_ROOT}/logs/streamSLM_extract_C2048_fixed}

CHECKPOINT=${CHECKPOINT:-/home/streamalign/streamASR/checkpoints/streamalign_r16/epoch_22.pt}
EMILIA_CSV=${EMILIA_CSV:-/home/datasets/Emilia/emilia_en_400h.csv}

WORLD=${WORLD:-8}
VRAM=${VRAM:-24}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-8}
TOKENIZER=${TOKENIZER:-llama}
BF16=${BF16:-1}
# QoS: long-running batch tokenization fits "low" quota best (mid quota's
# QOSMaxBillingPerUser cap blocks >~1 concurrent 24GB shard for this user).
SR_QOS=${SR_QOS:-q-low}
# Some titanrtx nodes (kandinsky, greco) report torch.cuda.is_available()=False
# even with --gres=gpu:1. Exclude them so shards don't crash mid-run.
SR_EXCLUDE=${SR_EXCLUDE:-kandinsky,greco,namjune}

export WANDB_MODE=${WANDB_MODE:-disabled}
export WANDB_DISABLED=${WANDB_DISABLED:-true}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

mkdir -p "${CACHE_ROOT}" "${LOG_DIR}"

echo "[plan] CACHE_ROOT=${CACHE_ROOT}"
echo "[plan] LOG_DIR   =${LOG_DIR}"
echo "[plan] CHECKPOINT=${CHECKPOINT}"
echo "[plan] WORLD=${WORLD} VRAM=${VRAM} batch=${BATCH_SIZE} workers=${NUM_WORKERS} bf16=${BF16} qos=${SR_QOS} exclude=${SR_EXCLUDE:-<none>}"
echo "[plan] order: ls/tc100 -> ls/tc360 -> ls/tc500 -> emilia/400h"
echo "[plan] each split: ${WORLD} parallel shards (sr 1 ${VRAM} --qos=${SR_QOS})"

# Run one split's 8 shards in parallel and block until they all finish.
#   $1 dataset    "librispeech" | "emilia"
#   $2 split arg  e.g. train-clean-100 | 400h
#   $3 extra args, comma-separated (e.g. "--emilia_csv,/path"); empty if none
run_split() {
  local dataset="$1"; local split="$2"; local extra="$3"
  local log_subdir="${LOG_DIR}/${dataset}/${split}"
  mkdir -p "${log_subdir}"
  IFS=',' read -r -a EXTRA <<< "${extra}"

  echo "[start] ${dataset}/${split}  $(date -Iseconds)"

  local pids=()
  for ((rank=0; rank<WORLD; rank++)); do
    local logf="${log_subdir}/shard${rank}_of${WORLD}.log"
    echo "  [submit] ${dataset}/${split} shard ${rank}/${WORLD}  ->  ${logf}"
    local bf16_flag=""
    if [[ "${BF16}" == "1" ]]; then bf16_flag="--bf16"; fi
    local qos_flag=""
    if [[ -n "${SR_QOS}" && "${SR_QOS}" != "mid" ]]; then qos_flag="--qos=${SR_QOS}"; fi
    local exclude_flag=""
    if [[ -n "${SR_EXCLUDE}" ]]; then exclude_flag="--exclude=${SR_EXCLUDE}"; fi
    sr 1 "${VRAM}" ${qos_flag} ${exclude_flag} python -u -m streamSLM.extract.extract_tokens \
      --dataset "${dataset}" \
      --split "${split}" \
      ${EXTRA[@]:+"${EXTRA[@]}"} \
      --variant rvq \
      --checkpoint "${CHECKPOINT}" \
      --cache_root "${CACHE_ROOT}" \
      --rank "${rank}" \
      --world_size "${WORLD}" \
      --batch_size "${BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}" \
      --tokenizer "${TOKENIZER}" \
      ${bf16_flag} \
      > "${logf}" 2>&1 &
    pids+=($!)
  done

  local fail=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      fail=$(( fail + 1 ))
    fi
  done

  if (( fail > 0 )); then
    echo "[fail ] ${dataset}/${split}: ${fail}/${WORLD} shards FAILED."
    return 1
  fi
  echo "[done ] ${dataset}/${split}  $(date -Iseconds)"
}

t0=$(date +%s)

run_split librispeech train-clean-100 ""           || exit 1
run_split librispeech train-clean-360 ""           || exit 1
run_split librispeech train-other-500 ""           || exit 1
run_split emilia      400h            "--emilia_csv,${EMILIA_CSV}" || exit 1

t1=$(date +%s)
elapsed=$(( t1 - t0 ))
printf "[ALL DONE] elapsed=%dh%02dm  $(date -Iseconds)\n" $((elapsed/3600)) $(((elapsed%3600)/60))
