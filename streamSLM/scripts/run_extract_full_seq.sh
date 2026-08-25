#!/usr/bin/env bash
# Full re-extraction with the fixed (word-boundary-preserving) path.
#
# Splits run sequentially in this order; within each split, WORLD GPU shards
# run in parallel (--world_size=WORLD). Each shard is one ${LAUNCHER} call
# (or a direct python process when LAUNCHER is empty).
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

# ---- environment (override via env) ----------------------------------------
CACHE_ROOT=${CACHE_ROOT:-${REPO_ROOT}/cache/streamSLM_units_C512_R16}
LOG_DIR=${LOG_DIR:-${REPO_ROOT}/logs/streamSLM_extract_C512_R16}

CHECKPOINTS_ROOT=${CHECKPOINTS_ROOT:-${REPO_ROOT}/weights}
CHECKPOINT=${CHECKPOINT:-${CHECKPOINTS_ROOT}/Streamalign-R16/rvq_teacher/epoch_22.pt}
EMILIA_CSV=${EMILIA_CSV:?set EMILIA_CSV to the Emilia 400h subset csv}
COSYVOICE_ROOT=${COSYVOICE_ROOT:-${REPO_ROOT}/streamASR/third_party/CosyVoice}
export COSYVOICE_ROOT
export PYTHONPATH="${COSYVOICE_ROOT}:${COSYVOICE_ROOT}/third_party/Matcha-TTS:${PYTHONPATH:-}"
# Job-submission prefix per shard; empty = run directly on this machine.
# Example (Slurm wrapper): LAUNCHER="sr 1 24 --qos=q-low --exclude=nodeA"
LAUNCHER=${LAUNCHER:-}
# ----------------------------------------------------------------------------

WORLD=${WORLD:-8}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-8}
TOKENIZER=${TOKENIZER:-llama}
BF16=${BF16:-1}

export WANDB_MODE=${WANDB_MODE:-disabled}
export WANDB_DISABLED=${WANDB_DISABLED:-true}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

mkdir -p "${CACHE_ROOT}" "${LOG_DIR}"

echo "[plan] CACHE_ROOT=${CACHE_ROOT}"
echo "[plan] LOG_DIR   =${LOG_DIR}"
echo "[plan] CHECKPOINT=${CHECKPOINT}"
echo "[plan] WORLD=${WORLD} batch=${BATCH_SIZE} workers=${NUM_WORKERS} bf16=${BF16} launcher=${LAUNCHER:-<direct>}"
echo "[plan] order: ls/tc100 -> ls/tc360 -> ls/tc500 -> emilia/400h"
echo "[plan] each split: ${WORLD} parallel shards"

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
    ${LAUNCHER} python -u -m streamSLM.extract.extract_tokens \
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
