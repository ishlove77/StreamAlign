#!/usr/bin/env bash
# ============================================================================
# launch_precompute_features_emilia.sh
# Launch N sharded CPU-only Emilia precompute workers, each as its own `sr 0`
# slurm job (so each worker gets its own CPU allocation — mirrors
# launch_precompute_features.sh for LibriTTS).
#
# Use this in preference to run_precompute_tokens_emilia.sh, which spawns 16
# subprocesses inside one sr 0 (4-CPU) allocation and is ~16x slower because
# of CPU oversubscription.
#
# Usage:
#     bash scripts/precompute/launch_precompute_features_emilia.sh
#     WORLD_SIZE=32 bash scripts/precompute/launch_precompute_features_emilia.sh
#     EXTRA="--overwrite" bash scripts/precompute/launch_precompute_features_emilia.sh
# ============================================================================

set -euo pipefail

STREAMASR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${STREAMASR_ROOT}"

WORLD_SIZE="${WORLD_SIZE:-16}"
EXTRA="${EXTRA:-}"
EMILIA_CSV="${EMILIA_CSV:-/home/datasets/Emilia/emilia_en_400h.csv}"
EMILIA_DATA_ROOT="${EMILIA_DATA_ROOT:-/home/datasets/Emilia}"
LOG_DIR="${LOG_DIR:-${STREAMASR_ROOT}/logs/precompute_emilia}"
EXCLUDE_NODES="${EXCLUDE_NODES:-greco}"
SR_QOS="${SR_QOS:-}"

mkdir -p "${LOG_DIR}"

QOS_ARG=()
if [[ -n "${SR_QOS}" ]]; then
    QOS_ARG=(--qos="${SR_QOS}")
fi

run_shard () {
    local rank="$1"
    local logf="${LOG_DIR}/shard_${rank}_of_${WORLD_SIZE}.log"
    echo ">>> rank=${rank}/${WORLD_SIZE}  (sr 0, cpu${SR_QOS:+, qos=${SR_QOS}})  log=${logf}"
    nohup sr 0 --exclude="${EXCLUDE_NODES}" "${QOS_ARG[@]}" \
        python -u scripts/precompute/precompute_speech_tokens_emilia.py \
            --rank "${rank}" --world_size "${WORLD_SIZE}" \
            --csv "${EMILIA_CSV}" --data_root "${EMILIA_DATA_ROOT}" ${EXTRA} \
        >"${logf}" 2>&1 &
    disown || true
}

for r in $(seq 0 $((WORLD_SIZE - 1))); do
    run_shard "$r"
done

echo
echo "Submitted ${WORLD_SIZE} cpu shards. Tail any log:"
echo "  tail -f ${LOG_DIR}/shard_*_of_${WORLD_SIZE}.log"
