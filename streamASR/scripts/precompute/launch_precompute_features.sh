#!/usr/bin/env bash
# Launch 13 sharded CPU-only preprocessing workers via `sr 0`:
# Each job runs `precompute_speech_tokens.py --rank N --world_size 13`,
# saving cached .speech_tokens.pt / .spk_emb.pt under the writable cache root.
# CPU-only since the speech-tokenizer ONNX runs on CPU.
# Existing caches are skipped; pass --overwrite to redo.
#
# Usage:
#     bash scripts/launch_precompute_features.sh
#     EXTRA="--overwrite" bash scripts/launch_precompute_features.sh

set -euo pipefail

STREAMASR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${STREAMASR_ROOT}"

WORLD_SIZE="${WORLD_SIZE:-16}"
EXTRA="${EXTRA:-}"

mkdir -p logs/preprocess

EXCLUDE_NODES="${EXCLUDE_NODES:-greco}"

run_shard () {
    local rank="$1"; shift
    local logf="logs/preprocess/shard_${rank}.log"
    echo ">>> rank=${rank}/${WORLD_SIZE}  (sr 0, cpu)  log=${logf}"
    nohup sr 0 --exclude="${EXCLUDE_NODES}" \
        python scripts/precompute/precompute_speech_tokens.py \
            --rank "${rank}" --world_size "${WORLD_SIZE}" ${EXTRA} \
        >"${logf}" 2>&1 &
    disown || true
}

for r in $(seq 0 $((WORLD_SIZE - 1))); do
    run_shard "$r"
done

echo
echo "Submitted ${WORLD_SIZE} cpu shards. Tail any log:  tail -f ${STREAMASR_ROOT}/logs/preprocess/shard_*.log"
