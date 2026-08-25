#!/usr/bin/env bash
# Launch sharded CPU-only preprocessing workers for **LibriSpeech**.
# Each job runs `precompute_speech_tokens.py --rank N --world_size W` with the
# LibriSpeech root and a `_librispeech` cache subdir, so the LibriTTS cache
# under streamASR/cache/cosyvoice_features/{train-clean-100, ...}/ is NOT
# touched. CPU-only since the speech-tokenizer ONNX runs on CPU.
# Existing caches are skipped; pass EXTRA="--overwrite" to redo.
#
# Usage:
#     bash scripts/precompute/launch_precompute_features_librispeech.sh
#     WORLD_SIZE=24 bash scripts/precompute/launch_precompute_features_librispeech.sh
#     EXTRA="--overwrite" bash scripts/precompute/launch_precompute_features_librispeech.sh

set -euo pipefail

STREAMASR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${STREAMASR_ROOT}"

LIBRISPEECH_ROOT="${LIBRISPEECH_ROOT:?set LIBRISPEECH_ROOT to your LibriSpeech root}"
CACHE_SUBDIR="${CACHE_SUBDIR:-_librispeech}"
SPLITS="${SPLITS:-train-clean-100 train-clean-360 train-other-500 dev-clean dev-other}"

WORLD_SIZE="${WORLD_SIZE:-16}"
EXTRA="${EXTRA:-}"
# Per-shard job-submission prefix; empty = run locally (e.g. a Slurm wrapper).
LAUNCHER="${LAUNCHER:-}"

mkdir -p logs/preprocess_librispeech


run_shard () {
    local rank="$1"; shift
    local logf="logs/preprocess_librispeech/shard_${rank}.log"
    echo ">>> rank=${rank}/${WORLD_SIZE}  (cpu)  log=${logf}"
    nohup ${LAUNCHER} \
        python scripts/precompute/precompute_speech_tokens.py \
            --rank "${rank}" --world_size "${WORLD_SIZE}" \
            --data_root "${LIBRISPEECH_ROOT}" \
            --cache_subdir "${CACHE_SUBDIR}" \
            --datasets ${SPLITS} \
            --ext wav \
            ${EXTRA} \
        >"${logf}" 2>&1 &
    disown || true
}

for r in $(seq 0 $((WORLD_SIZE - 1))); do
    run_shard "$r"
done

echo
echo "Submitted ${WORLD_SIZE} cpu shards for LibriSpeech."
echo "  data_root   = ${LIBRISPEECH_ROOT}"
echo "  cache_subdir= ${CACHE_SUBDIR}  (output under cache/cosyvoice_features/${CACHE_SUBDIR}/...)"
echo "  splits      = ${SPLITS}"
echo "Tail any log:  tail -f ${STREAMASR_ROOT}/logs/preprocess_librispeech/shard_*.log"
