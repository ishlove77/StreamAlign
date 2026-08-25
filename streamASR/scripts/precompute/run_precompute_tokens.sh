#!/usr/bin/env bash
# ============================================================================
# run_precompute_tokens.sh
# Precompute speech tokens + spk_emb for all LibriTTS wav files.
# Launches one worker per GPU in parallel.
#
# Usage:
#   bash run_precompute_tokens.sh          # CPU-only ONNX workers
# ============================================================================

set -euo pipefail

STREAMASR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COSYVOICE_ROOT="${COSYVOICE_ROOT:-${STREAMASR_ROOT}/third_party/CosyVoice}"
export COSYVOICE_ROOT

export PYTHONPATH="${STREAMASR_ROOT}/third_party/Matcha-TTS:${COSYVOICE_ROOT}/third_party/Matcha-TTS:${COSYVOICE_ROOT}:${STREAMASR_ROOT}:${PYTHONPATH:-}"

# ---- environment (override via env) ----------------------------------------
LIBRITTS_ROOT=${LIBRITTS_ROOT:?set LIBRITTS_ROOT to your LibriTTS root}
export LIBRITTS_ROOT
LOG_DIR="${LOG_DIR:-${STREAMASR_ROOT}/logs/precompute}"
mkdir -p "${LOG_DIR}"
# ----------------------------------------------------------------------------

# Use more workers since ONNX runs on CPU — limited by CPU cores, not GPUs
N_WORKERS=${N_WORKERS:-16}
echo "Launching ${N_WORKERS} parallel workers (CPU-based ONNX)"

PYTHON="${PYTHON:-python}"

PIDS=()
for RANK in $(seq 0 $((N_WORKERS - 1))); do
    CUDA_VISIBLE_DEVICES="" "$PYTHON" -u \
        "${STREAMASR_ROOT}/scripts/precompute/precompute_speech_tokens.py" \
        --rank "${RANK}" \
        --world_size "${N_WORKERS}" \
        --data_root "${LIBRITTS_ROOT}" \
        >> "${LOG_DIR}/precompute_rank${RANK}.log" 2>&1 &
    PIDS+=($!)
    echo "  Rank ${RANK} → GPU ${RANK}, PID ${PIDS[-1]}, log: ${LOG_DIR}/precompute_rank${RANK}.log"
done

echo "Waiting for all workers to finish..."
FAILED=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "  Worker rank ${i} FAILED (PID ${PIDS[$i]})"
        FAILED=1
    else
        echo "  Worker rank ${i} done."
    fi
done

if [ $FAILED -eq 0 ]; then
    echo "All workers completed successfully."
else
    echo "One or more workers failed — check logs."
    exit 1
fi
