#!/usr/bin/env bash
# ============================================================================
# run_precompute_tokens_emilia.sh
# Precompute CosyVoice speech tokens + spk_emb for the Emilia subset
# referenced by emilia_en_400h.csv (~400h, 1/100 of full Emilia).
# Launches multiple CPU-only ONNX workers in parallel (no GPU needed).
#
# Usage:
#   sr 0 bash run_precompute_tokens_emilia.sh                      # CPU job
#   sr 0 bash run_precompute_tokens_emilia.sh --qos=q-low          # long job
# ============================================================================

set -euo pipefail

STREAMASR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COSYVOICE_ROOT="/home/CosyVoice"

EMILIA_CSV="${EMILIA_CSV:-/home/datasets/Emilia/emilia_en_400h.csv}"
EMILIA_DATA_ROOT="${EMILIA_DATA_ROOT:-/home/datasets/Emilia}"
LOG_DIR="${LOG_DIR:-${STREAMASR_ROOT}/logs/precompute_emilia}"
mkdir -p "${LOG_DIR}"

export PYTHONPATH="${STREAMASR_ROOT}/third_party/Matcha-TTS:${COSYVOICE_ROOT}/third_party/Matcha-TTS:${COSYVOICE_ROOT}:${STREAMASR_ROOT}:${PYTHONPATH:-}"

# CPU-bound (ONNX speaker + tokenizer); scale by available cores, not GPUs.
N_WORKERS="${N_WORKERS:-16}"
echo "Launching ${N_WORKERS} parallel workers (CPU-based ONNX)"
echo "  CSV:       ${EMILIA_CSV}"
echo "  data_root: ${EMILIA_DATA_ROOT}"
echo "  log_dir:   ${LOG_DIR}"

PYTHON="/home/miniconda3/envs/streamASR/bin/python"
_CONDA_ENV="/home/miniconda3/envs/streamASR"
export LD_LIBRARY_PATH="${_CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"

PIDS=()
for RANK in $(seq 0 $((N_WORKERS - 1))); do
    CUDA_VISIBLE_DEVICES="" "$PYTHON" -u \
        "${STREAMASR_ROOT}/scripts/precompute/precompute_speech_tokens_emilia.py" \
        --rank "${RANK}" \
        --world_size "${N_WORKERS}" \
        --csv "${EMILIA_CSV}" \
        --data_root "${EMILIA_DATA_ROOT}" \
        >> "${LOG_DIR}/precompute_emilia_rank${RANK}.log" 2>&1 &
    PIDS+=($!)
    echo "  Rank ${RANK}, PID ${PIDS[-1]}, log: ${LOG_DIR}/precompute_emilia_rank${RANK}.log"
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
    echo "One or more workers failed — check logs in ${LOG_DIR}."
    exit 1
fi
