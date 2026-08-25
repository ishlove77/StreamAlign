#!/usr/bin/env bash
# ============================================================================
# run_precompute_segment_inputs.sh
# Precompute and cache z_raw, char_alignment, word_alignment, char_data,
# gt_char_indices, and spk_emb for all LibriTTS splits.
# ============================================================================

set -euo pipefail

STREAMASR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ---- environment (override via env) ----------------------------------------
PYTHON=${PYTHON:-python}
SEGMENT_CACHE_DIR=${SEGMENT_CACHE_DIR:-${STREAMASR_ROOT}/train/results/segment_cache}
CHAR_ASR_CKPT=${CHAR_ASR_CKPT:-${STREAMASR_ROOT}/results/char_asr_ckpt}
RESUME_PATH=${RESUME_PATH:?set RESUME_PATH to the tokenizer checkpoint (epoch_*.pt)}
# ----------------------------------------------------------------------------

NUM_GPUS=$("${PYTHON}" -c "import torch; print(max(1, torch.cuda.device_count()))")

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port=29503 \
    "${STREAMASR_ROOT}/data/precompute_segment_inputs.py" \
    --output_dir="${SEGMENT_CACHE_DIR}" \
    --batch_size=1 \
    --num_workers=8 \
    --chunk_size=4 \
    --left_context=32 \
    --hparams="${STREAMASR_ROOT}/hparams/alignment.yaml" \
    --checkpoint_path="${CHAR_ASR_CKPT}" \
    --resume_path="${RESUME_PATH}" \
    --splits="train,val"
