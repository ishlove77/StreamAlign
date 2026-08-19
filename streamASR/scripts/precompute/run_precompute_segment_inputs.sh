#!/usr/bin/env bash
# ============================================================================
# run_precompute_segment_inputs.sh
# Precompute and cache z_raw, char_alignment, word_alignment, char_data,
# gt_char_indices, and spk_emb for all LibriTTS splits.
# ============================================================================

set -euo pipefail

STREAMASR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NUM_GPUS=$(python -c "import torch; print(max(1, torch.cuda.device_count()))")

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port=29503 \
    "${STREAMASR_ROOT}/train/precompute_segment_inputs.py" \
    --output_dir="/home/streamalign/segment_cache" \
    --batch_size=1 \
    --num_workers=8 \
    --chunk_size=4 \
    --left_context=32 \
    --hparams="${STREAMASR_ROOT}/hparams/alignment.yaml" \
    --checkpoint_path="${STREAMASR_ROOT}/results/char_asr_ckpt" \
    --resume_path="${RESUME_PATH:-/home/streamalign/streamASR/checkpoints/streamalign_r16/epoch_22.pt}" \
    --splits="train,val"
