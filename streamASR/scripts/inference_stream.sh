#!/usr/bin/env bash
# ============================================================================
# run_inference.sh
# Run streaming inference and re-synthesis with the Data2Vec student model
# and CosyVoice flow+hift vocoder.
# ============================================================================

set -euo pipefail

STREAMASR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NUM_GPUS=$(python -c "import torch; print(max(1, torch.cuda.device_count()))")

python "${STREAMASR_ROOT}/inference/inference_core.py" \
    --input_dir="/home/datasets/LibriTTS" \
    --output_dir="${OUT_DIR:-/home/datasets/experiments/streamalign_recon}" \
    --split="test-clean" \
    --output_split="16k-test-clean" \
    --batch_size=1 \
    --chunk_size=4 \
    --left_context=32 \
    --hparams="${STREAMASR_ROOT}/hparams/alignment.yaml" \
    --truthmodel_checkpoint_path="${STREAMASR_ROOT}/results/char_asr_ckpt" \
    --studentmodel_checkpoint_path="${STUDENT_CKPT:-/home/streamalign/streamASR/checkpoints/streamalign_r16/epoch_22.pt}" \
    --test_csv="/home/datasets/LibriTTS/csv/test-clean.csv" \
    --word_hparams="${STREAMASR_ROOT}/hparams/chunk_streaming_word_fastemit.yaml" \
    --word_checkpoint="${STREAMASR_ROOT}/train/results/conformer_transducer_char/word_fastemit/save/word_asr_ckpt" \
    --boundary_classifier_ckpt="${STREAMASR_ROOT}/train/results/best_model.pt" \
    --world_size="${NUM_GPUS}" \
    --tokenizer="llama"
