#!/usr/bin/env bash
# ============================================================================
# run_train_asr.sh
# Train a Character-level Transducer ASR system with LibriTTS.
# ============================================================================

set -euo pipefail

STREAMASR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python "${STREAMASR_ROOT}/train/train_asr_word_fastemit.py" \
    "${STREAMASR_ROOT}/hparams/chunk_streaming_word_fastemit.yaml"
