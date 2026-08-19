#!/usr/bin/env bash
# ============================================================================
# run_train_asr.sh
# Train a Character-level Transducer ASR system with LibriTTS.
# ============================================================================

set -euo pipefail

STREAMASR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python "${STREAMASR_ROOT}/train/train_asr_char.py" \
    "${STREAMASR_ROOT}/hparams/chunk_streaming_char.yaml"