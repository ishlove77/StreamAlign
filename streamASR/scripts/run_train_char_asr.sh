#!/usr/bin/env bash
# ============================================================================
# run_train_char_asr.sh
# Train a Character-level Transducer ASR system on LibriSpeech.
# Prepares the LibriSpeech CSVs first (skipped if they already exist).
# ============================================================================

set -euo pipefail

STREAMASR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- environment (override via env) ----------------------------------------
PYTHON=${PYTHON:-python}
LIBRISPEECH_ROOT=${LIBRISPEECH_ROOT:?set LIBRISPEECH_ROOT to your LibriSpeech root}
export LIBRISPEECH_ROOT
# ----------------------------------------------------------------------------

# Output paths in the yaml are relative; run from train/ so results land
# under streamASR/train/results/ (the canonical location).
cd "${STREAMASR_ROOT}/train"

CSV_DIR="results/conformer_transducer_char/char_asr"
if [[ ! -f "${CSV_DIR}/train-clean-100.csv" ]]; then
  echo "[prep] generating LibriSpeech CSVs under ${CSV_DIR}"
  PYTHONPATH="${STREAMASR_ROOT}:${PYTHONPATH:-}" "${PYTHON}" -c "
import os
from utils.librispeech_prepare import prepare_librispeech
root = '${LIBRISPEECH_ROOT}'
def have(splits):
    kept = [s for s in splits if os.path.isdir(os.path.join(root, s))]
    for s in set(splits) - set(kept):
        print(f'[prep] split {s} not found under {root}; skipping')
    return kept
tr = have(['train-clean-100', 'train-clean-360', 'train-other-500'])
dev = have(['dev-clean'])
te = have(['test-clean', 'test-other'])
# Optional smoke knob: cap sentences per split (0/unset = all).
n = int(os.environ.get('LIBRISPEECH_PREP_N', '0'))
prepare_librispeech(
    data_folder=root,
    save_folder='${CSV_DIR}',
    tr_splits=tr,
    dev_splits=dev,
    te_splits=te,
    select_n_sentences=([n] * len(tr + dev + te) if n else None),
)
"
fi

"${PYTHON}" "${STREAMASR_ROOT}/train/train_asr_char.py" \
    "${STREAMASR_ROOT}/hparams/chunk_streaming_char.yaml" \
    --data_folder "${LIBRISPEECH_ROOT}" "$@"
