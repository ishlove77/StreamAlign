#!/usr/bin/env bash
# ============================================================================
# run_train_asr_word_fastemit.sh
# Train the word-level FastEmit Transducer ASR on LibriSpeech (+ Emilia).
# Prepares the LibriSpeech CSVs first (skipped if they already exist).
# ============================================================================

set -euo pipefail

STREAMASR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- environment (override via env) ----------------------------------------
PYTHON=${PYTHON:-python}
LIBRISPEECH_ROOT=${LIBRISPEECH_ROOT:?set LIBRISPEECH_ROOT to your LibriSpeech root}
EMILIA_ROOT=${EMILIA_ROOT:?set EMILIA_ROOT to your Emilia dataset root}
EMILIA_CSV=${EMILIA_CSV:-${EMILIA_ROOT}/emilia_en_400h.csv}
export LIBRISPEECH_ROOT EMILIA_ROOT
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

"${PYTHON}" "${STREAMASR_ROOT}/train/train_asr_word_fastemit.py" \
    "${STREAMASR_ROOT}/hparams/chunk_streaming_word_fastemit.yaml" \
    --data_folder "${LIBRISPEECH_ROOT}" \
    --emilia_data_folder "${EMILIA_ROOT}" \
    --emilia_train_csv "${EMILIA_CSV}" "$@"
