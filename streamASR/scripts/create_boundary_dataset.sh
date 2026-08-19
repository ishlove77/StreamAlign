#!/usr/bin/env bash
# create_boundary_dataset.sh
#
# Step 1 (BoundaryClassifier pipeline): Stream LibriSpeech audio through
# the frozen word-level StreamingASR model and extract (enc_feat, pred_feat,
# label) triples for training the BoundaryClassifier.
#
# Output .pt files are written to the dataset_dir declared in the hparams YAML
# (default: streamASR/data/boundary_dataset/).
#
# Usage
# -----
#   bash create_boundary_dataset.sh                         # default paths
#   bash create_boundary_dataset.sh --max-files 500         # quick test run
#   bash create_boundary_dataset.sh --tokenizer /path/to/tokenizer.ckpt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Defaults ─────────────────────────────────────────────────────────────────
HPARAMS="${SCRIPT_DIR}/hparams/boundary_classifier.yaml"
MAX_FILES_ARG=""
TOKENIZER_ARG=""

# ── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --hparams)      HPARAMS="$2";           shift 2 ;;
        --max-files)    MAX_FILES_ARG="--max_files_per_split $2"; shift 2 ;;
        --tokenizer)    TOKENIZER_ARG="--tokenizer_ckpt $2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

echo "========================================================"
echo "Creating BoundaryClassifier dataset"
echo "  Hparams : ${HPARAMS}"
echo "========================================================"

# shellcheck disable=SC2086
python "${SCRIPT_DIR}/data/create_boundary_dataset.py" \
    "${HPARAMS}"       \
    ${MAX_FILES_ARG}   \
    ${TOKENIZER_ARG}

echo ""
echo "Dataset creation complete."
