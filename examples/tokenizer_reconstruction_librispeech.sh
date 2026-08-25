#!/usr/bin/env bash
# =============================================================================
# Tokenizer example: LibriSpeech reconstruction.
#
# Encodes LibriSpeech utterances into StreamAlign R=32 units and resynthesizes
# them with the two-stage CosyVoice decoder (flow + HiFT). Output is one
# reconstructed .flac per input utterance, mirroring the input directory
# layout, which you can listen to or score with an ASR model.
#
# Usage:
#   bash examples/tokenizer_reconstruction_librispeech.sh [N_UTTS]
#
# Environment:
#   WEIGHTS         weights dir from download_weights.sh   (default ./weights)
#   LIBRISPEECH     LibriSpeech root
#   COSYVOICE_ROOT  CosyVoice checkout (provides the flow/HiFT decoder)
#   OUT_DIR         where reconstructions are written
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STREAMASR="${REPO_ROOT}/streamASR"

N_UTTS="${1:-10}"
: "${WEIGHTS:=${REPO_ROOT}/weights}"
: "${LIBRISPEECH:=${LIBRISPEECH_ROOT:-/data/LibriSpeech}}"
: "${COSYVOICE_ROOT:=${STREAMASR}/third_party/CosyVoice}"
: "${OUT_DIR:=${REPO_ROOT}/outputs/reconstruction}"
: "${SPLIT:=test-clean}"

W="${WEIGHTS}/Streamalign-R32"
TEST_CSV="${TEST_CSV:-${LIBRISPEECH}/csv/${SPLIT}.first${N_UTTS}.csv}"

if [ ! -f "${W}/rvq_tokenizer/final.pt" ]; then
    echo "Missing weights at ${W}. Run: bash examples/download_weights.sh" >&2
    exit 1
fi
if [ ! -f "${TEST_CSV}" ]; then
    echo "Missing manifest ${TEST_CSV}." >&2
    echo "Provide one via TEST_CSV=<path> with columns ID,duration,wav,spk_id,wrd." >&2
    exit 1
fi

# The vendored SpeechBrain lives under streamASR/; CosyVoice supplies the decoder.
export PYTHONPATH="${STREAMASR}:${COSYVOICE_ROOT}:${COSYVOICE_ROOT}/third_party/Matcha-TTS:${PYTHONPATH:-}"
export COSYVOICE_ROOT
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
# R=32 / C=512 must match the released tokenizer. Leave RVQ_CODEBOOK_DIM unset:
# codebook_dim defaults to feat_dim=256, which is what the checkpoint expects.
export RVQ_R=32 RVQ_CODEBOOK_SIZE=512
unset RVQ_CODEBOOK_DIM

mkdir -p "${OUT_DIR}"
cd "${STREAMASR}"

python inference/inference_stream.py \
    --input_dir="${LIBRISPEECH}" \
    --output_dir="${OUT_DIR}" \
    --split="${SPLIT}" --output_split="${SPLIT}" \
    --test_csv="${TEST_CSV}" \
    --batch_size=1 --chunk_size=4 --left_context=32 \
    --hparams="${W}/alignment.yaml" \
    --truthmodel_checkpoint_path="${W}/alignment_model" \
    --studentmodel_checkpoint_path="${W}/rvq_tokenizer/final.pt" \
    --word_hparams="${W}/streaming_asr/chunk_streaming_word_fastemit.yaml" \
    --word_checkpoint="${W}/streaming_asr" \
    --word_tokenizer_ckpt="${W}/streaming_asr/tokenizer.ckpt" \
    --boundary_classifier_ckpt="${W}/boundary_classifier/best_model.pt" \
    --world_size=1 --tokenizer=llama

echo
echo "Reconstructions written to ${OUT_DIR}/${SPLIT}/"
echo "Listen to them alongside the originals under ${LIBRISPEECH}/${SPLIT}/."
