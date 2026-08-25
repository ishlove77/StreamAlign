#!/usr/bin/env bash
# =============================================================================
# train_tokenizer_r32_pipeline.sh
#
# Full StreamAlign R=32 tokenizer pipeline (paper recipe), mirroring
# train_tokenizer_r16_pipeline.sh but with the paper's plain-RVQ trainers:
#
#   char_asr    : Stage 1 — char-level RNN-T aligner (frozen encoder for
#                 everything below). ~10 epochs sufficed (dev CER 1.27%);
#                 pick the best-CER SpeechBrain checkpoint.
#   boundary    : boundary-classifier dataset + training (needs the word ASR).
#   continuous  : Stage 2 phase A — continuous recon, quantizer bypassed
#                 (RVQ_BYPASS=1). Stop at L_REC plateau (~15 epochs).
#   r32         : Stage 2 phase B — plain RVQ ON (R=32, codebook 512),
#                 non-encoder weights initialized from the phase-A ckpt
#                 (fresh optimizer). Codebook converges here (~14 epochs).
#   cosine      : Stage 2 phase C — cosine LR decay 1e-5 -> 0, initialized
#                 from the phase-B ckpt. COSINE_TOTAL_STEPS must equal this
#                 phase's total optimizer steps.
#   eval        : streaming reconstruction (inference_core.py) on
#                 LibriSpeech test-clean + whisper-large-v3 WER/CER.
#
# Reference results (full test-clean 2620, streaming, whisper-large-v3):
#   phase C epoch 13: WER 4.43% / CER 1.92% / UTMOS 4.23 / SECS(RawNet3) 0.585
#   (paper: WER 4.41%)
#   NOTE: epoch 13 is the SELECTED final checkpoint. Epochs 14-20 (cosine
#   tail, LR 2.7e-6 -> 0) plateaued with no further loss/metric gain, so
#   there is little value in running the cosine stage past ~13 epochs.
#
# Usage:
#   bash scripts/train_tokenizer_r32_pipeline.sh <stage>
#     <stage> = char_asr | boundary | continuous | r32 | cosine | eval
#
# Override env vars on the command line, e.g.:
#   PHASE_CKPT=checkpoints/streamalign_r32_stage2a/epoch_15.pt \
#     bash scripts/train_tokenizer_r32_pipeline.sh r32
#
# Resume note (cosine): the cosine scheduler step is NOT checkpointed. To
# resume an interrupted cosine run, set LEARNING_RATE to the LR at the
# interruption point and COSINE_TOTAL_STEPS to the REMAINING steps (a "tail
# cosine"), plus RESUME_PATH=<epoch_N.pt>.
# =============================================================================
set -euo pipefail

STAGE="${1:?usage: train_tokenizer_r32_pipeline.sh <char_asr|boundary|continuous|r32|cosine|eval>}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STREAMASR_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${STREAMASR_ROOT}:${PYTHONPATH:-}"

# ---- Tokenizer / quantizer config -------------------------------------------
: "${RVQ_R:=32}"                 # paper: R=32 codebook layers
: "${RVQ_CODEBOOK_SIZE:=512}"
: "${COMMIT_WEIGHT:=1.0}"
export RVQ_R RVQ_CODEBOOK_SIZE COMMIT_WEIGHT
unset RVQ_CODEBOOK_DIM 2>/dev/null || true   # codebook_dim = feat_dim (256)

# ---- Data / checkpoint paths (override via env) -----------------------------
: "${PYTHON:=python}"
# TRUTH_CKPT: point (or symlink results/char_asr_ckpt) at your best char-ASR
# SpeechBrain checkpoint dir (train/results/conformer_transducer_char/char_asr/save/CKPT+...).
: "${TRUTH_CKPT:=${STREAMASR_ROOT}/results/char_asr_ckpt}"  # Stage-1 char RNN-T CKPT dir
: "${HPARAMS:=${STREAMASR_ROOT}/hparams/alignment.yaml}"
: "${LIBRISPEECH_ROOT:?set LIBRISPEECH_ROOT to your LibriSpeech root}"
: "${LIBRITTS_ROOT:?set LIBRITTS_ROOT to your LibriTTS root}"
: "${EMILIA_ROOT:?set EMILIA_ROOT to your Emilia dataset root}"
: "${EMILIA_CSV:=${EMILIA_ROOT}/emilia_en_400h.csv}"
: "${TEXTGRID_ROOT:=${LIBRITTS_ROOT}/chunk_textgrids_word_model_final2}"
: "${COSYVOICE_ROOT:=${STREAMASR_ROOT}/third_party/CosyVoice}"
# eval-only: reused word/BPE streaming ASR + its tokenizer + boundary clf
: "${WORD_ASR_CKPT:=${STREAMASR_ROOT}/train/results/conformer_transducer_char/word_fastemit/save/word_asr_ckpt}"
: "${WORD_TOKENIZER_CKPT:=${STREAMASR_ROOT}/train/results/conformer_transducer_char/word_fastemit/pretrained/tokenizer.ckpt}"
: "${BOUNDARY_CKPT:=${STREAMASR_ROOT}/train/results/boundary_classifier/save/best_model.pt}"
: "${MASTER_PORT:=29663}"
# Exported: the trainers / utils read these names from the environment.
export LIBRISPEECH_ROOT LIBRITTS_ROOT EMILIA_ROOT TEXTGRID_ROOT COSYVOICE_ROOT

NUM_GPUS="${NUM_GPUS:-$("${PYTHON}" -c "import torch; print(max(1, torch.cuda.device_count()))")}"
cd "${STREAMASR_ROOT}"

# Global batch 16 reproduced the paper (4 GPU x bs4 or 8 GPU x bs2).
: "${BATCH_SIZE:=4}"

common_args=(
    --batch_size="${BATCH_SIZE}"
    --weight_decay=1e-5
    --mask_prob=0.08
    --chunk_size=4
    --left_context=32
    --hparams="${HPARAMS}"
    --checkpoint_path="${TRUTH_CKPT}"
    --emilia_sample_ratio=0.1
    --emilia_csv="${EMILIA_CSV}"
    --emilia_data_root="${EMILIA_ROOT}"
    --data_root="${LIBRITTS_ROOT}"
    --use_precomputed_features
)

run_trainer() {  # run_trainer <trainer.py> <extra args...>
    local trainer="$1"; shift
    torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" \
        "${trainer}" "${common_args[@]}" "$@"
}

case "${STAGE}" in

  char_asr)
    # Stage 1: char-level RNN-T aligner (torchaudio RNN-T backend).
    # Yaml output paths are relative; run from train/ so results land under
    # streamASR/train/results/ (the canonical location).
    cd "${STREAMASR_ROOT}/train"
    # Prep the LibriSpeech CSVs first (skipped if they already exist).
    CSV_DIR="results/conformer_transducer_char/char_asr"
    if [[ ! -f "${CSV_DIR}/train-clean-100.csv" ]]; then
      echo "[prep] generating LibriSpeech CSVs under ${CSV_DIR}"
      "${PYTHON}" -c "
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
    exec torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" \
        "${STREAMASR_ROOT}/train/train_asr_char.py" \
        "${STREAMASR_ROOT}/hparams/chunk_streaming_char_h200.yaml" \
        --data_folder "${LIBRISPEECH_ROOT}" --precision=fp16
    ;;

  boundary)
    # Boundary classifier: dataset extraction, then training.
    "${PYTHON}" data/create_boundary_dataset.py hparams/boundary_classifier_h200.yaml \
        --data_folder "${LIBRISPEECH_ROOT}"
    exec "${PYTHON}" train/train_boundary_classifier.py hparams/boundary_classifier_h200.yaml
    ;;

  continuous)
    # Stage 2 phase A: continuous recon, quantizer bypassed. The quantizer is
    # still constructed, so this checkpoint initializes the r32 stage.
    export RVQ_BYPASS=1
    : "${LEARNING_RATE:=1e-4}"
    : "${EXP_NAME:=streamalign_r32_stage2a_continuous}"
    : "${NUM_EPOCHS:=60}"
    exec_args=()
    ;;

  r32)
    # Stage 2 phase B: plain RVQ ON. Non-encoder weights come from the
    # phase-A ckpt via --subalign_init_path (fresh optimizer, epoch reset) —
    # this is the recipe our reference numbers were produced with.
    export RVQ_BYPASS=0
    : "${PHASE_CKPT:?set PHASE_CKPT to the continuous-phase epoch_N.pt}"
    : "${LEARNING_RATE:=1e-4}"
    : "${EXP_NAME:=streamalign_r32_stage2b_rvq}"
    : "${NUM_EPOCHS:=28}"
    exec_args=(--subalign_init_path="${PHASE_CKPT}")
    ;;

  cosine)
    # Stage 2 phase C: cosine LR decay (peak 1e-5 -> 0), init from phase B.
    # COSINE_TOTAL_STEPS = steps_per_epoch x NUM_EPOCHS of THIS phase
    # (train_sample_ratio=1.0, global batch 16 -> ~18.5k steps/epoch).
    export RVQ_BYPASS=0
    : "${PHASE_CKPT:?set PHASE_CKPT to the r32-phase epoch_N.pt}"
    : "${LEARNING_RATE:=1e-5}"
    : "${EXP_NAME:=streamalign_r32_stage2c_cosine}"
    : "${NUM_EPOCHS:=20}"
    : "${COSINE_TOTAL_STEPS:=390000}"
    : "${COSINE_MIN_LR:=0}"
    export COSINE_TOTAL_STEPS COSINE_MIN_LR
    exec_args=(--subalign_init_path="${PHASE_CKPT}")
    ;;

  eval)
    # Streaming reconstruction + whisper WER. RVQ_BYPASS=1 evaluates a
    # continuous-phase ckpt; 0 (default) once RVQ is enabled.
    : "${CKPT:?set CKPT to the tokenizer epoch_N.pt to evaluate}"
    export RVQ_BYPASS="${RVQ_BYPASS:-0}"
    OUT_DIR="${OUT_DIR:-eval_out/$(basename "${CKPT%.pt}")}"
    mkdir -p "${OUT_DIR}"
    "${PYTHON}" inference/inference_core.py \
        --variant=rvq \
        --input_dir="${LIBRISPEECH_ROOT}" \
        --output_dir="${OUT_DIR}/recon" \
        --split=test-clean --output_split=test-clean \
        --batch_size=1 \
        --chunk_size=4 --left_context=32 \
        --hparams="${HPARAMS}" \
        --truthmodel_checkpoint_path="${TRUTH_CKPT}" \
        --studentmodel_checkpoint_path="${CKPT}" \
        --test_csv=data/librispeech_test-clean.csv \
        --word_hparams=hparams/chunk_streaming_word_fastemit.yaml \
        --word_checkpoint="${WORD_ASR_CKPT}" \
        --word_tokenizer_ckpt="${WORD_TOKENIZER_CKPT}" \
        --boundary_classifier_ckpt="${BOUNDARY_CKPT}" \
        --world_size=1 --tokenizer="${TOKENIZER:-llama}" \
        --streamability=true --resume=true
    OUT_DIR="${OUT_DIR}" "${PYTHON}" - <<'PY'
# Pair reconstructed flacs with reference transcripts -> recon.csv
import csv, glob, os
out_dir = os.environ["OUT_DIR"]
refs = {r["ID"]: r for r in csv.DictReader(open("data/librispeech_test-clean.csv"))}
rows = []
for fp in sorted(glob.glob(os.path.join(out_dir, "recon", "**", "*.flac"), recursive=True)):
    sid = os.path.splitext(os.path.basename(fp))[0]
    if sid in refs:
        rows.append({"ID": sid, "duration": refs[sid].get("duration", ""),
                     "wav": fp, "spk_id": refs[sid].get("spk_id", ""), "wrd": refs[sid]["wrd"]})
assert rows, "no reconstructed flac matched the reference CSV"
with open(os.path.join(out_dir, "recon.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["ID", "duration", "wav", "spk_id", "wrd"])
    w.writeheader(); w.writerows(rows)
print(f"[recon_csv] wrote {len(rows)} rows")
PY
    exec "${PYTHON}" evaluate/WER/evaluate_word_whisper.py \
        --test_csv="${OUT_DIR}/recon.csv" --model=openai/whisper-large-v3 \
        --num_samples=2620 --hyps_out="${OUT_DIR}/wer_hyps.json"
    ;;

  *)
    echo "unknown stage: ${STAGE} (expected char_asr|boundary|continuous|r32|cosine|eval)" >&2
    exit 1
    ;;
esac

# ── Stage-2 training launch (continuous / r32 / cosine share this) ───────────
TRAINER=train/train_tokenizer.py
[ "${STAGE}" = "cosine" ] && TRAINER=train/train_tokenizer_cosine.py
TRAIN_SAMPLE_RATIO=0.5
[ "${STAGE}" = "cosine" ] && TRAIN_SAMPLE_RATIO=1.0

echo "[r32-pipeline:${STAGE}] R=${RVQ_R} C=${RVQ_CODEBOOK_SIZE} lr=${LEARNING_RATE} exp=${EXP_NAME}"

run_trainer "${TRAINER}" \
    --exp_name="${EXP_NAME}" \
    --learning_rate="${LEARNING_RATE}" \
    --num_epochs="${NUM_EPOCHS}" \
    --train_sample_ratio="${TRAIN_SAMPLE_RATIO}" \
    ${RESUME_PATH:+--resume_path="${RESUME_PATH}"} \
    ${exec_args[@]+"${exec_args[@]}"}
