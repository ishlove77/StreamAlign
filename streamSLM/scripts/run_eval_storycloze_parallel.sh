#!/usr/bin/env bash
# Launch StreamSLM StoryCloze evaluation in WORLD parallel shards.
#
# Each shard = one ${LAUNCHER} job (or a local process when LAUNCHER is
# empty — for local runs prefer run_eval_storycloze.sh); they run concurrently and
# write per-shard JSONs. After all shards finish, this script merges them
# into a single summary file matching the layout of the single-GPU launcher.
#
# Override knobs via env:
#   SLM_CKPT     : streamSLM checkpoint (.pt)
#   TEACHER_CKPT : streamAlign RVQ teacher
#   DATASETS     : "sSC tSC" (default)
#   OUT_DIR      : results dir (auto-namespaced inside)
#   SCORING_MODE : "loss" (default, TASTE-style mean-CE) or "likelihood"
#   TEXT_WEIGHT     : weight on text-stream loss term     (default: 1.0)
#   ACOUSTIC_WEIGHT : weight on acoustic-stream loss term (default: 1.0)
#   RVQ_LOSS_WEIGHTS : per-codebook acoustic weights for 'loss' mode
#                      (comma list, e.g. 10,3,3,...; empty = uniform)
#   WORLD        : number of parallel shards (default: 8)
#   LAUNCHER     : per-shard job-submission prefix (e.g. "sr 1 24 --qos=q-low")

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# Defaults align with the current canonical hier-durfirst-durreg sweep
# (RVQ teacher + R16 SLM). Override SLM_CKPT / TEACHER_CKPT / VARIANT etc.
# via env vars to evaluate other configs.
CHECKPOINTS_ROOT=${CHECKPOINTS_ROOT:-${REPO_ROOT}/weights}
SLM_CKPT=${SLM_CKPT:-${CHECKPOINTS_ROOT}/Streamalign-SLM-R16/step_00185000.pt}
TEACHER_CKPT=${TEACHER_CKPT:-${CHECKPOINTS_ROOT}/Streamalign-R16/rvq_teacher/epoch_22.pt}
ASR_HPARAMS=${ASR_HPARAMS:-${REPO_ROOT}/streamASR/hparams/alignment.yaml}
TRUTH_MODEL_CKPT=${TRUTH_MODEL_CKPT:-${REPO_ROOT}/streamASR/results/char_asr_ckpt}
COSYVOICE_ROOT=${COSYVOICE_ROOT:-${REPO_ROOT}/streamASR/third_party/CosyVoice}
export COSYVOICE_ROOT
export PYTHONPATH="${COSYVOICE_ROOT}:${COSYVOICE_ROOT}/third_party/Matcha-TTS:${PYTHONPATH:-}"

DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to your StoryCloze dataset root}
# TGs are written alongside the wavs (see run_eval_tg_gen.sh); only set
# TG_ROOT if they live in a separate mirror.
TG_ROOT=${TG_ROOT:-${DATA_ROOT}}
OUT_DIR=${OUT_DIR:-${REPO_ROOT}/results/storycloze}
LOG_DIR=${LOG_DIR:-${REPO_ROOT}/logs/streamSLM_eval}
mkdir -p "${LOG_DIR}"

DATASETS=${DATASETS:-"sSC tSC"}
VARIANT=${VARIANT:-rvq}
# RVQ knobs.
RVQ_R=${RVQ_R:-16}
RVQ_CODEBOOK_SIZE=${RVQ_CODEBOOK_SIZE:-512}
SCORING_MODE=${SCORING_MODE:-loss}
# Text:acoustic loss ratio for the pair score; defaults give 1:1.
TEXT_WEIGHT=${TEXT_WEIGHT:-1.0}
ACOUSTIC_WEIGHT=${ACOUSTIC_WEIGHT:-1.0}
# Optional per-codebook acoustic weights for 'loss' mode (comma list,
# e.g. the training rvq_loss_weights); empty = uniform sum over R.
RVQ_LOSS_WEIGHTS=${RVQ_LOSS_WEIGHTS:-}
WORLD=${WORLD:-8}
# Per-shard job-submission prefix; empty = run all shards locally.
LAUNCHER=${LAUNCHER:-}

CKPT_TAG="$(basename "$(dirname "${SLM_CKPT}")")_$(basename "${SLM_CKPT}" .pt)"
TAG_DIR="${OUT_DIR}/${CKPT_TAG}/${SCORING_MODE}"
mkdir -p "${TAG_DIR}"

echo "[launch] SLM_CKPT=${SLM_CKPT}"
echo "[launch] DATASETS=${DATASETS}"
echo "[launch] OUT_DIR=${OUT_DIR}  TAG_DIR=${TAG_DIR}"
echo "[launch] SCORING_MODE=${SCORING_MODE}  WORLD=${WORLD}"
echo "[launch] TEXT_WEIGHT=${TEXT_WEIGHT}  ACOUSTIC_WEIGHT=${ACOUSTIC_WEIGHT}"
echo "[launch] RVQ_LOSS_WEIGHTS=${RVQ_LOSS_WEIGHTS:-<uniform>}"
echo "[launch] launcher=${LAUNCHER:-<direct>}"

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export RVQ_R="${RVQ_R}"
export RVQ_CODEBOOK_SIZE="${RVQ_CODEBOOK_SIZE}"
# Use SDPA attention so the eval works on Turing/Volta as well as Ampere.
# FlashAttention-2 requires Ampere+; some 24 GB nodes (e.g. basquiat,
# botticelli, RTX-6000-class) are Turing and would crash every pair.
export STREAMSLM_ATTN_IMPL="${STREAMSLM_ATTN_IMPL:-sdpa}"

PIDS=()
LOGS=()
for R in $(seq 0 $((WORLD-1))); do
  LOG_FILE="${LOG_DIR}/storycloze_${CKPT_TAG}_${SCORING_MODE}.shard${R}_of${WORLD}.log"
  rm -f "${LOG_FILE}"
  echo "[launch] shard ${R}/${WORLD} -> ${LOG_FILE}"
  nohup ${LAUNCHER} \
    python -m streamSLM.eval.storycloze \
      --hparams "${ASR_HPARAMS}" \
      --truthmodel_checkpoint "${TRUTH_MODEL_CKPT}" \
      --slm_checkpoint "${SLM_CKPT}" \
      --teacher_checkpoint "${TEACHER_CKPT}" \
      --data_root "${DATA_ROOT}" \
      --tg_root "${TG_ROOT}" \
      --datasets ${DATASETS} \
      --output_dir "${OUT_DIR}" \
      --variant "${VARIANT}" \
      --rvq_num_quantizers "${RVQ_R}" \
      --rvq_codebook_size "${RVQ_CODEBOOK_SIZE}" \
      --scoring_mode "${SCORING_MODE}" \
      --text_weight "${TEXT_WEIGHT}" \
      --acoustic_weight "${ACOUSTIC_WEIGHT}" \
      --acoustic_ch_weights "${RVQ_LOSS_WEIGHTS}" \
      --rank ${R} --world ${WORLD} \
    < /dev/null > "${LOG_FILE}" 2>&1 &
  PIDS+=($!)
  LOGS+=("${LOG_FILE}")
done

echo "[launch] all ${WORLD} shards submitted; PIDs=${PIDS[*]}"
echo "[launch] waiting for completion..."

for P in "${PIDS[@]}"; do
  wait "${P}" || echo "[warn] shard pid ${P} exited non-zero (continuing)"
done

echo "[merge] combining shard JSONs into ${TAG_DIR}"
python3 - <<EOF
import json, glob, os
from pathlib import Path

tag_dir = Path("${TAG_DIR}")
world = ${WORLD}
datasets = "${DATASETS}".split()
out_summary = {"scoring_mode": "${SCORING_MODE}",
               "text_weight": ${TEXT_WEIGHT}, "acoustic_weight": ${ACOUSTIC_WEIGHT},
               "rvq_loss_weights": "${RVQ_LOSS_WEIGHTS}",
               "datasets": {}}

for ds in datasets:
    shards = sorted(tag_dir.glob(f"storycloze_{ds}.shard*_of{world}.json"))
    if len(shards) != world:
        print(f"[merge] WARN {ds}: found {len(shards)} shards, expected {world}")
    correct = total = invalid = 0
    all_scores = []
    scoring_mode = None
    for sp in shards:
        d = json.loads(sp.read_text())
        correct += d["correct"]; total += d["total"]; invalid += d["invalid"]
        all_scores.extend(d.get("scores", []))
        scoring_mode = d.get("scoring_mode", scoring_mode)
    accuracy = correct / total if total else 0.0
    print(f"[merge] {ds}: acc={accuracy:.4f} ({correct}/{total}) invalid={invalid}")
    merged = {
        "dataset": ds, "scoring_mode": scoring_mode, "accuracy": accuracy,
        "correct": correct, "total": total, "invalid": invalid,
        "scores": sorted(all_scores, key=lambda x: x.get("ind", 0)),
    }
    (tag_dir / f"storycloze_{ds}.json").write_text(json.dumps(merged, indent=2))
    out_summary["datasets"][ds] = {
        "accuracy": accuracy, "correct": correct, "total": total, "invalid": invalid,
    }

(tag_dir / "storycloze_summary.json").write_text(json.dumps(out_summary, indent=2))
print(f"[merge] summary -> {tag_dir / 'storycloze_summary.json'}")
EOF

echo "[done] storycloze ${WORLD}-way parallel eval finished"
