#!/usr/bin/env bash
# Launch StreamSLM SALMon evaluation in WORLD parallel shards.
#
# Each shard = one `sr 1 24 --qos=q-low` GPU; they run concurrently and
# write per-shard JSONs. After all shards finish, this script merges them
# into a single summary file matching the layout of the single-GPU launcher
# (run_eval_salmon.sh).
#
# Override knobs via env:
#   SLM_CKPT     : streamSLM checkpoint (.pt)
#   TEACHER_CKPT : streamAlign RVQ teacher
#   DATASETS     : SALMon task folders (default: 5 in-scope consistency tasks)
#   OUT_DIR      : results dir (auto-namespaced inside)
#   SCORING_MODE : "loss" (default, TASTE-style mean-CE) or "likelihood"
#   TEXT_WEIGHT     : weight on text-stream loss term     (default: 1.0)
#   ACOUSTIC_WEIGHT : weight on acoustic-stream loss term (default: 1.0)
#   RVQ_LOSS_WEIGHTS : per-codebook acoustic weights for 'loss' mode
#                      (comma list, e.g. 1,1,...; empty = uniform)
#   WORLD        : number of parallel shards (default: 8)
#   VRAM         : per-shard GPU VRAM in GB (default: 24)
#   SR_QOS       : QoS pool, e.g. q-low (default). Empty = mid quota.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# Defaults align with the current canonical hier-durfirst-durreg sweep
# (RVQ R16 teacher + R16 SLM). Override via env vars to evaluate other configs.
SLM_CKPT=${SLM_CKPT:-/home/streamalign/checkpoints/streamSLM/streamalign_slm_r16/step_00070000.pt}
TEACHER_CKPT=${TEACHER_CKPT:-/home/streamalign/streamASR/checkpoints/streamalign_r16/epoch_22.pt}

DATA_ROOT=${DATA_ROOT:-/home/datasets/SALMon}
# TGs are written alongside the wavs (see run_eval_tg_gen.sh); only set
# TG_ROOT if they live in a separate mirror.
TG_ROOT=${TG_ROOT:-${DATA_ROOT}}
OUT_DIR=${OUT_DIR:-${REPO_ROOT}/results/salmon}
LOG_DIR=${LOG_DIR:-${REPO_ROOT}/logs/streamSLM_eval}
mkdir -p "${LOG_DIR}"

DATASETS=${DATASETS:-"energy_consistency gender_consistency pitch_consistency sentiment_consistency speaker_consistency"}
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
SR_EXCLUDE=${SR_EXCLUDE:-kandinsky,greco,namjune,matisse}
WORLD=${WORLD:-8}
VRAM=${VRAM:-24}
SR_QOS=${SR_QOS-q-low}

QOS_ARGS=()
if [[ -n "${SR_QOS}" ]]; then
  QOS_ARGS+=(--qos="${SR_QOS}")
fi

CKPT_TAG="$(basename "$(dirname "${SLM_CKPT}")")_$(basename "${SLM_CKPT}" .pt)"
TAG_DIR="${OUT_DIR}/${CKPT_TAG}/${SCORING_MODE}"
mkdir -p "${TAG_DIR}"

echo "[launch] SLM_CKPT=${SLM_CKPT}"
echo "[launch] DATASETS=${DATASETS}"
echo "[launch] OUT_DIR=${OUT_DIR}  TAG_DIR=${TAG_DIR}"
echo "[launch] SCORING_MODE=${SCORING_MODE}  WORLD=${WORLD}"
echo "[launch] TEXT_WEIGHT=${TEXT_WEIGHT}  ACOUSTIC_WEIGHT=${ACOUSTIC_WEIGHT}"
echo "[launch] RVQ_LOSS_WEIGHTS=${RVQ_LOSS_WEIGHTS:-<uniform>}"
echo "[launch] sr 1 ${VRAM} qos=${SR_QOS:-mid} exclude=${SR_EXCLUDE}"

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export RVQ_R="${RVQ_R}"
export RVQ_CODEBOOK_SIZE="${RVQ_CODEBOOK_SIZE}"
# SDPA attention so the eval works on Turing/Volta as well as Ampere.
export STREAMSLM_ATTN_IMPL="${STREAMSLM_ATTN_IMPL:-sdpa}"

PIDS=()
LOGS=()
for R in $(seq 0 $((WORLD-1))); do
  LOG_FILE="${LOG_DIR}/salmon_${CKPT_TAG}_${SCORING_MODE}.shard${R}_of${WORLD}.log"
  rm -f "${LOG_FILE}"
  echo "[launch] shard ${R}/${WORLD} -> ${LOG_FILE}"
  nohup sr 1 "${VRAM}" "${QOS_ARGS[@]}" --exclude="${SR_EXCLUDE}" \
    python -m streamSLM.eval.salmon \
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
import json
from pathlib import Path

tag_dir = Path("${TAG_DIR}")
world = ${WORLD}
tasks = "${DATASETS}".split()
out_summary = {"scoring_mode": "${SCORING_MODE}",
               "text_weight": ${TEXT_WEIGHT}, "acoustic_weight": ${ACOUSTIC_WEIGHT},
               "rvq_loss_weights": "${RVQ_LOSS_WEIGHTS}",
               "tasks": {}}

for task in tasks:
    shards = sorted(tag_dir.glob(f"salmon_{task}.shard*_of{world}.json"))
    if len(shards) != world:
        print(f"[merge] WARN {task}: found {len(shards)} shards, expected {world}")
    correct = total = invalid = 0
    all_scores = []
    scoring_mode = None
    for sp in shards:
        d = json.loads(sp.read_text())
        correct += d["correct"]; total += d["total"]; invalid += d["invalid"]
        all_scores.extend(d.get("scores", []))
        scoring_mode = d.get("scoring_mode", scoring_mode)
    accuracy = correct / total if total else 0.0
    print(f"[merge] {task}: acc={accuracy:.4f} ({correct}/{total}) invalid={invalid}")
    merged = {
        "task": task, "scoring_mode": scoring_mode, "accuracy": accuracy,
        "correct": correct, "total": total, "invalid": invalid,
        "scores": sorted(all_scores, key=lambda x: x.get("ind", 0)),
    }
    (tag_dir / f"salmon_{task}.json").write_text(json.dumps(merged, indent=2))
    out_summary["tasks"][task] = {
        "accuracy": accuracy, "correct": correct, "total": total, "invalid": invalid,
    }

(tag_dir / "salmon_summary.json").write_text(json.dumps(out_summary, indent=2))
accs = [v["accuracy"] for v in out_summary["tasks"].values()]
if accs:
    print(f"[merge] mean accuracy across {len(accs)} tasks: {sum(accs)/len(accs):.4f}")
print(f"[merge] summary -> {tag_dir / 'salmon_summary.json'}")
EOF

echo "[done] salmon ${WORLD}-way parallel eval finished"
