#!/usr/bin/env bash
# Generate streaming-ASR chunk TextGrids alongside each eval wav file.
#
# Why: SALMon + StoryCloze ship with Whisper transcripts, but Whisper sees
# the full utterance and leaks future context. The training data was unit-
# extracted under chunk TGs from the streaming Conformer-Transducer; the
# eval extractor must use the same.
#
# Output: one <wav_stem>.TextGrid per input wav, written next to the source
# wav (no separate output tree). Existing TextGrids are skipped — pass
# OVERWRITE=1 to regenerate.
#
# Resources: WORLD shards × `sr 1 24 --qos=q-low`, default WORLD=16 per
# dataset. SALMon (2000 wavs) and StoryCloze (7484 wavs) launch in parallel
# → 32 shards total, well within the 32×24GB cap.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# --- 3419 streaming-ASR ckpt (canonical) --------------------------------- #
WORD_CKPT=${WORD_CKPT:-/home/streamalign/streamASR/results/conformer_transducer_char/word_fastemit/save/word_asr_ckpt}
WORD_HPARAMS=${WORD_HPARAMS:-/home/streamalign/streamASR/hparams/chunk_streaming_word_fastemit.yaml}
TOKENIZER_CKPT=${TOKENIZER_CKPT:-/home/streamalign/streamASR/results/conformer_transducer_char/word_fastemit/pretrained/tokenizer.ckpt}

# Streaming knobs match the training-data extraction (see CLAUDE.md).
CHUNK_SIZE=${CHUNK_SIZE:-4}
LEFT_CONTEXT=${LEFT_CONTEXT:-32}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_WORKERS=${NUM_WORKERS:-4}

# --- Sharding ------------------------------------------------------------ #
WORLD_SALMON=${WORLD_SALMON:-16}
WORLD_STORYCLOZE=${WORLD_STORYCLOZE:-16}

# --- Manifests ----------------------------------------------------------- #
MANIFEST_DIR=${MANIFEST_DIR:-${REPO_ROOT}/cache/eval_tg_manifests}
SALMON_CSV=${SALMON_CSV:-${MANIFEST_DIR}/salmon.csv}
STORYCLOZE_CSV=${STORYCLOZE_CSV:-${MANIFEST_DIR}/storycloze.csv}

# --- TG output -----------------------------------------------------------#
# Default: write <wav_stem>.TextGrid directly next to each source wav.
# Override SALMON_TG_ROOT / STORYCLOZE_TG_ROOT only if you need a mirror
# (e.g. read-only data tree). The eval scripts default --tg_root to the
# wav root and pick the TGs up from there.
SALMON_DATA_ROOT=${SALMON_DATA_ROOT:-/home/datasets/SALMon}
STORYCLOZE_DATA_ROOT=${STORYCLOZE_DATA_ROOT:-/home/datasets/StoryCloze}
SALMON_TG_ROOT=${SALMON_TG_ROOT:-${SALMON_DATA_ROOT}}
STORYCLOZE_TG_ROOT=${STORYCLOZE_TG_ROOT:-${STORYCLOZE_DATA_ROOT}}

# --- Slurm knobs --------------------------------------------------------- #
SR_QOS=${SR_QOS:-q-low}
SR_EXCLUDE=${SR_EXCLUDE:-kandinsky,greco,namjune,matisse}

# --- Logging ------------------------------------------------------------- #
LOG_DIR=${LOG_DIR:-${REPO_ROOT}/logs/eval_tg_gen}
mkdir -p "${LOG_DIR}/salmon" "${LOG_DIR}/storycloze"

OVERWRITE_ARG=""
if [[ "${OVERWRITE:-0}" == "1" ]]; then OVERWRITE_ARG="--overwrite"; fi

GEN_PY="${REPO_ROOT}/streamASR/data/generate_chunk_textgrids.py"

# Auto-build manifests if missing. Each row is just the wav path; the
# generator script consumes it via --csv_manifest. cache/ is gitignored so
# we always regenerate on first run rather than ship the CSVs.
#
# SALMon: restrict to the 5 in-scope tasks (alignment / bg / rir variants
# are out of scope for the pair-classification suite). StoryCloze: both
# sSC + tSC.
SALMON_TASKS=${SALMON_TASKS:-"energy_consistency gender_consistency pitch_consistency sentiment_consistency speaker_consistency"}
STORYCLOZE_SUBSETS=${STORYCLOZE_SUBSETS:-"sSC tSC"}

mkdir -p "${MANIFEST_DIR}"
if [[ ! -s "${SALMON_CSV}" ]]; then
  echo "[run_eval_tg_gen] building ${SALMON_CSV}"
  {
    echo "wav"
    for t in ${SALMON_TASKS}; do
      find "${SALMON_DATA_ROOT}/${t}" -maxdepth 1 -name '*.wav' | sort
    done
  } > "${SALMON_CSV}"
fi
if [[ ! -s "${STORYCLOZE_CSV}" ]]; then
  echo "[run_eval_tg_gen] building ${STORYCLOZE_CSV}"
  {
    echo "wav"
    for s in ${STORYCLOZE_SUBSETS}; do
      find "${STORYCLOZE_DATA_ROOT}/${s}" -name '*.wav' | sort
    done
  } > "${STORYCLOZE_CSV}"
fi
for f in "${SALMON_CSV}" "${STORYCLOZE_CSV}"; do
  if [[ ! -s "${f}" ]]; then
    echo "[fatal] manifest missing or empty: ${f}" >&2
    exit 1
  fi
done

echo "[run_eval_tg_gen] WORD_CKPT=${WORD_CKPT}"
echo "[run_eval_tg_gen] SALMon WORLD=${WORLD_SALMON} StoryCloze WORLD=${WORLD_STORYCLOZE}"
echo "[run_eval_tg_gen] manifests: $(wc -l <"${SALMON_CSV}") SALMon lines, $(wc -l <"${STORYCLOZE_CSV}") StoryCloze lines"

# Output dirs always exist when TG_ROOT==DATA_ROOT (the wav tree); only
# mkdir when the user pointed somewhere else.
[[ "${SALMON_TG_ROOT}"     != "${SALMON_DATA_ROOT}"     ]] && mkdir -p "${SALMON_TG_ROOT}"
[[ "${STORYCLOZE_TG_ROOT}" != "${STORYCLOZE_DATA_ROOT}" ]] && mkdir -p "${STORYCLOZE_TG_ROOT}"

launch_shard() {
  local ds=$1 csv=$2 data_root=$3 tg_root=$4 rank=$5 world=$6
  local log="${LOG_DIR}/${ds}/shard${rank}_of${world}.log"
  sr 1 24 --qos="${SR_QOS}" -x "${SR_EXCLUDE}" \
    python "${GEN_PY}" \
      --hparams_file "${WORD_HPARAMS}" \
      --checkpoint   "${WORD_CKPT}" \
      --tokenizer_ckpt "${TOKENIZER_CKPT}" \
      --csv_manifest "${csv}" \
      --data_root    "${data_root}" \
      --output_dir   "${tg_root}" \
      --chunk_size   "${CHUNK_SIZE}" \
      --left_context "${LEFT_CONTEXT}" \
      --batch_size   "${BATCH_SIZE}" \
      --num_workers  "${NUM_WORKERS}" \
      --rank         "${rank}" \
      --world_size   "${world}" \
      ${OVERWRITE_ARG} \
    >"${log}" 2>&1 &
  echo "[run_eval_tg_gen] ${ds} rank ${rank}/${world} -> ${log} (pid $!)"
}

PIDS=()
for r in $(seq 0 $((WORLD_SALMON-1))); do
  launch_shard salmon "${SALMON_CSV}" "${SALMON_DATA_ROOT}" "${SALMON_TG_ROOT}" "${r}" "${WORLD_SALMON}"
  PIDS+=($!)
done
for r in $(seq 0 $((WORLD_STORYCLOZE-1))); do
  launch_shard storycloze "${STORYCLOZE_CSV}" "${STORYCLOZE_DATA_ROOT}" "${STORYCLOZE_TG_ROOT}" "${r}" "${WORLD_STORYCLOZE}"
  PIDS+=($!)
done

echo "[run_eval_tg_gen] launched ${#PIDS[@]} shards; waiting…"
FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    echo "[run_eval_tg_gen] worker pid ${pid} exited non-zero" >&2
    FAIL=1
  fi
done

# Done-line check.
echo
echo "[run_eval_tg_gen] DONE-line tail per shard:"
for ds in salmon storycloze; do
  for f in "${LOG_DIR}/${ds}/"shard*.log; do
    line=$(grep -E "Summary —|Done\.$" "${f}" | tail -1)
    echo "  ${ds}/$(basename "${f}"): ${line}"
  done
done

if [[ ${FAIL} -ne 0 ]]; then
  echo "[run_eval_tg_gen] some shards failed; inspect logs under ${LOG_DIR}" >&2
  exit 1
fi
echo "[run_eval_tg_gen] all shards complete"
