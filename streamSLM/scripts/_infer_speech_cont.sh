#!/usr/bin/env bash
# Speech continuation from the streamalign_slm_r16 checkpoint.
# Discrete-RVQ head (token_type=rvq R=16 C=512), so generate.py emits codes
# directly and StreamAlign rvq variant decodes them. Teacher checkpoint
# below is the same plain-RVQ R16/C512 teacher used during extraction.
#
# Pipeline:
#   .units.pt prompt --> generate.py (RVQ codes) --> .units.pt
#                    --> reconstruct.py (StreamAlign rvq + CosyVoice) --> .wav
#                    --> whisper(base.en) --> ASR transcript (sanity)

set -uo pipefail

REPO_ROOT=${REPO_ROOT:-/home/streamalign}
cd "${REPO_ROOT}"

CKPT_DIR=${REPO_ROOT}/checkpoints/streamSLM/streamalign_slm_r16
SLM_CKPT=${SLM_CKPT:-${CKPT_DIR}/step_00010000.pt}
TEACHER_CKPT=${TEACHER_CKPT:-/home/streamalign/streamASR/checkpoints/streamalign_r16/epoch_22.pt}

# Test-clean prompts for this RVQ cache (32-way).
LIBRI_TC_ROOT=${LIBRI_TC_ROOT:-/home/datasets/LibriSpeech/test-clean}
UNITS_CACHE=${UNITS_CACHE:-${REPO_ROOT}/cache/streamSLM_units_C512_R16_rvq/librispeech/test-clean/test-clean}
MANIFEST_GLOB=${MANIFEST_GLOB:-${REPO_ROOT}/cache/streamSLM_units_C512_R16_rvq/librispeech/test-clean/manifest_shard*_of32.csv}

NUM_UTTS=${NUM_UTTS:-5}
SAMPLE_SEED=${SAMPLE_SEED:-0}
if [[ -z "${PROMPT_UTTS:-}" ]]; then
  PROMPT_UTTS=$(NUM_UTTS=${NUM_UTTS} SAMPLE_SEED=${SAMPLE_SEED} \
                MANIFEST_GLOB="${MANIFEST_GLOB}" python3 - <<'PY'
import glob, os, random
rels = []
for f in sorted(glob.glob(os.environ["MANIFEST_GLOB"])):
    with open(f) as fp:
        next(fp)
        for line in fp:
            rel = line.split(",", 1)[0].strip()
            if rel:
                rels.append(os.path.splitext(os.path.basename(rel))[0])
random.seed(int(os.environ["SAMPLE_SEED"]))
random.shuffle(rels)
print(" ".join(rels[: int(os.environ["NUM_UTTS"]) ]))
PY
)
fi

PROMPT_MAX_SUBWORDS=${PROMPT_MAX_SUBWORDS:-20}

CKPT_TAG=${CKPT_TAG:-rvq_R16}
OUT_DIR=${REPO_ROOT}/out/slm_rvq_${CKPT_TAG}
LOG=${REPO_ROOT}/logs/streamSLM_train/infer_rvq_${CKPT_TAG}.log
mkdir -p "${OUT_DIR}" "$(dirname "${LOG}")"

SR_EXCLUDE=${SR_EXCLUDE:-kandinsky,greco,namjune,matisse}
SR_QOS=${SR_QOS:-q-low}
VRAM=${VRAM:-24}

# RVQ teacher knobs (same as the extraction worker).
export RVQ_R=${RVQ_R:-16}
export RVQ_CODEBOOK_SIZE=${RVQ_CODEBOOK_SIZE:-512}
# Intentionally NOT setting RVQ_CODEBOOK_DIM — defaults to feat_dim=256.

# SDPA on the 24GB pool (Turing/Volta cards crash on FA2 here). Inference only.
export STREAMSLM_ATTN_IMPL=${STREAMSLM_ATTN_IMPL:-sdpa}

log() { echo "[$(date -Iseconds)] $*" | tee -a "${LOG}"; }

log "ckpt: ${SLM_CKPT} ($(ls -la "${SLM_CKPT}" 2>/dev/null | awk '{print $5,$9}'))"
log "teacher: ${TEACHER_CKPT}"
log "prompt source: librispeech test-clean (${UNITS_CACHE})"
log "speaker_ref source: ${LIBRI_TC_ROOT}"
N_UTTS_RUN=$(echo "${PROMPT_UTTS}" | wc -w)
log "prompt utts: ${N_UTTS_RUN} sampled (seed=${SAMPLE_SEED}, num=${NUM_UTTS})"
log "prompt_max_subwords: ${PROMPT_MAX_SUBWORDS}"
log "RVQ_R=${RVQ_R} RVQ_CODEBOOK_SIZE=${RVQ_CODEBOOK_SIZE}"

UTTS_FILE=${OUT_DIR}/_utts.txt
printf '%s\n' ${PROMPT_UTTS} > "${UTTS_FILE}"

ZERO_PROMPT_AUDIO=${ZERO_PROMPT_AUDIO:-0}
ZERO_FLAG=()
if [[ "${ZERO_PROMPT_AUDIO}" == "1" ]]; then
  ZERO_FLAG=(--zero_prompt_audio)
  log "zero_prompt_audio: ON"
fi

ALLOWED_FLAG=()
if [[ -n "${ALLOWED_TEXT_TOKEN_IDS:-}" ]]; then
  ALLOWED_FLAG=(--allowed_text_token_ids "${ALLOWED_TEXT_TOKEN_IDS}")
  log "allowed_text_token_ids: ${ALLOWED_TEXT_TOKEN_IDS}"
fi

REDO_FLAG=()
if [[ -n "${REDO_FINAL_ACO:-}" && "${REDO_FINAL_ACO}" != "0" ]]; then
  REDO_FLAG=(--redo_final_aco "${REDO_FINAL_ACO}")
  log "redo_final_aco: ${REDO_FINAL_ACO}"
fi

sr 1 ${VRAM} --qos=${SR_QOS} --exclude="${SR_EXCLUDE}" \
  ${PYTHON_BIN:-python} -u -m streamSLM.inference.speech_cont_pipeline \
    --checkpoint "${SLM_CKPT}" \
    --teacher_checkpoint "${TEACHER_CKPT}" \
    --utts_csv "${UTTS_FILE}" \
    --units_cache "${UNITS_CACHE}" \
    --speaker_root "${LIBRI_TC_ROOT}" \
    --out_dir "${OUT_DIR}" \
    --prompt_max_subwords "${PROMPT_MAX_SUBWORDS}" \
    --max_new_tokens 20 \
    --temperature_text 0.9 --top_p_text 0.95 \
    --seed 1 \
    --bf16 \
    --variant ${VARIANT:-rvq} \
    "${ZERO_FLAG[@]}" \
    "${ALLOWED_FLAG[@]}" \
    "${REDO_FLAG[@]}" \
    2>&1 | tee -a "${LOG}"

log "DONE"
