#!/usr/bin/env bash
# Download the StreamAlign weights from the Hugging Face Hub.
#
#   dd3434/Streamalign-R32  R=32 tokenizer stack (RVQ + alignment
#                                      ASR + streaming ASR + boundary clf).
#   dd3434/Streamalign-SLM-R16         speech LM (Llama-3.2-1B backbone) and
#   dd3434/Streamalign-R16             R=16 tokenizer stack, used only by the
#                                      SLM continuation example.
#
# Usage:
#   bash examples/download_weights.sh [DEST]      # default: ./weights
set -euo pipefail

DEST="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/weights}"
mkdir -p "${DEST}"

python - "$DEST" <<'PY'
import sys
from huggingface_hub import snapshot_download

dest = sys.argv[1]
repos = (
    "dd3434/Streamalign-R32",   # tokenizer example (R=32)
    "dd3434/Streamalign-R16",              # SLM example only
    "dd3434/Streamalign-SLM-R16",          # SLM example only
)
for repo in repos:
    name = repo.split("/")[-1]
    try:
        path = snapshot_download(repo, local_dir=f"{dest}/{name}", max_workers=8)
        print(f"{repo} -> {path}")
    except Exception as e:
        print(f"WARN: could not download {repo}: {type(e).__name__} "
              f"(the examples that need it will not run)")
PY

echo
echo "Weights are in ${DEST}:"
echo "  ${DEST}/Streamalign-R32/rvq_tokenizer/final.pt    R=32 RVQ tokenizer (phase-C ep13)"
echo "  ${DEST}/Streamalign-R32/alignment_model/          char-level alignment ASR"
echo "  ${DEST}/Streamalign-R32/streaming_asr/            word-level streaming ASR"
echo "  ${DEST}/Streamalign-R32/boundary_classifier/      proactive boundary classifier"
echo "  ${DEST}/Streamalign-SLM-R16/step_00185000.pt                speech LM (R=16 units)"
