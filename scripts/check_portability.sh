#!/usr/bin/env bash
# Static portability gate: bash syntax, python syntax, and a grep asserting
# no internal absolute paths remain in code (docs/ excluded).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
FAIL=0

echo "[1/3] bash -n on all shell scripts"
while IFS= read -r -d '' f; do
  if ! bash -n "$f"; then echo "  SYNTAX FAIL: $f"; FAIL=1; fi
done < <(find . -name '*.sh' -not -path './.git/*' -print0)

echo "[2/3] python compile on all python files"
if ! python3 - <<'PY'
import pathlib, py_compile, sys
fail = 0
for p in pathlib.Path(".").rglob("*.py"):
    if ".git" in p.parts or "third_party" in p.parts or "speechbrain" in p.parts:
        continue
    try:
        py_compile.compile(str(p), doraise=True)
    except py_compile.PyCompileError as e:
        print(f"  COMPILE FAIL: {p}: {e.msg}")
        fail = 1
sys.exit(fail)
PY
then FAIL=1; fi

echo "[3/3] hardcoded internal path gate"
HITS=$(grep -rnE '/home/streamalign|/home/datasets|/home/CosyVoice|/home/miniconda3|/wbl-fast' \
    --include='*.sh' --include='*.py' --include='*.yaml' \
    --exclude-dir=.git --exclude-dir=docs --exclude=check_portability.sh --exclude-dir=third_party --exclude-dir=speechbrain --exclude-dir=_smoke2 . || true)
if [[ -n "${HITS}" ]]; then
  echo "  HARDCODED PATHS REMAIN:"
  echo "${HITS}"
  FAIL=1
fi

if [[ ${FAIL} -ne 0 ]]; then echo "FAILED"; exit 1; fi
echo "OK"
