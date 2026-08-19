"""Tiny wrapper that forces STREAMSLM_ATTN_IMPL=sdpa before invoking
``streamSLM.inference.speech_cont_from_wavs``.

Why this exists: env vars set in `env(1) -> sr -> srun --pty -> nohup`
chains have intermittently failed to propagate, sending the model down
its default flash_attention_2 path. On 24 GB nodes (mixed generations)
that hangs at startup. Setting the env var from inside Python before
importing the model code is the one reliable path.

Usage: same args as ``python -m streamSLM.inference.speech_cont_from_wavs``.
"""
import os
import runpy
import sys
from pathlib import Path

# streamSLM/scripts/examples/_run_speech_cont_sdpa.py -> repo root is ../../..
REPO_ROOT = str(Path(__file__).resolve().parents[3])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ["STREAMSLM_ATTN_IMPL"] = "sdpa"
print(f"[wrap] STREAMSLM_ATTN_IMPL={os.environ['STREAMSLM_ATTN_IMPL']}", flush=True)

sys.argv = ["streamSLM.inference.speech_cont_from_wavs"] + sys.argv[1:]
runpy.run_module("streamSLM.inference.speech_cont_from_wavs", run_name="__main__")
