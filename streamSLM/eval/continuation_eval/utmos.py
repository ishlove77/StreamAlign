"""UTMOS22-strong scorer.

Loads the same checkpoint VERSA's utterance_metrics/pseudo_mos.py uses
(`ftshijt/SpeechMOS:main`, ``utmos22_strong``), pointing torch.hub at the
shared versa cache so nothing is re-downloaded.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch

from ._audio import load_mono_16k


DEFAULT_HUB_CACHE = Path.home() / "versa" / "versa_cache"


def _load_model(device: str) -> torch.nn.Module:
    cache_dir = Path(os.environ.get("UTMOS_HUB_CACHE", str(DEFAULT_HUB_CACHE)))
    torch.hub.set_dir(str(cache_dir))
    t0 = time.time()
    model = torch.hub.load(
        "ftshijt/SpeechMOS:main",
        "utmos22_strong",
        trust_repo=True,
        source="github",
    )
    print(f"[utmos] loaded in {time.time() - t0:.1f}s (cache={cache_dir})", flush=True)
    return model.to(device).float().eval()


@torch.no_grad()
def score_wavs(
    wav_paths: list[Path],
    device: str | None = None,
) -> dict[str, float]:
    """Score a list of wav files. Returns {wav_path -> UTMOS}.

    Wavs are resampled to 16 kHz mono before scoring (the model's expected SR).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_model(device)
    sr_tensor = torch.tensor([16000], device=device)
    out: dict[str, float] = {}
    for p in wav_paths:
        wav = load_mono_16k(p)
        x = torch.from_numpy(wav).float().unsqueeze(0).to(device)
        score = float(model(x, sr_tensor).item())
        out[str(p)] = score
    return out
