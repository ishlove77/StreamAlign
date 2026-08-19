"""Speaker Embedding Cosine Similarity (SECS).

Uses the same speaker model the previous step100k vocabmask sweep used
(`espnet/voxcelebs12_rawnet3` via ESPnet2 `Speech2Embedding`), L2-normalized
cosine similarity.

Reference / generated audio convention (matches the prior reference script
``scratch/score_secs_step100k_vocabmask_n100.py``):

  reference   = the audio used as the CosyVoice speaker prompt
  generated   = the model's "with prompt" output (prompt + continuation)

For ``speech_cont_from_wavs`` the speaker prompt is the file under
``input_wavs/<stem>.wav``; for that sweep it was the LibriSpeech .flac.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from espnet2.bin.spk_inference import Speech2Embedding

from ._audio import load_mono_16k


def _l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x) + 1e-12)


def _load_model(device: str):
    t0 = time.time()
    model = Speech2Embedding.from_pretrained(
        model_tag="espnet/voxcelebs12_rawnet3",
        device=device,
    )
    print(f"[secs] loaded in {time.time() - t0:.1f}s", flush=True)
    return model


def _embed(model, wav: np.ndarray) -> np.ndarray:
    emb = model(wav).squeeze(0).detach().cpu().numpy()
    return _l2(emb)


@torch.no_grad()
def score_pairs(
    pairs: list[tuple[Path, Path]],
    device: str | None = None,
) -> dict[str, float]:
    """For each (reference, generated) pair, return cosine similarity.

    Returns {generated_path -> SECS}. The reference is the speaker source,
    matched against the model output; a per-reference embedding cache makes
    repeats over the same speaker free.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_model(device)
    ref_cache: dict[str, np.ndarray] = {}
    out: dict[str, float] = {}
    for ref, gen in pairs:
        ref_key = str(ref)
        if ref_key not in ref_cache:
            ref_cache[ref_key] = _embed(model, load_mono_16k(ref))
        gen_emb = _embed(model, load_mono_16k(gen))
        out[str(gen)] = float(np.dot(ref_cache[ref_key], gen_emb))
    return out
