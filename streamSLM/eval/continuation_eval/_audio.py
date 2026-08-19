"""Audio I/O helpers shared by the three scorers."""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np


TARGET_SR = 16000


def load_mono_16k(path: str | Path) -> np.ndarray:
    wav, _ = librosa.load(str(path), sr=TARGET_SR, mono=True)
    return wav.astype(np.float32)


def list_with_prompt_wavs(model_dir: Path) -> list[Path]:
    """Return sorted `*_with_prompt.wav` files under a model output dir."""
    return sorted(model_dir.glob("*_with_prompt.wav"))


def stem_from_with_prompt(p: Path) -> str:
    name = p.stem
    assert name.endswith("_with_prompt"), name
    return name[: -len("_with_prompt")]
