#!/usr/bin/env python3
"""RVQ + cosine LR variant (no distill).

Plain RVQ training (using the temp models.model_tokenizer) with cosine
LR decay applied via env-var-driven AdamW patch.

Env vars:
    COSINE_TOTAL_STEPS  total cosine length in optimizer steps (required)
    COSINE_MIN_LR       minimum LR at end of schedule (default 0)
"""
import math
import os
import sys

_STREAMASR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STREAMASR_ROOT)

import torch
from models.model_tokenizer import Data2VecSemanticAcousticModel
import train.train_tokenizer_base as _orig

_orig.Data2VecSemanticAcousticModel = Data2VecSemanticAcousticModel


_TOTAL_STEPS = int(os.environ.get("COSINE_TOTAL_STEPS", "0"))
_MIN_LR = float(os.environ.get("COSINE_MIN_LR", "0"))


def _patch_adamw_with_cosine():
    if _TOTAL_STEPS <= 0:
        print("[cosine] COSINE_TOTAL_STEPS not set; running without cosine.", flush=True)
        return
    _OrigAdamW = torch.optim.AdamW

    class _CosineAdamW(_OrigAdamW):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._cosine_step = 0
            self._base_lrs = [g["lr"] for g in self.param_groups]
            print(f"[cosine] AdamW patched: total_steps={_TOTAL_STEPS}, "
                  f"min_lr={_MIN_LR}, base_lrs={self._base_lrs}", flush=True)

        def step(self, *a, **kw):
            t = min(self._cosine_step, _TOTAL_STEPS)
            cos = 0.5 * (1.0 + math.cos(math.pi * t / max(1, _TOTAL_STEPS)))
            for g, base_lr in zip(self.param_groups, self._base_lrs):
                g["lr"] = _MIN_LR + (base_lr - _MIN_LR) * cos
            self._cosine_step += 1
            return super().step(*a, **kw)

    torch.optim.AdamW = _CosineAdamW


_patch_adamw_with_cosine()


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    _orig.main()
