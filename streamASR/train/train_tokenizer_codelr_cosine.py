#!/usr/bin/env python3
"""RVQ no-distill trainer with codebook LR multiplier + cosine LR decay.

Combines the codelr split (codebook params @ N× base lr) with a cosine schedule
on the BASE lr; both groups follow the cosine decay (each from its own peak
toward `COSINE_MIN_LR`).

Env vars:
    CODEBOOK_LR_MULT     default 10
    COSINE_TOTAL_STEPS   total cosine length (required)
    COSINE_MIN_LR        floor at end of decay (default 0)
"""
import math
import os
import sys

_STREAMASR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STREAMASR_ROOT)

import torch
from models.model_tokenizer_learnable import Data2VecSemanticAcousticModel as _BaseModel
import train.train_tokenizer_base as _orig


_codebook_split = {"applied": False, "model_ref": None}


class Data2VecSemanticAcousticModel(_BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _codebook_split["model_ref"] = self


_orig.Data2VecSemanticAcousticModel = Data2VecSemanticAcousticModel


_LR_MULT = float(os.environ.get("CODEBOOK_LR_MULT", "10"))
_COSINE_TOTAL_STEPS = int(os.environ.get("COSINE_TOTAL_STEPS", "0"))
_COSINE_MIN_LR = float(os.environ.get("COSINE_MIN_LR", "0"))


_OrigAdamW = torch.optim.AdamW


class _CodelrCosineAdamW(_OrigAdamW):
    def __init__(self, params, **kwargs):
        params = list(params)
        if params and isinstance(params[0], torch.nn.Parameter):
            model = _codebook_split["model_ref"]
            base_lr = kwargs.get("lr", 1e-4)
            if model is not None:
                cb_set = set()
                for n, p in model.named_parameters():
                    if "_codebook" in n and p.requires_grad:
                        cb_set.add(id(p))
                cb_params = [p for p in params if id(p) in cb_set and p.requires_grad]
                other_params = [p for p in params if id(p) not in cb_set and p.requires_grad]
                if cb_params:
                    new_groups = [
                        {"params": other_params, "lr": base_lr},
                        {"params": cb_params, "lr": base_lr * _LR_MULT},
                    ]
                    print(f"[codelr+cosine] split AdamW: codebook={len(cb_params)} @ "
                          f"lr={base_lr*_LR_MULT:.2e}, other={len(other_params)} @ "
                          f"lr={base_lr:.2e}, cosine_total={_COSINE_TOTAL_STEPS}, "
                          f"min={_COSINE_MIN_LR}", flush=True)
                    super().__init__(new_groups, **kwargs)
                    self._cosine_step = 0
                    self._base_lrs = [g["lr"] for g in self.param_groups]
                    _codebook_split["applied"] = True
                    return
        super().__init__(params, **kwargs)
        self._cosine_step = 0
        self._base_lrs = [g["lr"] for g in self.param_groups]
        print(f"[codelr+cosine] no codebook split — single group, "
              f"cosine_total={_COSINE_TOTAL_STEPS}, min={_COSINE_MIN_LR}", flush=True)

    def step(self, *a, **kw):
        if _COSINE_TOTAL_STEPS > 0:
            t = min(self._cosine_step, _COSINE_TOTAL_STEPS)
            cos = 0.5 * (1.0 + math.cos(math.pi * t / max(1, _COSINE_TOTAL_STEPS)))
            for g, base_lr in zip(self.param_groups, self._base_lrs):
                g["lr"] = _COSINE_MIN_LR + (base_lr - _COSINE_MIN_LR) * cos
            self._cosine_step += 1
        return super().step(*a, **kw)


torch.optim.AdamW = _CodelrCosineAdamW


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    _orig.main()
