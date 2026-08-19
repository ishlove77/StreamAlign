#!/usr/bin/env python3
"""RVQ no-distill trainer with codebook LR multiplier (DAC-style).

- Model: learnable-codebook RVQ (so codebook entries are nn.Parameters).
- AdamW is monkey-patched on first construction: if the parameter list contains
  model parameters, split into two param groups —
      - "codebook" group (matches name segment "_codebook"): lr = base_lr * CODEBOOK_LR_MULT
      - everything else: lr = base_lr

Env vars:
    CODEBOOK_LR_MULT   default 10
    RVQ_R, RVQ_CODEBOOK_SIZE, RVQ_CODEBOOK_DIM (model)
"""
import os
import sys

_STREAMASR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STREAMASR_ROOT)

import torch
from models.model_tokenizer_learnable import Data2VecSemanticAcousticModel as _BaseModel
import train.train_tokenizer_base as _orig


_codebook_split = {"applied": False, "model_ref": None}


class Data2VecSemanticAcousticModel(_BaseModel):
    """Stash a global reference on init so the patched AdamW can introspect."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _codebook_split["model_ref"] = self


_orig.Data2VecSemanticAcousticModel = Data2VecSemanticAcousticModel


_LR_MULT = float(os.environ.get("CODEBOOK_LR_MULT", "10"))


# Monkey-patch AdamW
_OrigAdamW = torch.optim.AdamW


class _CodebookLRAdamW(_OrigAdamW):
    def __init__(self, params, **kwargs):
        params = list(params)
        # Heuristic: if all params are tensors (not param-group dicts), split.
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
                    print(f"[codelr] split AdamW: codebook={len(cb_params)} params @ "
                          f"lr={base_lr*_LR_MULT:.2e}, other={len(other_params)} @ lr={base_lr:.2e}",
                          flush=True)
                    super().__init__(new_groups, **kwargs)
                    _codebook_split["applied"] = True
                    return
                else:
                    print("[codelr] no codebook params found in named_parameters — "
                          "running with single LR", flush=True)
        super().__init__(params, **kwargs)


torch.optim.AdamW = _CodebookLRAdamW


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    _orig.main()
