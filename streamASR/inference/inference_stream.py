#!/usr/bin/env python3
"""Prod-style inference but using the RVQ model.

Imports the production inference module by file path, monkey-patches its
``Data2VecSemanticAcousticModel`` to ``models.model_tokenizer`` (so the
RVQ residual_vq weights actually load and quantize), and runs main().
"""
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_STREAMASR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _STREAMASR_ROOT not in sys.path:
    sys.path.insert(0, _STREAMASR_ROOT)

import torch
import torch.nn as nn

from models.model_tokenizer import Data2VecSemanticAcousticModel as _RVQModelBase


class _RVQModel(_RVQModelBase):
    """Fix the multi-GPU device bug present in the RVQ model: a few attribute
    tensors and submodules don't get migrated by the default ``.to(device)``.
    Override ``.to`` to walk encoder.hparams and pull everything onto the
    target device. Safe to keep on single-GPU too."""

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        device = None
        for a in args:
            if isinstance(a, (str, torch.device)):
                try:
                    device = torch.device(a); break
                except Exception:
                    pass
        if device is None:
            d = kwargs.get("device")
            if d is not None:
                device = torch.device(d)
        if device is None:
            return result
        enc = getattr(self, "encoder", None)
        if enc is not None:
            try:
                enc.device = device
            except Exception:
                pass
            hp = getattr(enc, "hparams", None)
            if isinstance(hp, dict):
                for v in hp.values():
                    if isinstance(v, nn.Module):
                        v.to(device)
                    elif torch.is_tensor(v):
                        try:
                            v.data = v.data.to(device)
                        except Exception:
                            pass
        return result


# Load the production inference module from its file path so we can
# monkey-patch its model class without modifying the original file.
_PROD_INFER = os.path.join(_STREAMASR_ROOT, "inference", "inference_core.py")
_spec = importlib.util.spec_from_file_location("inference_core_prod", _PROD_INFER)
_orig = importlib.util.module_from_spec(_spec)
sys.modules["inference_core_prod"] = _orig
_spec.loader.exec_module(_orig)

_orig.Data2VecSemanticAcousticModel = _RVQModel


if __name__ == "__main__":
    args = _orig.parse_args()
    if args.output_split is None:
        args.output_split = args.split

    if args.world_size > 1:
        local_rank_env = os.environ.get("LOCAL_RANK")
        if local_rank_env is not None:
            _orig.run(int(local_rank_env), args)
        else:
            torch.multiprocessing.spawn(_orig.run, nprocs=args.world_size, args=(args,))
    else:
        _orig.run(0, args)
