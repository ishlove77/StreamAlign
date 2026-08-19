#!/usr/bin/env python3
"""RVQ + WavLM-distillation trainer wrapper.

- Patches the model class to RVQ+distill subclass.
- Adds WavLM distill into ``total_loss`` and surfaces it in
  ``losses['loss_vq']`` for TB visibility (alongside the RVQ commit loss).
  We sum the two so Train/Loss_vq_step shows ``commit + distill`` —
  if you'd rather see distill alone, swap the assignment below.

Logging convention used here:
    losses['loss_vq']  = commit + WavLM_distill
    losses['total']    = loss_u + losses['loss_vq']
"""
import os
import sys

import torch

_STREAMASR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _STREAMASR_ROOT not in sys.path:
    sys.path.insert(0, _STREAMASR_ROOT)

from models.model_tokenizer_distill import Data2VecSemanticAcousticModel as _DistillModel
import train.train_tokenizer_base as _orig
import utils.train_utils_cosyvoice as _utils

_orig.Data2VecSemanticAcousticModel = _DistillModel


_orig_compute_losses = _utils.compute_losses
_distill_step_counter = {"n": 0}


def _compute_losses_with_distill(batch, model, model_out, char_alignment,
                                 word_alignment, use_device):
    losses = _orig_compute_losses(
        batch, model, model_out, char_alignment, word_alignment, use_device
    )
    main_out = model_out["forward_output"]
    distill = main_out.get("distill_loss", None)
    if distill is not None:
        losses["total_loss"] = losses["total_loss"] + distill
        losses["loss_vq"] = losses["loss_vq"].detach() + distill.detach()
        losses["loss_distill"] = distill.detach()
        _distill_step_counter["n"] += 1
        if _distill_step_counter["n"] % 50 == 1 and int(os.environ.get("RANK", "0")) == 0:
            print(f"[rvq-distill] step={_distill_step_counter['n']}  "
                  f"loss_u={losses['loss_u'].item():.4f}  "
                  f"loss_distill={distill.item():.4f}  "
                  f"loss_commit={(losses['loss_vq']-distill.detach()).item():.4f}  "
                  f"total={losses['total_loss'].item():.4f}", flush=True)
    else:
        losses["loss_distill"] = torch.tensor(0.0, device=use_device)
    return losses


_utils.compute_losses = _compute_losses_with_distill
_orig.compute_losses = _compute_losses_with_distill


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    _orig.main()
