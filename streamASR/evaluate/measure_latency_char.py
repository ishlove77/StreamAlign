#!/usr/bin/env python3
"""Latency eval wrapper that patches StreamingCharModel.encode_chunk to handle
encoders whose forward_streaming returns just `x` (no hidden states).

The prod model.py:392 unconditionally unpacks `x, hidden = enc.forward_streaming(...)`,
which fails (`ValueError: not enough values to unpack`) when the encoder returns
1 value. This wrapper patches encode_chunk to fall back to no-hidden-states.
"""
import importlib.util
import os
import sys

import torch

_PROD_ROOT = "/home/streamalign/streamASR"
if _PROD_ROOT not in sys.path:
    sys.path.insert(0, _PROD_ROOT)

from models.model import StreamingCharModel


@torch.no_grad()
def _encode_chunk_patched(self, context, chunk, chunk_len=None):
    if chunk_len is None:
        chunk_len = torch.ones((chunk.size(0),))
    chunk = chunk.float()
    chunk, chunk_len = chunk.to(self.device), chunk_len.to(self.device)
    x = self.hparams["fea_streaming_extractor"](
        chunk, context=context.fea_extractor_context, lengths=chunk_len
    )
    out = self.mods["enc"].forward_streaming(x, context.encoder_context)
    if isinstance(out, tuple) and len(out) == 2:
        x, hidden = out
    else:
        x, hidden = out, None
    x = self.mods["proj_enc"](x)
    if hidden is not None and self.output_hidden_states:
        hidden = torch.cat(hidden, dim=0)
        return x, hidden
    return x


StreamingCharModel.encode_chunk = _encode_chunk_patched


_PROD = os.path.join(_PROD_ROOT, "evaluate", "latency", "measure_latency.py")
_spec = importlib.util.spec_from_file_location("_prod_measure_latency", _PROD)
_prod = importlib.util.module_from_spec(_spec)
sys.modules["_prod_measure_latency"] = _prod
_spec.loader.exec_module(_prod)


if __name__ == "__main__":
    _prod.main()
