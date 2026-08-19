"""Shared per-subword unit type used across extract / data / model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SubwordUnits:
    """One utterance, decomposed into per-subword (w_n, q_n, d_n) tuples.

    Shapes (N = number of subwords in the utterance):
        subword_ids     (N,)     int64    token id in the LLM tokenizer's vocab
        q_codes         (N, R)   int64    RVQ codebook indices
        duration_frames (N,)     int64    encoder-frame count per subword
        pre_quant_feat  (N, D)   bfloat16 OPTIONAL: pre-quantization feature
                                          fed into ResidualVQ. D = teacher's
                                          residual_vq.dim (typically 256). Saved
                                          in bfloat16 to halve disk footprint;
                                          cast to fp32 at consumer time.

    Durations are stored in encoder-frame units (no hardcoded frame rate).
    Convert to seconds at consumer time via the active encoder's hop.

    `pre_quant_feat` is OPTIONAL (None for caches written before its
    introduction) so old `.units.pt` files still load cleanly under the new
    schema.
    """
    subword_ids: torch.Tensor
    q_codes: torch.Tensor
    duration_frames: torch.Tensor
    pre_quant_feat: Optional[torch.Tensor] = None

    def __post_init__(self):
        n = self.subword_ids.shape[0]
        assert self.q_codes.shape[0] == n, "q_codes/subword length mismatch"
        assert self.duration_frames.shape[0] == n, "duration/subword length mismatch"
        if self.pre_quant_feat is not None:
            assert self.pre_quant_feat.shape[0] == n, "pre_quant_feat/subword length mismatch"
            assert self.pre_quant_feat.ndim == 2, "pre_quant_feat must be (N, D)"

    @property
    def num_subwords(self) -> int:
        return int(self.subword_ids.shape[0])

    @property
    def num_quantizers(self) -> int:
        return int(self.q_codes.shape[1]) if self.q_codes.ndim == 2 else 0

    @property
    def feat_dim(self) -> int:
        return int(self.pre_quant_feat.shape[1]) if self.pre_quant_feat is not None else 0

    def save(self, path: str) -> None:
        d = {
            "subword_ids":     self.subword_ids.to(torch.int64).cpu(),
            "q_codes":         self.q_codes.to(torch.int64).cpu(),
            "duration_frames": self.duration_frames.to(torch.int64).cpu(),
        }
        if self.pre_quant_feat is not None:
            # bfloat16 to halve cache size; fp32->bf16 here, consumers cast back.
            d["pre_quant_feat"] = self.pre_quant_feat.detach().to(torch.bfloat16).cpu()
        torch.save(d, path)

    @classmethod
    def load(cls, path: str) -> "SubwordUnits":
        d = torch.load(path, map_location="cpu", weights_only=True)
        return cls(
            subword_ids=d["subword_ids"],
            q_codes=d["q_codes"],
            duration_frames=d["duration_frames"],
            pre_quant_feat=d.get("pre_quant_feat", None),
        )
