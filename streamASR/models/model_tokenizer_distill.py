"""RVQ + WavLM distillation.

Subclass of the RVQ model. Adds WavLM teacher and a Linear projection from
the FIRST quantizer's output (``residual_vq.layers[0]``) to WavLM hidden dim.
Distill loss = ``(1 - cos_sim(linear(layer0_q), wavlm_word_emb)).mean()``.

Also overrides ``_rvq_quantize`` to drop the MSE consistency term (we keep
RVQ commitment loss intact). Total loss = loss_u + RVQ_commit + WavLM_distill.

Env vars:
  DISTILL_WAVLM       "1" to enable
  DISTILL_WAVLM_NAME  default "microsoft/wavlm-base-plus"
  DISTILL_WAVLM_DIM   default 768
  DISTILL_WEIGHT      default 1.0 (multiplier on WavLM cos-sim distill ONLY)
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from vector_quantize_pytorch import ResidualVQ

from models.model_tokenizer import Data2VecSemanticAcousticModel as _BaseRVQ


def _wavlm_active() -> bool:
    return os.environ.get("DISTILL_WAVLM", "0") == "1"


class _WavLMTeacher(nn.Module):
    def __init__(self, model_name: str = "microsoft/wavlm-base-plus"):
        super().__init__()
        from transformers import WavLMModel
        self.wavlm = WavLMModel.from_pretrained(model_name)
        for p in self.wavlm.parameters():
            p.requires_grad = False
        self.wavlm.eval()

    @torch.no_grad()
    def forward(self, wavs_16k: torch.Tensor) -> torch.Tensor:
        return self.wavlm(input_values=wavs_16k).last_hidden_state.detach()


class Data2VecSemanticAcousticModel(_BaseRVQ):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace residual_vq so codebook_dim is honored (the base temp RVQ
        # model doesn't read RVQ_CODEBOOK_DIM and falls back to feat_dim).
        feat_dim = getattr(self.residual_vq, "dim", 256)
        codebook_size = int(os.environ.get("RVQ_CODEBOOK_SIZE", "512"))
        codebook_dim = int(os.environ.get("RVQ_CODEBOOK_DIM", str(feat_dim)))
        rvq_r = int(os.environ.get("RVQ_R", str(getattr(self, "rvq_r", 3))))
        commit_weight = float(os.environ.get("COMMIT_WEIGHT", "1.0"))
        if codebook_dim != feat_dim:
            self.residual_vq = ResidualVQ(
                dim=feat_dim,
                num_quantizers=rvq_r,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                kmeans_init=True,
                kmeans_iters=100,
                threshold_ema_dead_code=2,
                commitment_weight=commit_weight,
            )
            print(f"[rvq-distill] rebuilt residual_vq with codebook_dim={codebook_dim} "
                  f"R={rvq_r} C={codebook_size}", flush=True)
        self._cached_wavs = None
        if _wavlm_active():
            wavlm_name = os.environ.get("DISTILL_WAVLM_NAME", "microsoft/wavlm-base-plus")
            wavlm_dim = int(os.environ.get("DISTILL_WAVLM_DIM", "768"))
            self._wavlm_teacher = _WavLMTeacher(wavlm_name)
            rvq_dim = getattr(self.residual_vq, "dim", 256)
            self._distill_proj = nn.Linear(rvq_dim, wavlm_dim, bias=False)
            self._distill_weight = float(os.environ.get("DISTILL_WEIGHT", "1.0"))
            print(f"[rvq-distill] WavLM={wavlm_name} dim={wavlm_dim} weight={self._distill_weight}",
                  flush=True)

    def _rvq_quantize(self, seg_z: torch.Tensor):
        # Same as base but DROP the DISTILL_WEIGHT * MSE consistency term.
        z_q, _, losses = self.residual_vq(seg_z)
        vq_loss = losses if losses.ndim == 0 else losses.sum()
        if getattr(self, "passthrough", False) and self.training:
            alpha = self._current_alpha()
            seg_z_q = z_q + (1.0 - alpha) * (seg_z - z_q).detach()
            self.forward_step_count = getattr(self, "forward_step_count", 0) + 1
        else:
            seg_z_q = z_q
        return seg_z_q, vq_loss

    def forward(self, *args, **kwargs):
        wavs = kwargs.get("waveforms", None)
        if wavs is None and len(args) > 0:
            wavs = args[0]
        self._cached_wavs = wavs
        try:
            return super().forward(*args, **kwargs)
        finally:
            self._cached_wavs = None

    def _segment_pass(self, z_raw, char_alignment, word_alignment, char_data,
                      gt_char_indices, spk_emb):
        out = super()._segment_pass(
            z_raw, char_alignment, word_alignment, char_data, gt_char_indices, spk_emb
        )
        if not _wavlm_active() or not hasattr(self, "_wavlm_teacher"):
            return out
        if self._cached_wavs is None:
            return out
        try:
            seg_z, _ = self._recompute_seg_z(z_raw, word_alignment, char_alignment)
            distill = self._compute_distill_loss(seg_z, word_alignment, self._cached_wavs)
            if distill is not None:
                out["distill_loss"] = distill
        except Exception as e:
            print(f"[rvq-distill] failed: {e}", flush=True)
        return out

    def _recompute_seg_z(self, z_raw, word_alignment, char_alignment):
        agg, info = self._pool_to_word_segments_transformer(
            z_raw, word_alignment, char_alignment
        )
        B, T_words = word_alignment.shape[0], word_alignment.shape[1]
        feat_dim = agg.size(-1) if agg.dim() == 3 else agg.size(1)
        device = agg.device
        seg_z = self._reconstruct_word_features_batch(
            agg, info, B, T_words, feat_dim, device
        )
        return seg_z, info

    def _compute_distill_loss(self, seg_z, word_alignment, waveforms):
        layer0 = self.residual_vq.layers[0]
        out0 = layer0(seg_z)
        # ResidualVQ layer adapter returns (z_q, indices, _latents, commit, codebook)
        if isinstance(out0, (tuple, list)):
            q1 = out0[0]
        else:
            q1 = out0
        q1_proj = self._distill_proj(q1)
        with torch.no_grad():
            wavlm_emb = self._wavlm_teacher(waveforms)
        T_align = word_alignment.shape[-1]
        T_w = wavlm_emb.shape[1]
        T = min(T_align, T_w)
        wavlm_emb = wavlm_emb[:, :T, :]
        align = word_alignment[:, :, :T].float()
        denom = align.sum(dim=-1, keepdim=True).clamp_min(1.0)
        word_target = torch.bmm(align, wavlm_emb) / denom
        cos = F.cosine_similarity(q1_proj, word_target, dim=-1)
        return (1.0 - cos).mean() * self._distill_weight
