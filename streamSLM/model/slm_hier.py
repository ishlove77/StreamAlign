"""StreamSLMHier: Moshi-style hierarchical step-internal depth-AR variant.

Differs from :class:`streamSLM.model.slm.StreamSLM` only in what happens to
``h_n`` after the LLM backbone:

  1. ``lm_head(h_n) -> w_{n+1}``                                  (unchanged)
  2. Depth transformer AR over the slots
        [text_in, dur_in, q^0_in, q^1_in, ..., q^{R-1}_in]   (duration_first)
        [text_in, q^0_in, q^1_in, ..., q^{R-1}_in, dur_in]   (duration_last)
     where slot 0 (text) is fed the dep_bos and its output is discarded,
     slot 1's input is the (shared Llama) ``Emb_text(w_{n+1})`` projected to
     depformer_dim, and each later slot's input is the previous AR step's
     GT (training) or sample (inference).

The audio-input fusion, delay shift, weighted-layer mix, KL distillation,
sequence packing, and per-codebook acoustic targets follow the symmetric-delay
convention shared with :class:`StreamSLM`: targets at position n are
(w_{n+1}, q_{n-D+1}, d_{n-D+1}), so the depth-AR teacher-forcing inputs
(``q_codes`` and ``dur_input``) must also be right-shifted by D-1 — they
represent the GT q^{<k} that the depth transformer should attend to when
producing q^k at the same n. Forward() applies the shift before calling
``acoustic_predictor`` so the predictor sees aligned (target, teacher-input)
pairs.

INCOMPATIBILITY NOTE — checkpoints trained under the old asymmetric-delay
formulation are NOT compatible. Retrain from scratch under the new convention.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM

from streamSLM.config import ModelConfig, TokenizerConfig
from streamSLM.model.predictors import HierDepthTransformerPredictor
from streamSLM.model.slm import (
    WeightedSumFusion,
    WeightedLayerMix,
    _maybe_apply_liger,
)


class StreamSLMHier(nn.Module):
    """Hierarchical step-internal AR variant of StreamSLM.

    Same external forward / compute_loss signature as ``StreamSLM`` so the
    training loop and validation paths work without branching beyond the
    model construction step.
    """

    is_hier_ar = True

    def __init__(
        self,
        model_cfg: ModelConfig,
        tok_cfg: TokenizerConfig,
        skip_backbone_weights: bool = False,
    ):
        super().__init__()
        self.cfg = model_cfg
        self.tok_cfg = tok_cfg
        self.delay = max(int(model_cfg.delay), 1)

        # Backbone load matches StreamSLM.__init__ verbatim.
        attn_impl = os.environ.get("STREAMSLM_ATTN_IMPL", "flash_attention_2")
        backbone_dtype = (
            torch.bfloat16 if attn_impl == "flash_attention_2" else torch.float32
        )
        _maybe_apply_liger(model_cfg.backbone)
        if skip_backbone_weights:
            cfg = AutoConfig.from_pretrained(
                model_cfg.backbone, attn_implementation=attn_impl,
            )
            cfg.torch_dtype = backbone_dtype
            self.backbone = AutoModelForCausalLM.from_config(
                cfg, torch_dtype=backbone_dtype,
            )
        else:
            self.backbone = AutoModelForCausalLM.from_pretrained(
                model_cfg.backbone,
                torch_dtype=backbone_dtype,
                attn_implementation=attn_impl,
            )
        if os.environ.get("STREAMSLM_GRADIENT_CHECKPOINTING", "0") == "1":
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.d_hidden = self.backbone.config.hidden_size
        self.vocab_size = self.backbone.config.vocab_size
        self.num_hidden_layers = self.backbone.config.num_hidden_layers

        # Optional weighted mix over hidden states for the acoustic head.
        # Text head still reads out.logits (= lm_head over the final post-norm
        # layer), matching the legacy model.
        self.acoustic_layer_mix = getattr(model_cfg, "acoustic_layer_mix", "last")
        if self.acoustic_layer_mix == "weighted":
            self.acoustic_h_mix = WeightedLayerMix(self.num_hidden_layers + 1)
        elif self.acoustic_layer_mix == "last":
            self.acoustic_h_mix = None
        else:
            raise ValueError(
                f"unknown acoustic_layer_mix: {self.acoustic_layer_mix}"
            )

        self.R = tok_cfg.num_quantizers
        self.K = tok_cfg.codebook_size
        self.audio_dim = int(getattr(model_cfg, "audio_dim", 256))

        # --- audio-input fusion: identical to StreamSLM. ---
        self.acoustic_emb = nn.ModuleList(
            [nn.Embedding(self.K, self.audio_dim) for _ in range(self.R)]
        )
        self.dur_in = nn.Sequential(
            nn.Linear(1, self.audio_dim),
            nn.GELU(),
            nn.Linear(self.audio_dim, self.audio_dim, bias=False),
        )
        self.pad_audio_unit_embed = nn.Parameter(
            torch.zeros(self.audio_dim, dtype=torch.float32)
        )
        self.fusion_mode = getattr(model_cfg, "fusion_mode", "gated_softmax")
        self.fuse_for_bridge_in_llm = WeightedSumFusion(
            weight_init_type="zero_audio",
            audio_dim=self.audio_dim,
            llm_dim=self.d_hidden,
            fusion_mode=self.fusion_mode,
        )

        scale_cfg = str(getattr(model_cfg, "audio_emb_init_scale", "auto"))
        if scale_cfg == "auto":
            per_book_std = 0.02 / (self.R ** 0.5)
        else:
            per_book_std = float(scale_cfg)
        for emb in self.acoustic_emb:
            nn.init.normal_(emb.weight, mean=0.0, std=per_book_std)
        nn.init.zeros_(self.dur_in[-1].weight)

        # Duration coupling knobs. `feed_duration_to_depformer` is obsolete in
        # this class — duration is part of the depth-AR, so its conditioning
        # is implicit in `hier_ar_order` and we ignore the legacy flag.
        self.duration_loss_type = getattr(model_cfg, "duration_loss_type", "regression")
        self.duration_num_buckets = int(getattr(model_cfg, "duration_num_buckets", 256))
        self.hier_ar_order = getattr(model_cfg, "hier_ar_order", "duration_last")

        # The depth predictor owns BOTH the per-codebook acoustic heads AND
        # the duration head; lm_head stays on the backbone for w_{n+1}.
        self.acoustic_predictor = HierDepthTransformerPredictor(
            d_hidden=self.d_hidden,
            d_text_emb=self.d_hidden,           # Llama input emb dim == hidden_size
            R=self.R,
            K=self.K,
            depformer_dim=getattr(model_cfg, "depformer_dim", 128),
            depformer_num_heads=getattr(model_cfg, "depformer_num_heads", 4),
            depformer_num_layers=getattr(model_cfg, "depformer_num_layers", 2),
            depformer_dim_feedforward=getattr(model_cfg, "depformer_dim_feedforward", None),
            depformer_multi_linear=getattr(model_cfg, "depformer_multi_linear", False),
            depformer_weights_per_step=getattr(model_cfg, "depformer_weights_per_step", False),
            depformer_norm=getattr(model_cfg, "depformer_norm", False),
            duration_loss_type=self.duration_loss_type,
            duration_num_buckets=self.duration_num_buckets,
            hier_ar_order=self.hier_ar_order,
            prev_depth_output_dropout=float(
                getattr(model_cfg, "prev_depth_output_dropout", 0.0)
            ),
        )

        self.ce_loss_module = nn.CrossEntropyLoss(reduction="mean", ignore_index=-100)
        self.kl_loss_module = nn.KLDivLoss(reduction="batchmean", log_target=False)

        rvq_w_cfg = getattr(model_cfg, "rvq_loss_weights", None)
        if rvq_w_cfg is None:
            self.register_buffer("rvq_loss_weights", torch.empty(0), persistent=False)
        else:
            if len(rvq_w_cfg) != self.R:
                raise ValueError(
                    f"rvq_loss_weights length {len(rvq_w_cfg)} != R={self.R}"
                )
            self.register_buffer(
                "rvq_loss_weights",
                torch.tensor(list(rvq_w_cfg), dtype=torch.float32),
                persistent=False,
            )

    # ------------------------------------------------------------------ #
    # Input embedding fusion (identical to StreamSLM)
    # ------------------------------------------------------------------ #
    def _embed_audio(
        self,
        q_codes: torch.Tensor,
        duration_frames: torch.Tensor,
    ) -> torch.Tensor:
        e = sum(self.acoustic_emb[r](q_codes[..., r]) for r in range(self.R))
        x = torch.log1p(duration_frames.to(torch.float32)).unsqueeze(-1)
        return e + self.dur_in(x)

    def _delay_shift_audio(
        self,
        audio_e: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D = audio_e.shape
        if position_ids is None:
            pref = self.pad_audio_unit_embed.view(1, 1, D).expand(B, self.delay, D)
            return torch.cat([pref, audio_e[:, : T - self.delay]], dim=1)
        ar = torch.arange(T, device=audio_e.device).unsqueeze(0).expand(B, T)
        src = (ar - self.delay).clamp(min=0).unsqueeze(-1).expand(-1, -1, D)
        shifted = torch.gather(audio_e, 1, src)
        use_pad = (position_ids < self.delay).unsqueeze(-1)
        pad = self.pad_audio_unit_embed.to(audio_e.dtype).view(1, 1, D)
        return torch.where(use_pad, pad, shifted)

    def fuse_inputs(
        self,
        subword_ids: torch.Tensor,
        q_codes: torch.Tensor,
        duration_frames: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_e = self.backbone.get_input_embeddings()(subword_ids)
        audio_e = self._delay_shift_audio(
            self._embed_audio(q_codes, duration_frames),
            position_ids=position_ids,
        )
        return self.fuse_for_bridge_in_llm(text_e, audio_e)

    # ------------------------------------------------------------------ #
    # w_{n+1} embedding for depth-AR teacher forcing. Re-uses the Llama
    # input embedding so we do not introduce a separate (V, depformer_dim)
    # table — the predictor's text_to_depformer linearly projects this into
    # depformer space.
    # ------------------------------------------------------------------ #
    def _w_next_embeddings(
        self,
        subword_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Shift left by one along T; the last column has no GT next-token, so
        # zero-fill it. In packed sequences a doc-boundary position would
        # otherwise leak the next doc's first token into the depth-AR; zero
        # it out so the model never sees cross-doc conditioning.
        B, T = subword_ids.shape
        w_next_ids = F.pad(subword_ids[:, 1:], (0, 1), value=0)
        if position_ids is not None:
            next_boundary = F.pad(
                position_ids[:, 1:] == 0, (0, 1), value=True,
            )                                       # (B, T) bool
            w_next_ids = w_next_ids.masked_fill(next_boundary, 0)
        return self.backbone.get_input_embeddings()(w_next_ids)

    # ------------------------------------------------------------------ #
    # Forward + loss
    # ------------------------------------------------------------------ #
    def forward(
        self,
        subword_ids: torch.Tensor,
        q_codes: torch.Tensor,
        duration_frames: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        inputs_embeds = self.fuse_inputs(
            subword_ids, q_codes, duration_frames, position_ids=position_ids,
        )
        bb_kwargs = dict(
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        if position_ids is not None:
            bb_kwargs["position_ids"] = position_ids
        else:
            bb_kwargs["attention_mask"] = attention_mask.to(torch.long)
        out = self.backbone(**bb_kwargs)

        text_logits = out.logits
        if self.acoustic_h_mix is not None:
            h = self.acoustic_h_mix(out.hidden_states)
        else:
            h = out.hidden_states[-1]

        # GT w_{n+1} embedding for teacher-forced depth-AR.
        w_next_emb = self._w_next_embeddings(subword_ids, position_ids=position_ids)

        # Depth-AR teacher-forcing: at position n the predictor must produce
        # (q^0_{n-D+1}..q^{R-1}_{n-D+1}, d_{n-D+1}). The depth attention reads
        # prior-slot GT, so q_codes / dur_input fed here must also be
        # right-shifted by D-1 along the time axis (same shift as compute_loss
        # applies to the targets).
        T_full = subword_ids.shape[1]
        shift_pad = self.delay - 1
        if shift_pad > 0:
            q_codes_aligned = F.pad(q_codes, (0, 0, shift_pad, 0))[:, :T_full, :]
            duration_aligned = F.pad(duration_frames, (shift_pad, 0))[:, :T_full]
        else:
            q_codes_aligned = q_codes
            duration_aligned = duration_frames

        if self.duration_loss_type == "classification":
            dur_input = duration_aligned.clamp(
                min=0, max=self.duration_num_buckets - 1,
            ).long()
        else:
            dur_input = torch.log1p(duration_aligned.to(torch.float32))

        aco_logits, dur_pred = self.acoustic_predictor(
            h, w_next_emb, q_codes_aligned, dur_input=dur_input,
        )

        if self.duration_loss_type == "regression":
            dur_pred = dur_pred.squeeze(-1)              # (B, T)

        return {
            "text_logits":     text_logits,
            "acoustic_logits": aco_logits,
            "duration_pred":   dur_pred,
            "hidden":          h,
        }

    # KL helper: same as StreamSLM (re-implemented inline to avoid a cross-
    # class import; chunked to keep peak memory under control at long T).
    def _compute_text_kl(
        self,
        ref_model: nn.Module,
        subword_ids: torch.Tensor,
        text_logits: torch.Tensor,
        shift_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            ref_model.eval()
            ref_kwargs = dict(input_ids=subword_ids, use_cache=False, return_dict=True)
            if position_ids is not None:
                ref_kwargs["position_ids"] = position_ids
            ref_out = ref_model(**ref_kwargs)
            ref_logits = ref_out.logits.detach()
            target = F.softmax(ref_logits[..., :-1, :], dim=-1)
            del ref_logits, ref_out

        spoken_shift = text_logits[..., :-1, :]
        log_p = F.log_softmax(spoken_shift, dim=-1)
        B, Tm1, V = log_p.shape
        n_valid = shift_mask.sum().clamp_min(1).to(log_p.dtype)
        T_chunk = 1024 if Tm1 > 1024 else Tm1
        kl_sum = log_p.new_zeros(())
        for t in range(0, Tm1, T_chunk):
            e = min(t + T_chunk, Tm1)
            sm = shift_mask[..., t:e].reshape(-1)
            if not sm.any():
                continue
            lp = log_p[..., t:e, :].reshape(-1, V)[sm]
            tg = target[..., t:e, :].reshape(-1, V)[sm]
            kl_sum = kl_sum + F.kl_div(lp, tg, reduction="sum", log_target=False)
        return kl_sum / n_valid

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        subword_ids: torch.Tensor,
        q_codes: torch.Tensor,
        duration_frames: torch.Tensor,
        attention_mask: torch.Tensor,
        ref_model: Optional[nn.Module] = None,
        pre_quant_feat: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        text_label_mask: Optional[torch.Tensor] = None,
        aco_label_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del pre_quant_feat  # hier model is discrete-only; argument kept for API parity
        IGNORE = -100
        B, T = subword_ids.shape
        mask = attention_mask.to(torch.bool)

        # Mask fallback (back-compat for callers that still pass attention only).
        if text_label_mask is None:
            if position_ids is None:
                lengths_eos = mask.sum(dim=1)
                pos = torch.arange(T, device=mask.device).unsqueeze(0).expand(B, T)
                text_label_mask = pos < (lengths_eos - 1).unsqueeze(1)
            else:
                boundary_next = F.pad(
                    position_ids[..., 1:] == 0, (0, 1), value=True,
                )
                eos_slot = mask & boundary_next
                text_label_mask = mask & ~eos_slot
        if aco_label_mask is None:
            if position_ids is None:
                lengths_eos = mask.sum(dim=1)
                pos = torch.arange(T, device=mask.device).unsqueeze(0).expand(B, T)
                aco_label_mask = (pos >= (self.delay - 1)) & (
                    pos <= (lengths_eos + self.delay - 2).unsqueeze(1)
                )
            else:
                raise RuntimeError(
                    "packed-mode compute_loss requires aco_label_mask from "
                    "the collator (post-symmetric-delay refactor)."
                )

        text_label_mask = text_label_mask.to(torch.bool)
        aco_label_mask = aco_label_mask.to(torch.bool)

        # ---- text CE: HF causal-LM shift gated by prediction-position mask. ----
        shift_logits = outputs["text_logits"][..., :-1, :].contiguous()
        shift_labels = subword_ids[..., 1:].contiguous().clone()
        shift_mask = text_label_mask[..., : T - 1]
        shift_labels = shift_labels.masked_fill(~shift_mask, IGNORE)
        text_ce = self.ce_loss_module(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
        )

        kl_w = float(getattr(self.cfg, "text_kl_weight", 0.9))
        if ref_model is not None and kl_w > 0.0:
            text_kl = self._compute_text_kl(
                ref_model, subword_ids, outputs["text_logits"], shift_mask,
                position_ids=position_ids,
            )
            text_loss = kl_w * text_kl + (1.0 - kl_w) * text_ce
        else:
            text_kl = torch.zeros((), device=text_ce.device, dtype=text_ce.dtype)
            text_loss = text_ce

        # ---- acoustic + duration target shift (symmetric delay). ----
        shift_pad = self.delay - 1
        if shift_pad > 0:
            q_targets = F.pad(q_codes, (0, 0, shift_pad, 0))[:, :T, :]
            dur_targets_full = F.pad(duration_frames, (shift_pad, 0))[:, :T]
        else:
            q_targets = q_codes
            dur_targets_full = duration_frames

        aco_logits = outputs["acoustic_logits"]
        aco_targets = q_targets[..., : self.R].clone()
        aco_targets = aco_targets.masked_fill(~aco_label_mask.unsqueeze(-1), IGNORE)
        ce_per_pos = F.cross_entropy(
            aco_logits.reshape(-1, self.K),
            aco_targets.reshape(-1),
            ignore_index=IGNORE,
            reduction="none",
        ).view(B, T, self.R)
        valid_count_r = (aco_targets != IGNORE).to(ce_per_pos.dtype).sum(
            dim=(0, 1)
        ).clamp_min(1.0)
        per_r_loss = ce_per_pos.sum(dim=(0, 1)) / valid_count_r
        aco_loss_uniform = per_r_loss.mean()
        if self.training and self.rvq_loss_weights.numel() == self.R:
            w = self.rvq_loss_weights.to(per_r_loss.dtype)
            aco_loss = (per_r_loss * w).sum() / w.sum()
        else:
            aco_loss = aco_loss_uniform

        # ---- duration: shifted by D-1 like acoustic. ----
        dur_pred = outputs["duration_pred"]
        valid = aco_label_mask
        if self.duration_loss_type == "classification":
            dur_target = dur_targets_full.clamp(
                min=0, max=self.duration_num_buckets - 1,
            ).long()
            dur_target = dur_target.masked_fill(~valid, IGNORE)
            dur_loss = self.ce_loss_module(
                dur_pred.reshape(-1, self.duration_num_buckets),
                dur_target.reshape(-1),
            )
        else:
            dur_target = torch.log1p(dur_targets_full.to(torch.float32))
            if self.cfg.duration_loss == "mse":
                d = (dur_pred - dur_target).pow(2)
            else:
                d = (dur_pred - dur_target).abs()
            dur_loss = (d * valid.to(d.dtype)).sum() / valid.sum().clamp_min(1).to(d.dtype)

        total = (
            self.cfg.loss_w_text * text_loss
            + self.cfg.loss_w_acoustic * aco_loss
            + self.cfg.loss_w_duration * dur_loss
        )
        return {
            "loss":          total,
            "loss_text":     text_loss.detach(),
            "loss_text_ce":  text_ce.detach(),
            "loss_text_kl":  text_kl.detach(),
            "loss_acoustic": aco_loss_uniform.detach(),
            "loss_duration": dur_loss.detach(),
        }

    # ------------------------------------------------------------------ #
    # Inference helper for AR generation.
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample_acoustic_duration(
        self,
        h_last: torch.Tensor,        # (B, d_hidden)
        w_next: torch.Tensor,        # (B,) long  -- already-sampled next subword
        temperature_aco: float,
        top_k_aco: int,
        top_p_aco: float,
        temperature_dur: float = 0.0,
        top_k_dur: int = 0,
        top_p_dur: float = 1.0,
    ):
        """Step-internal AR sampling of (q_n, d_n) given (h_n, w_{n+1}).

        Returns ``(q_codes (B, R) long, d_frames (B,) long)``. ``d_frames`` is
        already in encoder-frame units (regression: round(expm1) clamp>=1;
        classification: bucket clamp [0, num_buckets-1]).
        """
        w_next_emb = self.backbone.get_input_embeddings()(w_next.view(-1))
        return self.acoustic_predictor.sample(
            h_last, w_next_emb,
            temperature_aco=temperature_aco,
            top_k_aco=top_k_aco,
            top_p_aco=top_p_aco,
            temperature_dur=temperature_dur,
            top_k_dur=top_k_dur,
            top_p_dur=top_p_dur,
        )
