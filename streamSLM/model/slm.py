"""StreamSLM model: shared LLM backbone + multi-head + symmetric-delay objective.

Audio/text fusion follows TASTE-SpokenLM
(``taste_speech/modules_taste/bridge.py::WeightedSumFusion``):

    audio_emb_n = sum_r acoustic_emb_r(q_n_r) + dur_in(log1p(d_n))    # in audio_dim
    audio_emb_shifted = front-pad-with(pad_audio_unit_embed)          # delay shift
    fused_n = w_text * text_emb_n + w_audio * Linear_audio2hidden(audio_emb_shifted_n)

with ``[w_audio, w_text] = softmax([-2, 2])`` so audio contributes ~1.8% at
init and the LLM essentially sees the pretrained text embedding. The two
softmax logits are learnable so the model can open the audio path during
training.

Heads:
    subword     : reuse backbone.lm_head            (CE on next-token shift)
    acoustic    : Linear(d_hidden, R*K)             (CE per-codebook, summed)
    duration    : Linear(d_hidden, 1)               (L1 / MSE on log-frames)

Symmetric (MusicGen-style) next-step delay convention. Per-doc the collator
emits T_i = L_i + delay tokens: [w_0..w_{L_i-1}, EOS, PAD, ..., PAD]. With
delay D and hidden h_n at position n,

    Input at position n  : (w_n, q_{n-D}, d_{n-D})                       (PAD at first D positions)
    Target at position n : (w_{n+1}, q_{n-D+1}, d_{n-D+1})               (HF causal shift)

Valid target masks (pre-baked into the batch by the collator/packer):
    text_label_mask : True at [0, L_i-1]   -> position L_i-1 predicts stored EOS.
    aco_label_mask  : True at [D-1, L_i+D-2] -> predicts q_0..q_{L_i-1}, d_0..d_{L_i-1}.

aco/duration targets are right-shifted by D-1 via F.pad(..., (D-1, 0)) so the
target at position n is q_{n-D+1}/d_{n-D+1}.

INCOMPATIBILITY NOTE — old asymmetric-delay checkpoints (target = q_n at
position n; inference backfill at slot n rather than n-D+1) are NOT
compatible. The state-dict shapes match, but the loaded model would predict
acoustic codes that are misaligned by D-1 frames relative to the saved heads.
Retrain from scratch under the new convention.

Text loss with frozen reference LLM (TASTE stage-2):
    text_loss = 0.9 * KL(ref_softmax || spoken_log_softmax) + 0.1 * CE
where the KL is reduction='batchmean' over non-IGNORE next-token positions,
mirroring ``modeling_taste._calcuate_loss_text_kl`` (lines 796-809) and the
blend at line 903. Without a ref_model, text_loss collapses to plain CE.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from streamSLM.config import ModelConfig, TokenizerConfig
from streamSLM.model.predictors import (
    DepthTransformerPredictor,
    MossLocalDepthPredictor,
    MultiHeadPredictor,
)


# --------------------------------------------------------------------------- #
# Liger-Kernel patch. Fuses RMSNorm, RoPE, SwiGLU (and optionally lm_head+CE)
# in the Llama/Qwen backbone. Module-level monkey-patch, idempotent, applied
# before from_pretrained so the loaded modules are the fused variants.
#
# fused_linear_cross_entropy is left OFF: this codebase reads `out.logits`
# directly and computes text CE outside the backbone (alongside acoustic +
# duration losses), so the fused-linear-CE path can't fire and we want to
# avoid touching the patched forward.
#
# Toggle with STREAMSLM_LIGER=0 to disable for an A/B comparison.
# --------------------------------------------------------------------------- #
_LIGER_APPLIED: set[str] = set()


def _maybe_apply_liger(backbone_name: str) -> None:
    if os.environ.get("STREAMSLM_LIGER", "1") == "0":
        return
    name = (backbone_name or "").lower()
    if "llama" in name:
        family = "llama"
    elif "qwen3" in name:
        family = "qwen3"
    elif "qwen2" in name or "qwen" in name:
        family = "qwen2"
    else:
        return  # unsupported family; leave the model untouched
    if family in _LIGER_APPLIED:
        return
    try:
        if family == "llama":
            from liger_kernel.transformers import apply_liger_kernel_to_llama as fn
        elif family == "qwen3":
            from liger_kernel.transformers import apply_liger_kernel_to_qwen3 as fn
        else:
            from liger_kernel.transformers import apply_liger_kernel_to_qwen2 as fn
    except ImportError:
        print(f"[liger] not installed; skipping {family} patch", flush=True)
        return
    fn(rope=True, rms_norm=True, swiglu=True,
       cross_entropy=False, fused_linear_cross_entropy=False)
    _LIGER_APPLIED.add(family)
    print(f"[liger] applied {family} kernel patches "
          f"(rope+rms_norm+swiglu)", flush=True)


# --------------------------------------------------------------------------- #
# TASTE-style fusion, ported verbatim from
# TASTE-SpokenLM/taste_speech/modules_taste/bridge.py::WeightedSumFusion.
# --------------------------------------------------------------------------- #
class WeightedSumFusion(nn.Module):
    """Softmax-weighted sum of (Linear-projected audio) and text embeddings.

    With weight_init_type='zero_audio' the two learnable logits are
    initialized to [-2, 2], so softmax weights are ~[0.018, 0.982]: audio
    contributes ~1.8% at init, text ~98.2%. The LLM therefore receives
    essentially the pretrained text embedding at step 0.
    """

    def __init__(
        self,
        weight_init_type: str = "zero_audio",
        audio_dim: int = 256,
        llm_dim: int = 2048,
        fusion_mode: str = "gated_softmax",
    ):
        super().__init__()
        self.linear = nn.Linear(audio_dim, llm_dim)
        self.fusion_mode = fusion_mode
        if fusion_mode == "gated_softmax":
            if weight_init_type == "balance":
                self.weights = nn.Parameter(torch.tensor([1.0, 1.0]), requires_grad=True)
            elif weight_init_type == "zero_audio":
                self.weights = nn.Parameter(torch.tensor([-2.0, 2.0]), requires_grad=True)
            else:
                raise ValueError(f"unknown weight_init_type: {weight_init_type}")
        elif fusion_mode == "sum":
            # Moshi-style: text + Linear(audio) with no learnable gate.
            self.register_parameter("weights", None)
        else:
            raise ValueError(f"unknown fusion_mode: {fusion_mode}")

    def forward(
        self,
        text_embeds: torch.Tensor,    # (B, T, llm_dim)
        audio_embeds: torch.Tensor,   # (B, T, audio_dim)
    ) -> torch.Tensor:
        if self.fusion_mode == "sum":
            return text_embeds + self.linear(audio_embeds)
        weights = F.softmax(self.weights, dim=0).view(2, 1, 1, 1)
        inputs = torch.stack([self.linear(audio_embeds), text_embeds], dim=0)
        return (weights * inputs).sum(dim=0)


# --------------------------------------------------------------------------- #
# TASTE-style weighted-layer mix, ported from
# TASTE-SpokenLM/taste_speech/modules_taste/bridge.py::WeightedLayerExtract.
# Linear projection is intentionally omitted: the downstream acoustic predictor
# (MultiHeadPredictor / DepthTransformerPredictor) already provides it.
# --------------------------------------------------------------------------- #
class WeightedLayerMix(nn.Module):
    """Softmax-weighted sum over all transformer hidden states.

    HF returns `output_hidden_states=True` as a tuple of length
    `num_hidden_layers + 1`: index 0 is the embedding output, indices 1..L are
    the post-block outputs, and the last entry has the final RMSNorm applied.
    This module learns a per-layer scalar logit and softmax-mixes them.

    Init: all logits = 0 -> uniform 1/(L+1) mix at step 0.
    """

    def __init__(self, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, hidden_states_tuple) -> torch.Tensor:
        # hidden_states_tuple: tuple of (B, T, D) tensors, length = num_layers
        stacked = torch.stack(list(hidden_states_tuple), dim=0)   # (L, B, T, D)
        weights = F.softmax(self.weights.float(), dim=0).to(stacked.dtype)
        return (weights.view(self.num_layers, 1, 1, 1) * stacked).sum(dim=0)


class StreamSLM(nn.Module):
    """Streaming SLM over per-subword (w_n, q_n, d_n) units.

    Audio (RVQ codes + duration) is fused into the LLM input via TASTE-style
    ``WeightedSumFusion`` so the pretrained text embedding dominates at init.
    """

    def __init__(self, model_cfg: ModelConfig, tok_cfg: TokenizerConfig,
                 skip_backbone_weights: bool = False):
        super().__init__()
        self.cfg = model_cfg
        self.tok_cfg = tok_cfg
        self.delay = max(int(model_cfg.delay), 1)

        # Override the attention backend with STREAMSLM_ATTN_IMPL (e.g. "sdpa",
        # "eager") for benchmarks or non-FA2 environments. FA2 requires the
        # weights themselves to be bf16/fp16 — autocasting inputs is not
        # enough — so when FA2 is requested, load the backbone in bf16. SDPA
        # works fine with fp32 master weights + bf16 autocast.
        attn_impl = os.environ.get("STREAMSLM_ATTN_IMPL", "flash_attention_2")
        backbone_dtype = (
            torch.bfloat16 if attn_impl == "flash_attention_2" else torch.float32
        )
        _maybe_apply_liger(model_cfg.backbone)
        if skip_backbone_weights:
            # Build empty scaffold from config — no NFS read of the 2.4 GB
            # safetensors. Caller must restore weights from a ckpt before use.
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
        # Optional HF gradient checkpointing on the backbone. use_reentrant=False
        # is DDP-safe and works with use_cache=False (already set in forward()).
        # Trades ~all backbone activation memory for one extra forward per layer.
        if os.environ.get("STREAMSLM_GRADIENT_CHECKPOINTING", "0") == "1":
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.d_hidden = self.backbone.config.hidden_size
        self.vocab_size = self.backbone.config.vocab_size
        self.num_hidden_layers = self.backbone.config.num_hidden_layers

        # Hidden-state source for the acoustic + duration heads (TASTE-style
        # learnable mix vs. final-layer-only). Text head always reads
        # out.logits, which is lm_head(post-norm last layer).
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
        # TASTE's audio_tower_config.audio_embed_dim is 256.
        self.audio_dim = int(getattr(model_cfg, "audio_dim", 256))

        # Per-codebook embeddings, summed -> audio_dim.
        # Acts as the RVQ analog of TASTE's
        # vq_module.get_output_from_indices(): we only stored the discrete
        # indices, not the codebook vectors, so the per-codebook embedding
        # tables learn to produce a single audio_dim-d vector per word.
        self.acoustic_emb = nn.ModuleList(
            [nn.Embedding(self.K, self.audio_dim) for _ in range(self.R)]
        )

        # Duration scalar -> audio_dim (added into the same audio stream).
        # TASTE has a single audio channel into the LLM; we fold duration in
        # there so the WeightedSumFusion still sees one (text, audio) pair.
        # log(1+frames) keeps the scalar in a sane range.
        self.dur_in = nn.Sequential(
            nn.Linear(1, self.audio_dim),
            nn.GELU(),
            nn.Linear(self.audio_dim, self.audio_dim, bias=False),
        )

        # TASTE-style audio pad: a single zero-initialized vector at audio_dim,
        # repeated `delay` times when front-padding the audio stream so the
        # model never sees the audio of the subword it is asked to predict.
        self.pad_audio_unit_embed = nn.Parameter(
            torch.zeros(self.audio_dim, dtype=torch.float32)
        )

        # TASTE WeightedSumFusion over (text_emb, audio_emb).
        # weight_init_type='zero_audio' matches the TASTE configs/model/ours.json
        # default for spoken_lm_config.in_llm_module='weighted_sum'.
        self.fusion_mode = getattr(model_cfg, "fusion_mode", "gated_softmax")
        self.fuse_for_bridge_in_llm = WeightedSumFusion(
            weight_init_type="zero_audio",
            audio_dim=self.audio_dim,
            llm_dim=self.d_hidden,
            fusion_mode=self.fusion_mode,
        )

        # Match TASTE's audio_embed magnitude. In TASTE, audio_embeds come from
        # vq_module.get_output_from_indices() — a single small (~std 0.02)
        # vector per word. Here audio_emb is the SUM of R independent RVQ
        # codebook embeddings, so we init each table at std 0.02/sqrt(R) to
        # keep the sum at ~0.02 (text-embedding scale). Without this scaling
        # the WeightedSumFusion's 1.8% audio weight is still big enough to
        # corrupt the pretrained text input.
        scale_cfg = str(getattr(model_cfg, "audio_emb_init_scale", "auto"))
        if scale_cfg == "auto":
            per_book_std = 0.02 / (self.R ** 0.5)
        else:
            per_book_std = float(scale_cfg)
        for emb in self.acoustic_emb:
            nn.init.normal_(emb.weight, mean=0.0, std=per_book_std)
        # Zero-init the duration MLP's last linear so the duration channel
        # contributes zero at init (TASTE has no duration stream into the
        # LLM; this preserves the "pretrained text dominates" property).
        nn.init.zeros_(self.dur_in[-1].weight)

        # Duration coupling knobs (Exp A / Exp B).
        self.duration_head_input = getattr(model_cfg, "duration_head_input", "llm")
        self.duration_loss_type = getattr(model_cfg, "duration_loss_type", "regression")
        self.duration_num_buckets = int(getattr(model_cfg, "duration_num_buckets", 256))
        self.feed_duration_to_depformer = bool(
            getattr(model_cfg, "feed_duration_to_depformer", False)
        )

        # Output heads
        # Acoustic prediction: discrete (RVQ codes via multihead/depth) OR
        # continuous (regress to a teacher pre-quantizer feature, e.g. 256-dim).
        self.acoustic_target = getattr(model_cfg, "acoustic_target", "rvq")
        self.acoustic_feat_dim = int(getattr(model_cfg, "acoustic_feat_dim", 256))
        self.acoustic_feat_loss = getattr(model_cfg, "acoustic_feat_loss", "l1")

        # Acoustic predictor (multihead or depth_transformer); both expose
        # forward(h, q_codes_targets) -> (B, T, R, K) and sample(h_last, ...).
        self.predictor_type = getattr(model_cfg, "acoustic_predictor_type", "multihead")
        if self.acoustic_target == "continuous":
            # Continuous regression head replaces the discrete predictor.
            # The depth-transformer's whole point is autoregressive coupling
            # along R discrete codebooks; with a single D-dim continuous
            # output it's irrelevant, so we use a flat Linear from h.
            if self.duration_head_input == "depformer_last":
                raise ValueError(
                    "duration_head_input='depformer_last' is incompatible "
                    "with acoustic_target='continuous' (no depth transformer)."
                )
            if self.feed_duration_to_depformer:
                raise ValueError(
                    "feed_duration_to_depformer=True is incompatible with "
                    "acoustic_target='continuous' (no depth transformer)."
                )
            self.acoustic_predictor = None
            self.acoustic_feat_head = nn.Linear(self.d_hidden, self.acoustic_feat_dim)
        elif self.predictor_type == "multihead":
            if self.duration_head_input == "depformer_last":
                raise ValueError(
                    "duration_head_input='depformer_last' requires "
                    "acoustic_predictor_type='depth_transformer'"
                )
            if self.feed_duration_to_depformer:
                raise ValueError(
                    "feed_duration_to_depformer=True requires "
                    "acoustic_predictor_type='depth_transformer'"
                )
            self.acoustic_predictor = MultiHeadPredictor(self.d_hidden, self.R, self.K)
        elif self.predictor_type == "depth_transformer":
            self.acoustic_predictor = DepthTransformerPredictor(
                d_hidden=self.d_hidden,
                R=self.R,
                K=self.K,
                depformer_dim=getattr(model_cfg, "depformer_dim", 128),
                depformer_num_heads=getattr(model_cfg, "depformer_num_heads", 4),
                depformer_num_layers=getattr(model_cfg, "depformer_num_layers", 2),
                depformer_dim_feedforward=getattr(model_cfg, "depformer_dim_feedforward", None),
                depformer_multi_linear=getattr(model_cfg, "depformer_multi_linear", False),
                depformer_weights_per_step=getattr(model_cfg, "depformer_weights_per_step", False),
                depformer_norm=getattr(model_cfg, "depformer_norm", False),
                feed_duration=self.feed_duration_to_depformer,
                duration_num_buckets=self.duration_num_buckets,
                prev_depth_output_dropout=float(
                    getattr(model_cfg, "prev_depth_output_dropout", 0.0)
                ),
            )
        elif self.predictor_type == "moss_local_depth":
            if self.duration_head_input == "depformer_last":
                raise ValueError(
                    "duration_head_input='depformer_last' is not supported "
                    "by acoustic_predictor_type='moss_local_depth'."
                )
            if self.feed_duration_to_depformer:
                raise ValueError(
                    "feed_duration_to_depformer=True is not supported by "
                    "acoustic_predictor_type='moss_local_depth'."
                )
            self.acoustic_predictor = MossLocalDepthPredictor(
                d_hidden=self.d_hidden,
                R=self.R,
                K=self.K,
                local_hidden=int(getattr(model_cfg, "moss_local_hidden", 1536)),
                local_num_heads=int(getattr(model_cfg, "moss_local_num_heads", 16)),
                local_num_layers=int(getattr(model_cfg, "moss_local_num_layers", 4)),
                local_ffn=int(getattr(model_cfg, "moss_local_ffn", 8960)),
                bridge_ffn=int(getattr(model_cfg, "moss_local_bridge_ffn", 2048)),
            )
        else:
            raise ValueError(f"unknown acoustic_predictor_type: {self.predictor_type}")

        # Duration head. Input dim depends on whether we read the LLM hidden
        # state (legacy) or the depth-transformer's final-position output u^{R-1}.
        # Output dim depends on regression (1) vs classification (N buckets).
        if self.duration_head_input == "llm":
            dur_in_dim = self.d_hidden
        elif self.duration_head_input == "depformer_last":
            dur_in_dim = int(getattr(model_cfg, "depformer_dim", 128))
        else:
            raise ValueError(f"unknown duration_head_input: {self.duration_head_input}")
        if self.duration_loss_type == "regression":
            dur_out_dim = 1
        elif self.duration_loss_type == "classification":
            dur_out_dim = self.duration_num_buckets
        else:
            raise ValueError(f"unknown duration_loss_type: {self.duration_loss_type}")
        self.duration_head = nn.Linear(dur_in_dim, dur_out_dim)

        # Loss modules - mirror TASTE-SpokenLM/taste_speech/modeling_taste.py:616-618.
        # ignore_index here is the sentinel we use in compute_loss (-100), not -1.
        self.ce_loss_module = nn.CrossEntropyLoss(reduction="mean", ignore_index=-100)
        self.kl_loss_module = nn.KLDivLoss(reduction="batchmean", log_target=False)

        # Optional per-codebook weights for residual-VQ training loss. Held as
        # a buffer so it moves with .to(device). Validation always uses uniform
        # mean -> `loss_acoustic` is comparable across configs.
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

        # Legacy checkpoints saved the flat head as `acoustic_head.{weight,bias}`.
        # The new multihead predictor wraps that head under
        # `acoustic_predictor.head.{...}`, so on load we transparently rename
        # the keys. Only kicks in for the multihead variant (depth_transformer
        # checkpoints will only ever exist post-refactor).
        self._register_load_state_dict_pre_hook(self._legacy_acoustic_head_load_hook)

    @staticmethod
    def _legacy_acoustic_head_load_hook(state_dict, prefix, *_):
        for src, dst in (
            ("acoustic_head.weight", "acoustic_predictor.head.weight"),
            ("acoustic_head.bias",   "acoustic_predictor.head.bias"),
        ):
            full_src = prefix + src
            if full_src in state_dict:
                state_dict[prefix + dst] = state_dict.pop(full_src)

    # ------------------------------------------------------------------ #
    # Input embedding fusion
    # ------------------------------------------------------------------ #
    def _embed_audio(
        self,
        q_codes: torch.Tensor,
        duration_frames: torch.Tensor,
    ) -> torch.Tensor:
        """q_codes (B, T, R), duration_frames (B, T) -> audio_emb (B, T, audio_dim)."""
        e = sum(self.acoustic_emb[r](q_codes[..., r]) for r in range(self.R))
        x = torch.log1p(duration_frames.to(torch.float32)).unsqueeze(-1)  # (B, T, 1)
        return e + self.dur_in(x)

    def _delay_shift_audio(
        self,
        audio_e: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Front-pad the audio stream by `delay` using pad_audio_unit_embed.

        Under the symmetric convention, each doc of original length L_i is
        extended to T_i = L_i + delay, with stored q at slots [0, L_i-1]
        and zeros at [L_i, T_i-1] (trailing EOS+PAD positions). Front-padding
        by `delay` therefore produces audio_input = [PAD]*delay ++ q[:L_i] ++
        [emb(0)]*(delay-1), where only the first L_i + delay positions are
        consumed (the very last `delay` audio inputs would be embeddings of
        zero, but their corresponding target positions are masked out).

        Without position_ids: legacy global shift along the time axis, correct
        when each batch row contains exactly one doc.

        With position_ids (sequence-packed mode): per-doc shift. Positions
        with ``position_ids[t] < delay`` (the first `delay` tokens of every
        doc) are replaced by ``pad_audio_unit_embed``; remaining positions
        gather ``audio_e[t - delay]``, which is guaranteed to belong to the
        same doc as long as every doc length (extended) >= delay+1.
        """
        B, T, D = audio_e.shape
        if position_ids is None:
            # Front-pad with min(delay, T) PAD slots, then take up to T-delay
            # real slots; the final slice ensures output length is exactly T
            # even when T < delay (e.g., during the first inference iteration
            # of a text-only prompt with delay >= 2).
            n_pad = min(self.delay, T)
            pref = self.pad_audio_unit_embed.view(1, 1, D).expand(B, n_pad, D)
            n_real = T - n_pad
            if n_real > 0:
                out = torch.cat([pref, audio_e[:, :n_real]], dim=1)
            else:
                out = pref
            return out
        ar = torch.arange(T, device=audio_e.device).unsqueeze(0).expand(B, T)
        src = (ar - self.delay).clamp(min=0).unsqueeze(-1).expand(-1, -1, D)
        shifted = torch.gather(audio_e, 1, src)
        use_pad = (position_ids < self.delay).unsqueeze(-1)        # (B, T, 1)
        pad = self.pad_audio_unit_embed.to(audio_e.dtype).view(1, 1, D)
        return torch.where(use_pad, pad, shifted)

    def fuse_inputs(
        self,
        subword_ids: torch.Tensor,
        q_codes: torch.Tensor,
        duration_frames: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns (B, T, d_hidden) fused input embeddings (TASTE WeightedSumFusion)."""
        text_e = self.backbone.get_input_embeddings()(subword_ids)
        audio_e = self._delay_shift_audio(
            self._embed_audio(q_codes, duration_frames),
            position_ids=position_ids,
        )
        return self.fuse_for_bridge_in_llm(text_e, audio_e)

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
        """Returns logits and the regressed duration scalar.

        Shapes:
            subword_ids     (B, T)        long
            q_codes         (B, T, R)     long
            duration_frames (B, T)        long
            attention_mask  (B, T)        bool/0-1
            position_ids    (B, T)        long, optional. When provided, takes
                                          over from attention_mask: we pass
                                          ``attention_mask=None`` to the
                                          backbone so its create_causal_mask
                                          detects the packed-sequence format
                                          via position_ids resets and routes
                                          FA2 to the varlen path. Required to
                                          be non-None for sequence packing.
        """
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
            # Packed path: rely on position_ids resets to delineate docs.
            # attention_mask must be None for create_causal_mask to detect
            # packing (transformers/masking_utils.py:700).
            bb_kwargs["position_ids"] = position_ids
        else:
            bb_kwargs["attention_mask"] = attention_mask.to(torch.long)
        out = self.backbone(**bb_kwargs)
        # Text head reads from out.logits (= lm_head over post-norm last
        # layer); acoustic + duration heads read either the same final layer
        # or a learnable softmax mix over all (num_hidden_layers + 1) layers.
        text_logits = out.logits                        # (B, T, V)
        if self.acoustic_h_mix is not None:
            h = self.acoustic_h_mix(out.hidden_states)  # (B, T, d)
        else:
            h = out.hidden_states[-1]                   # (B, T, d)

        # Build GT duration buckets when injecting into the depth transformer.
        # bucket = clamp(frames, 0, N-1). Inference would substitute the
        # predicted-bucket sample; training uses GT (teacher-forcing).
        dur_buckets = None
        if self.feed_duration_to_depformer:
            dur_buckets = duration_frames.clamp(
                min=0, max=self.duration_num_buckets - 1
            ).long()

        # Continuous-acoustic mode: regress to a teacher pre-quantizer feature.
        # The discrete predictor is absent; duration head reads h directly.
        if self.acoustic_target == "continuous":
            aco_feat_pred = self.acoustic_feat_head(h)   # (B, T, acoustic_feat_dim)
            dur_pred = self.duration_head(h)             # (B, T, dur_out_dim)
            if self.duration_loss_type == "regression":
                dur_pred = dur_pred.squeeze(-1)
            return {
                "text_logits":         text_logits,
                "acoustic_feat_pred":  aco_feat_pred,
                "duration_pred":       dur_pred,
                "hidden":              h,
            }

        # Teacher-forced acoustic logits over (B, T, R, K). For multihead this
        # ignores q_codes; for depth_transformer it conditions each codebook
        # k on the ground-truth q^{<k} via Moshi-style depth attention.
        if self.duration_head_input == "depformer_last":
            aco_logits, u_last = self.acoustic_predictor(
                h, q_codes,
                dur_buckets=dur_buckets,
                return_depformer_last=True,
            )
            dur_pred = self.duration_head(u_last)        # (B, T, dur_out_dim)
        else:
            if self.predictor_type == "depth_transformer":
                aco_logits = self.acoustic_predictor(h, q_codes, dur_buckets=dur_buckets)
            else:
                aco_logits = self.acoustic_predictor(h, q_codes)
            dur_pred = self.duration_head(h)             # (B, T, dur_out_dim)

        if self.duration_loss_type == "regression":
            dur_pred = dur_pred.squeeze(-1)              # (B, T)
        # Classification: keep (B, T, N) for CE in compute_loss.

        return {
            "text_logits":     text_logits,
            "acoustic_logits": aco_logits,
            "duration_pred":   dur_pred,
            "hidden":          h,
        }

    @torch.no_grad()
    def dur_pred_to_bucket(self, dur_pred: torch.Tensor) -> torch.Tensor:
        """Round a regression duration prediction to an embedding bucket.

        dur_pred is log1p(frames) under regression. Inference path (not
        training): undo log1p, round to integer frames, clamp into the
        embedding range. Training uses GT integer frames directly so this
        helper is unused inside forward().
        """
        if not self.feed_duration_to_depformer:
            raise RuntimeError(
                "dur_pred_to_bucket is only meaningful when "
                "feed_duration_to_depformer=True"
            )
        frames = torch.expm1(dur_pred).round()
        return frames.clamp(min=0, max=self.duration_num_buckets - 1).long()

    def _compute_text_kl(
        self,
        ref_model: nn.Module,
        subword_ids: torch.Tensor,
        text_logits: torch.Tensor,
        shift_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """KL(ref_softmax || spoken_log_softmax) on valid next-token positions.

        Mirrors TASTE-SpokenLM/taste_speech/modeling_taste.py:_calcuate_loss_text_kl
        (lines 796-809) under streamSLM's HF causal-LM shift convention. ref_model
        runs under torch.no_grad() and its logits are detached, so no gradient
        flows back into ref_model parameters.

        When ``position_ids`` is provided, ref_model also runs in packed mode
        (attention_mask=None) so each doc's softmax is computed in isolation.

        Implementation note: at large T (e.g. packed sequences of 8K tokens
        with V=128K llama vocab) the masked gather
        ``log_p_flat[mask_flat]`` materializes a fresh (~T*V*2B) tensor which
        OOMs the 48GB pool. We chunk along T so peak memory scales with
        T_chunk*V instead of T*V, and explicitly drop ref_logits as soon as
        the target distribution is computed.
        """
        with torch.no_grad():
            ref_model.eval()
            ref_kwargs = dict(input_ids=subword_ids, use_cache=False, return_dict=True)
            if position_ids is not None:
                ref_kwargs["position_ids"] = position_ids
            ref_out = ref_model(**ref_kwargs)
            ref_logits = ref_out.logits.detach()
            # HF causal shift: position k predicts subword_{k+1}.
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
        """Per-spec loss: L = w_text * L_w + w_aco * mean_r L_q_r + w_dur * L_d.

        With ref_model provided, L_w = 0.9 * KL(ref || spoken) + 0.1 * CE,
        matching TASTE stage-2 (modeling_taste.py:903).

        Symmetric-delay masks are sourced from the collator/packer:
            text_label_mask : True over prediction positions [0, L_i-1] per
                doc. Applied AFTER the HF causal shift, so position L_i-1's
                target (stored EOS at slot L_i) is supervised, but position
                L_i (whose target would be PAD) is IGNORE'd.
            aco_label_mask  : True over [D-1, L_i+D-2]. Acoustic and duration
                targets are right-shifted by D-1 so the target at position n
                is q_{n-D+1} / d_{n-D+1}.

        Fallback (back-compat): if the caller doesn't supply the masks, we
        reconstruct them from attention_mask + position_ids, treating the
        whole row as one doc whose EOS sits one past the last attended slot.
        Training/eval callers should pass the collator's masks directly.
        """
        IGNORE = -100
        B, T = subword_ids.shape
        mask = attention_mask.to(torch.bool)

        # Mask reconstruction fallback. attention_mask covers real + EOS slots
        # (i.e. T_i = L_i + 1 in unpacked mode); the trailing delay-1 PAD slots
        # are zero. Recover L_i = sum(mask) - 1 for each row.
        if text_label_mask is None:
            if position_ids is None:
                lengths_eos = mask.sum(dim=1)                  # (B,) = L_i + 1
                pos = torch.arange(T, device=mask.device).unsqueeze(0).expand(B, T)
                text_label_mask = pos < (lengths_eos - 1).unsqueeze(1)
            else:
                # Packed mode w/o pre-baked masks: every position whose
                # attention is True except the last attended position per
                # doc (the EOS slot). Detect EOS slot by next-position
                # boundary (position_ids[..., t+1] == 0) or doc tail.
                boundary_next = F.pad(
                    position_ids[..., 1:] == 0, (0, 1), value=True,
                )
                # Position is EOS iff (mask True) AND (boundary_next or
                # next position is masked out / new doc start).
                eos_slot = mask & boundary_next
                text_label_mask = mask & ~eos_slot
        if aco_label_mask is None:
            if position_ids is None:
                lengths_eos = mask.sum(dim=1)                  # L_i + 1
                pos = torch.arange(T, device=mask.device).unsqueeze(0).expand(B, T)
                aco_label_mask = (
                    (pos >= (self.delay - 1)) &
                    (pos < (lengths_eos + self.delay - 2).unsqueeze(1) + 1)
                )
                # cap to the valid range [delay-1, L_i+delay-2]
                aco_label_mask = aco_label_mask & (
                    pos <= (lengths_eos + self.delay - 2).unsqueeze(1)
                )
            else:
                # Packed fallback: mark all positions p where p >= delay-1
                # within the doc and the target slot p-delay+1 is still a real
                # position (i.e. position_ids[p] < doc_L + delay - 1).
                # The collator/packer always supplies aco_label_mask, so this
                # path is informational only — fail loud if it's reached
                # without explicit masks.
                raise RuntimeError(
                    "packed-mode compute_loss requires aco_label_mask from "
                    "the collator (post-symmetric-delay refactor)."
                )

        text_label_mask = text_label_mask.to(torch.bool)
        aco_label_mask = aco_label_mask.to(torch.bool)

        # ---- text CE: HF causal-LM shift: logits[..., :-1] vs labels[..., 1:].
        # text_label_mask is over *prediction* positions, so we apply it as the
        # gate on shift_labels (the labels for positions 0..T-2, length T-1).
        labels = subword_ids.clone()
        shift_logits = outputs["text_logits"][..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_mask = text_label_mask[..., : T - 1]
        shift_labels = shift_labels.masked_fill(~shift_mask, IGNORE)
        text_ce = self.ce_loss_module(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
        )

        # ---- text KL distillation from frozen ref_model (TASTE stage-2).
        # Blend ratio is configurable via cfg.text_kl_weight; setting it to
        # 0.0 drops KL entirely (text_loss == CE) and lets the trainer skip
        # the ref_model load altogether.
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

        # ---- acoustic + duration target shift (symmetric delay).
        # Target at position n: q_{n-D+1} (or d_{n-D+1}). Achieved by
        # right-shifting along T by D-1; F.pad prepends D-1 zeros (any value
        # works since aco_label_mask is False at the first D-1 positions).
        shift_pad = self.delay - 1
        if shift_pad > 0:
            q_targets = F.pad(q_codes, (0, 0, shift_pad, 0))[:, :T, :]
            dur_targets_full = F.pad(duration_frames, (shift_pad, 0))[:, :T]
        else:
            q_targets = q_codes
            dur_targets_full = duration_frames

        if self.acoustic_target == "continuous":
            # Regress h_n -> teacher pre-quantizer feature (B, T, D_feat), shifted.
            if pre_quant_feat is None:
                raise RuntimeError(
                    "acoustic_target='continuous' but pre_quant_feat was not "
                    "passed to compute_loss; ensure the dataset returns it."
                )
            if shift_pad > 0:
                feat_target_full = F.pad(
                    pre_quant_feat, (0, 0, shift_pad, 0)
                )[:, :T, :]
            else:
                feat_target_full = pre_quant_feat
            feat_pred = outputs["acoustic_feat_pred"]               # (B, T, D_feat)
            feat_target = feat_target_full.to(feat_pred.dtype)
            denom = aco_label_mask.sum().clamp_min(1).to(feat_pred.dtype)
            m = aco_label_mask.to(feat_pred.dtype)
            if self.acoustic_feat_loss == "mse":
                err = (feat_pred - feat_target).pow(2).mean(dim=-1)
                aco_loss = (err * m).sum() / denom
            elif self.acoustic_feat_loss == "cosine":
                cos = F.cosine_similarity(feat_pred, feat_target, dim=-1)
                err = 1.0 - cos
                aco_loss = (err * m).sum() / denom
            elif self.acoustic_feat_loss == "mse_l1":
                mse_err = (feat_pred - feat_target).pow(2).mean(dim=-1)
                l1_err  = (feat_pred - feat_target).abs().mean(dim=-1)
                mse_loss = (mse_err * m).sum() / denom
                l1_loss  = (l1_err  * m).sum() / denom
                aco_loss = 0.5 * mse_loss + 0.5 * l1_loss
            else:  # "l1" (default)
                err = (feat_pred - feat_target).abs().mean(dim=-1)
                aco_loss = (err * m).sum() / denom
            aco_loss_uniform = aco_loss
        else:
            aco_logits = outputs["acoustic_logits"]                  # (B, T, R, K)
            aco_targets = q_targets[..., : self.R].clone()           # (B, T, R)
            aco_targets = aco_targets.masked_fill(~aco_label_mask.unsqueeze(-1), IGNORE)
            # Per-codebook CE so we can (a) optionally apply per-codebook training
            # weights and (b) always log a uniform-mean `loss_acoustic` for fair
            # cross-config comparison even when training-time weighting differs.
            ce_per_pos = F.cross_entropy(
                aco_logits.reshape(-1, self.K),
                aco_targets.reshape(-1),
                ignore_index=IGNORE,
                reduction="none",
            ).view(B, T, self.R)
            valid_mask_r = (aco_targets != IGNORE).to(ce_per_pos.dtype)  # (B, T, R)
            valid_count_r = valid_mask_r.sum(dim=(0, 1)).clamp_min(1.0)  # (R,)
            per_r_loss = ce_per_pos.sum(dim=(0, 1)) / valid_count_r       # (R,)
            aco_loss_uniform = per_r_loss.mean()
            if self.training and self.rvq_loss_weights.numel() == self.R:
                w = self.rvq_loss_weights.to(per_r_loss.dtype)
                aco_loss = (per_r_loss * w).sum() / w.sum()
            else:
                aco_loss = aco_loss_uniform

        # ---- duration: regression on log(1+frames) OR classification over
        # `duration_num_buckets` linear frame buckets. Same mask as acoustic;
        # target is the D-1-right-shifted duration_frames.
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
            # Always the uniform-mean per-codebook CE so logs are comparable
            # across configs that may apply different training-time RVQ weights.
            "loss_acoustic": aco_loss_uniform.detach(),
            "loss_duration": dur_loss.detach(),
        }


def load_lm_tokenizer(tok_cfg: TokenizerConfig):
    """Mirror inference_core.py: pad_token <- eos_token."""
    name = "Qwen/Qwen3-1.7B" if tok_cfg.text_tokenizer == "qwen3" else "meta-llama/Llama-3.2-1B"
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok
