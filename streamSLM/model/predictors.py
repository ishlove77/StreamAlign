"""Residual-VQ codebook predictors used by StreamSLM.

Three interchangeable predictor heads share the same external API:

    forward(h, q_codes_targets) -> (B, T, R, K) teacher-forced logits
    sample(h_last, temperature, top_k, top_p) -> (B, R) long sample

`MultiHeadPredictor` is the legacy flat head: a single Linear(d_hidden, R*K)
with no inter-codebook coupling, equivalent to the original implementation.

`DepthTransformerPredictor` reuses Moshi's `StreamingTransformer` (see
`~/moshi/moshi/moshi/modules/transformer.py`) to predict codebooks
autoregressively along the R axis: codebook k is conditioned on the LLM
hidden state h plus the embeddings of codebooks 0..k-1. This is the same
"depformer" recipe as Moshi's RQ-Transformer for audio tokens, adapted to
our single-stream setup (the temporal transformer is the LLM backbone, so we
only own the depth transformer here).

`MossLocalDepthPredictor` is a parallel implementation that mirrors the
"local transformer" from MOSS-TTS (MossTTSLocal). It is intentionally a
separate class so the Moshi-based DepthTransformerPredictor stays untouched
for concurrent experiments. Differences vs. the Moshi recipe:
  - Qwen3-style block: pre-RMSNorm, SwiGLU FFN, attention without any
    positional encoding (MOSS's MossTTSAttentionWithoutPositionalEmbedding).
  - SwiGLU MLP bridge global-hidden -> local-hidden, matching MOSS's
    `additional_mlp_ffn_hidden_size` parameter.
  - No optional knobs (multi_linear / weights_per_step / dur_emb): this
    class is tuned to the MOSS Local config and nothing else.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Sampling helper (mirrors generate.py:_top_k_top_p / _sample_categorical).
# Local copy so this module has no upward dependency.
# --------------------------------------------------------------------------- #
def _filter_logits(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    if top_k > 0:
        v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
        logits = torch.where(logits < v[..., -1:], torch.full_like(logits, -float("inf")), logits)
    if top_p < 1.0:
        s, idx = torch.sort(logits, descending=True)
        cum = F.softmax(s, dim=-1).cumsum(dim=-1)
        keep = cum <= top_p
        keep[..., 0] = True
        s = torch.where(keep, s, torch.full_like(s, -float("inf")))
        logits = torch.empty_like(logits).scatter_(-1, idx, s)
    return logits


def _sample(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1)
    z = _filter_logits(logits / temperature, top_k=top_k, top_p=top_p)
    p = F.softmax(z, dim=-1)
    flat = p.reshape(-1, p.size(-1))
    out = torch.multinomial(flat, num_samples=1).squeeze(-1)
    return out.reshape(p.shape[:-1])


# --------------------------------------------------------------------------- #
# Multi-head: legacy flat predictor. Default behavior, kept identical.
# --------------------------------------------------------------------------- #
class MultiHeadPredictor(nn.Module):
    def __init__(self, d_hidden: int, R: int, K: int):
        super().__init__()
        self.R = R
        self.K = K
        self.head = nn.Linear(d_hidden, R * K, bias=False)

    def forward(self, h: torch.Tensor, q_codes: torch.Tensor) -> torch.Tensor:
        # q_codes is unused (no AR coupling along R); kept for API parity.
        del q_codes
        B, T, _ = h.shape
        return self.head(h).view(B, T, self.R, self.K)

    @torch.no_grad()
    def sample(
        self,
        h_last: torch.Tensor,           # (B, d_hidden)
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:
        logits = self.head(h_last).view(h_last.size(0), self.R, self.K)
        return _sample(logits, temperature, top_k, top_p)


# --------------------------------------------------------------------------- #
# Moshi import shim (path-based, lazy).
# --------------------------------------------------------------------------- #
def _import_moshi_streaming_transformer():
    """Import Moshi's StreamingTransformer from a path-based location.

    We intentionally avoid pip-installing moshi (the package brings sounddevice,
    aiohttp, and other server-only deps). The transformer module itself only
    needs torch + einops and is fine to import directly.
    """
    moshi_root = os.environ.get("MOSHI_ROOT", os.path.expanduser("~/moshi/moshi"))
    if moshi_root not in sys.path:
        sys.path.insert(0, moshi_root)
    from moshi.modules.transformer import StreamingTransformer  # noqa: E402
    return StreamingTransformer


# --------------------------------------------------------------------------- #
# Depth-transformer predictor: Moshi-style depformer over the R axis.
# --------------------------------------------------------------------------- #
class DepthTransformerPredictor(nn.Module):
    """Depth-wise autoregressive RVQ predictor.

    For every timestep n the predictor consumes the LLM hidden state h_n and
    produces logits for all R codebooks autoregressively:

        z^k_n = dep_in_k(h_n) + dep_emb_{k-1}(q^{k-1}_n)         for k > 0
        z^0_n = dep_in_0(h_n) + dep_bos                          for k = 0
        [u^0_n, ..., u^{R-1}_n] = StreamingTransformer([z^0_n, ..., z^{R-1}_n])
        logits^k_n = linear_k( norm_k( u^k_n ) )

    The StreamingTransformer is causal along the depth axis, so u^k_n only
    depends on z^{<=k}_n. Training is teacher-forced; inference loops over k.

    Notes vs. Moshi's depformer:
      - Single-stream: we have no second text/audio stream, so depth 0 uses a
        learnable BOS instead of Moshi's `depformer_text_emb`.
      - Temporal transformer = LLM backbone; we don't own it here.
      - We default to shared depth weights (no weights_per_step) so the
        gating='none' Moshi default works; flip via depformer_weights_per_step.
    """

    def __init__(
        self,
        d_hidden: int,
        R: int,
        K: int,
        depformer_dim: int = 128,
        depformer_num_heads: int = 4,
        depformer_num_layers: int = 2,
        depformer_dim_feedforward: Optional[int] = None,
        depformer_multi_linear: bool = False,
        depformer_weights_per_step: bool = False,
        depformer_norm: bool = False,
        feed_duration: bool = False,
        duration_num_buckets: int = 256,
        prev_depth_output_dropout: float = 0.0,
    ):
        super().__init__()
        StreamingTransformer = _import_moshi_streaming_transformer()

        self.R = R
        self.K = K
        self.depformer_dim = depformer_dim
        self.multi_linear = depformer_multi_linear
        self.feed_duration = feed_duration
        self.duration_num_buckets = duration_num_buckets
        # Dropout applied to the previous-codebook embedding fed into the
        # next depth-layer prediction (the AR coupling branch only — NOT to
        # the LLM-hidden projection `dep_in(h)`). At k=0 the input is just
        # the BOS, which has no "previous depth output" to drop.
        self.prev_depth_output_dropout_p = float(prev_depth_output_dropout)
        self.prev_depth_output_dropout = (
            nn.Dropout(p=self.prev_depth_output_dropout_p)
            if self.prev_depth_output_dropout_p > 0.0
            else nn.Identity()
        )

        if depformer_dim_feedforward is None:
            depformer_dim_feedforward = 4 * depformer_dim

        # Project LLM hidden -> depformer dim. One projection per cb if
        # multi_linear, else a single shared one (Moshi convention).
        n_in = R if depformer_multi_linear else 1
        self.dep_in = nn.ModuleList(
            [nn.Linear(d_hidden, depformer_dim, bias=False) for _ in range(n_in)]
        )

        # Per-codebook embeddings for previously-emitted codes (k = 1..R-1).
        # The last codebook is never an input -> R-1 tables.
        self.dep_emb = nn.ModuleList(
            [nn.Embedding(K, depformer_dim) for _ in range(R - 1)]
        )

        # Depth-0 BOS: depthwise input at k=0 is just dep_in_0(h) + dep_bos.
        self.dep_bos = nn.Parameter(torch.zeros(depformer_dim))

        # Depth transformer (Moshi's StreamingTransformer).
        kwargs = dict(
            d_model=depformer_dim,
            num_heads=depformer_num_heads,
            num_layers=depformer_num_layers,
            dim_feedforward=depformer_dim_feedforward,
            causal=True,
            positional_embedding="sin",
            context=None,
        )
        if depformer_weights_per_step:
            kwargs["weights_per_step"] = R
            # weights_per_step in Moshi requires gating != "none".
            kwargs["gating"] = "silu"
        self.depformer = StreamingTransformer(**kwargs)

        # Per-codebook output norm + linear head.
        if depformer_norm:
            self.dep_norms = nn.ModuleList([nn.LayerNorm(depformer_dim) for _ in range(R)])
        else:
            self.dep_norms = nn.ModuleList([nn.Identity() for _ in range(R)])
        self.linears = nn.ModuleList([nn.Linear(depformer_dim, K, bias=False) for _ in range(R)])

        # Optional duration-bucket embedding added to every depth-position
        # input z^k_n. Used by Exp B to make acoustic prediction depend on
        # the predicted/GT duration bucket.
        if feed_duration:
            self.dur_emb = nn.Embedding(duration_num_buckets, depformer_dim)
        else:
            self.dur_emb = None

        # Truncated-normal init (Moshi `_init_layer` style: std = 1/sqrt(in)).
        for m in [*self.dep_in, *self.linears]:
            nn.init.trunc_normal_(m.weight, std=1.0 / (m.in_features ** 0.5))
        for emb in self.dep_emb:
            nn.init.trunc_normal_(emb.weight, std=1.0 / (emb.embedding_dim ** 0.5))
        nn.init.trunc_normal_(self.dep_bos, std=1.0 / (depformer_dim ** 0.5))
        if self.dur_emb is not None:
            nn.init.trunc_normal_(self.dur_emb.weight, std=1.0 / (depformer_dim ** 0.5))

    # ------------------------------------------------------------------ #
    # Build depth inputs: stack of per-codebook (h-projection + token-embed).
    # Output shape: (..., R, depformer_dim).
    # `prev_codes` shape = (..., R-1) holds the AR inputs (q^{k-1} for k>=1).
    # `up_to_k` lets streaming inference build only the prefix it needs.
    # ------------------------------------------------------------------ #
    def _depth_inputs(
        self,
        h: torch.Tensor,                       # (..., d_hidden)
        prev_codes: torch.Tensor,              # (..., R-1)  long
        up_to_k: Optional[int] = None,
        dur_buckets: Optional[torch.Tensor] = None,  # (...,)  long, in [0, N)
    ) -> torch.Tensor:
        K_steps = self.R if up_to_k is None else up_to_k
        if dur_buckets is not None and self.dur_emb is None:
            raise ValueError(
                "dur_buckets passed but feed_duration=False on this predictor"
            )
        dur_e = self.dur_emb(dur_buckets) if (self.dur_emb is not None and dur_buckets is not None) else None
        outs = []
        for k in range(K_steps):
            in_lin = self.dep_in[k] if self.multi_linear else self.dep_in[0]
            transformer_in = in_lin(h)                                # (..., depformer_dim)
            if k == 0:
                token_in = self.dep_bos.expand_as(transformer_in)
            else:
                token_in = self.dep_emb[k - 1](prev_codes[..., k - 1])
                token_in = self.prev_depth_output_dropout(token_in)
            slot = transformer_in + token_in
            if dur_e is not None:
                slot = slot + dur_e
            outs.append(slot)
        return torch.stack(outs, dim=-2)                              # (..., R, depformer_dim)

    # ------------------------------------------------------------------ #
    # Training: teacher-forced over the full (B, T, R, K) tensor.
    # ------------------------------------------------------------------ #
    def forward(
        self,
        h: torch.Tensor,
        q_codes: torch.Tensor,
        dur_buckets: Optional[torch.Tensor] = None,
        return_depformer_last: bool = False,
    ):
        """h (B, T, d_hidden), q_codes (B, T, R) -> logits (B, T, R, K).

        If `dur_buckets` (B, T) is given and the predictor was built with
        `feed_duration=True`, a per-position duration embedding is added to
        every depth-position input z^k_n.

        If `return_depformer_last=True`, also returns u^{R-1} of shape
        (B, T, depformer_dim) for downstream consumers (Exp A).
        """
        B, T, _ = h.shape
        # Teacher-forcing: prev_codes^k = q^{k-1} for k=1..R-1.
        prev = q_codes[..., : self.R - 1]                             # (B, T, R-1)
        z = self._depth_inputs(h, prev, dur_buckets=dur_buckets)      # (B, T, R, depformer_dim)
        z_flat = z.reshape(B * T, self.R, self.depformer_dim)

        u = self.depformer(z_flat)                                    # (B*T, R, depformer_dim)
        u = u.reshape(B, T, self.R, self.depformer_dim)

        all_logits = []
        for k in range(self.R):
            all_logits.append(self.linears[k](self.dep_norms[k](u[..., k, :])))
        logits = torch.stack(all_logits, dim=-2)                      # (B, T, R, K)
        if return_depformer_last:
            return logits, u[..., self.R - 1, :]                      # (B, T, depformer_dim)
        return logits

    # ------------------------------------------------------------------ #
    # Inference: depth-wise AR sampling at a single time step.
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample(
        self,
        h_last: torch.Tensor,           # (B, d_hidden)
        temperature: float,
        top_k: int,
        top_p: float,
        dur_bucket: Optional[torch.Tensor] = None,  # (B,) long if feed_duration
    ) -> torch.Tensor:
        B = h_last.size(0)
        device = h_last.device
        prev = torch.zeros(B, self.R - 1, dtype=torch.long, device=device)
        out_codes = torch.zeros(B, self.R, dtype=torch.long, device=device)
        for k in range(self.R):
            # Build z^{0..k} fresh each step (cheap: R<=32, B<=1 in current
            # generate.py). Run the depth transformer over the prefix and
            # take its kth output.
            z_prefix = self._depth_inputs(
                h_last, prev, up_to_k=k + 1, dur_buckets=dur_bucket,
            )                                                            # (B, k+1, D)
            u = self.depformer(z_prefix)                                # (B, k+1, D)
            logits_k = self.linears[k](self.dep_norms[k](u[:, k]))      # (B, K)
            q_k = _sample(logits_k, temperature, top_k, top_p)          # (B,)
            out_codes[:, k] = q_k
            if k < self.R - 1:
                prev[:, k] = q_k
        return out_codes


# --------------------------------------------------------------------------- #
# MOSS-TTS-Local-style depth predictor.
#
# Self-contained Qwen3-style block (RMSNorm + SwiGLU + attention without
# positional encoding) plus a SwiGLU MLP bridge from the global LLM hidden
# dim to the local hidden dim. Mirrors MossTTSLocalConfig:
#   local_hidden_size = 1536, local_num_layers = 4, local_ffn_hidden_size = 8960,
#   local_num_heads = 16, additional_mlp_ffn_hidden_size = 2048, no PE.
#
# Kept as a separate class so DepthTransformerPredictor (Moshi-based) and any
# experiments referencing it stay untouched.
# --------------------------------------------------------------------------- #
class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = x.to(torch.float32)
        rms = v.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (v * rms).to(x.dtype) * self.weight


class _SwiGLU(nn.Module):
    """Llama/Qwen-style SwiGLU FFN.

    Supports an asymmetric out dim so it doubles as MOSS's "additional MLP"
    bridge (in_dim != out_dim). When `d_out is None`, standard symmetric SwiGLU.
    """
    def __init__(self, d_in: int, d_ffn: int, d_out: Optional[int] = None):
        super().__init__()
        if d_out is None:
            d_out = d_in
        self.gate_proj = nn.Linear(d_in, d_ffn, bias=False)
        self.up_proj = nn.Linear(d_in, d_ffn, bias=False)
        self.down_proj = nn.Linear(d_ffn, d_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _NoPEAttention(nn.Module):
    """Causal multi-head attention with NO positional encoding.

    Mirrors MOSS-TTS's MossTTSAttentionWithoutPositionalEmbedding: plain QKV
    projections, SDPA with is_causal=True, output projection. Depth axis is
    anchored by per-codebook input embeddings, so no rotary/sin/learned PE.
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


class _MossLocalBlock(nn.Module):
    """Qwen3-style transformer block: pre-RMSNorm -> NoPEAttention, pre-RMSNorm -> SwiGLU."""
    def __init__(self, d_model: int, num_heads: int, d_ffn: int):
        super().__init__()
        self.attn_norm = _RMSNorm(d_model)
        self.attn = _NoPEAttention(d_model, num_heads)
        self.ffn_norm = _RMSNorm(d_model)
        self.ffn = _SwiGLU(d_model, d_ffn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MossLocalDepthPredictor(nn.Module):
    """Depth-wise AR RVQ predictor matching MOSS-TTS Local's local transformer.

    For each timestep n it consumes the global LLM hidden h_n (d_hidden) and
    autoregressively predicts logits for all R codebooks. The local transformer
    is causal along the depth (R) axis and uses no positional encoding:

        h_local           = bridge(h_n)                                 ∈ R^d_local
        z^0_n             = h_local + dep_bos
        z^k_n             = h_local + dep_emb_{k-1}(q^{k-1}_n)          for k = 1..R-1
        [u^0_n,..,u^{R-1}_n] = MossLocalStack([z^0_n,..,z^{R-1}_n])
        logits^k_n        = linear_k(u^k_n)

    Inference loops k=0..R-1 sampling q^k_n autoregressively.
    """

    def __init__(
        self,
        d_hidden: int,
        R: int,
        K: int,
        local_hidden: int = 1536,
        local_num_heads: int = 16,
        local_num_layers: int = 4,
        local_ffn: int = 8960,
        bridge_ffn: int = 2048,
    ):
        super().__init__()
        self.R = R
        self.K = K
        self.local_hidden = local_hidden

        # SwiGLU MLP bridge from global LLM hidden -> local hidden. Matches
        # MOSS's additional_mlp_ffn_hidden_size=2048.
        self.bridge = _SwiGLU(d_hidden, bridge_ffn, d_out=local_hidden)

        # Per-codebook embeddings for previously-emitted codes (k = 1..R-1).
        self.dep_emb = nn.ModuleList(
            [nn.Embedding(K, local_hidden) for _ in range(R - 1)]
        )
        # Depth-0 BOS.
        self.dep_bos = nn.Parameter(torch.zeros(local_hidden))

        # Stack of Qwen3-style blocks.
        self.blocks = nn.ModuleList([
            _MossLocalBlock(local_hidden, local_num_heads, local_ffn)
            for _ in range(local_num_layers)
        ])
        self.final_norm = _RMSNorm(local_hidden)

        # Per-codebook output heads.
        self.linears = nn.ModuleList([
            nn.Linear(local_hidden, K, bias=False) for _ in range(R)
        ])

        # Init: Qwen3 default (trunc_normal std=0.02 for all linear / embedding
        # weights; RMSNorm gains stay at 1; BOS at 0.02).
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.trunc_normal_(self.dep_bos, std=0.02)

    def _depth_inputs(
        self,
        h: torch.Tensor,                    # (..., d_hidden)
        prev_codes: torch.Tensor,           # (..., R-1) long
        up_to_k: Optional[int] = None,
    ) -> torch.Tensor:
        K_steps = self.R if up_to_k is None else up_to_k
        h_local = self.bridge(h)            # (..., local_hidden)
        outs = []
        for k in range(K_steps):
            if k == 0:
                token_in = self.dep_bos.expand_as(h_local)
            else:
                token_in = self.dep_emb[k - 1](prev_codes[..., k - 1])
            outs.append(h_local + token_in)
        return torch.stack(outs, dim=-2)    # (..., K_steps, local_hidden)

    def _run_local(self, z: torch.Tensor) -> torch.Tensor:
        x = z
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)

    def forward(
        self,
        h: torch.Tensor,
        q_codes: torch.Tensor,
        dur_buckets: Optional[torch.Tensor] = None,
        return_depformer_last: bool = False,
    ):
        """h (B, T, d_hidden), q_codes (B, T, R) -> logits (B, T, R, K).

        `dur_buckets` and `return_depformer_last` are accepted for API parity
        with DepthTransformerPredictor but this predictor does not support
        either (the MOSS Local recipe has no duration injection). Pass-throughs
        from upstream callers that always send `None` / `False` are fine.
        """
        if dur_buckets is not None:
            raise ValueError(
                "MossLocalDepthPredictor does not consume duration embeddings; "
                "use acoustic_predictor_type='depth_transformer' for that."
            )
        B, T, _ = h.shape
        prev = q_codes[..., : self.R - 1]                      # (B, T, R-1)
        z = self._depth_inputs(h, prev)                        # (B, T, R, local_hidden)
        z_flat = z.reshape(B * T, self.R, self.local_hidden)
        u = self._run_local(z_flat)                            # (B*T, R, local_hidden)
        u = u.reshape(B, T, self.R, self.local_hidden)
        all_logits = [self.linears[k](u[..., k, :]) for k in range(self.R)]
        logits = torch.stack(all_logits, dim=-2)               # (B, T, R, K)
        if return_depformer_last:
            return logits, u[..., self.R - 1, :]
        return logits

    @torch.no_grad()
    def sample(
        self,
        h_last: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
        dur_bucket: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del dur_bucket
        B = h_last.size(0)
        device = h_last.device
        prev = torch.zeros(B, self.R - 1, dtype=torch.long, device=device)
        out_codes = torch.zeros(B, self.R, dtype=torch.long, device=device)
        for k in range(self.R):
            z_prefix = self._depth_inputs(h_last, prev, up_to_k=k + 1)
            u = self._run_local(z_prefix)
            logits_k = self.linears[k](u[:, k])
            q_k = _sample(logits_k, temperature, top_k, top_p)
            out_codes[:, k] = q_k
            if k < self.R - 1:
                prev[:, k] = q_k
        return out_codes


# --------------------------------------------------------------------------- #
# Hierarchical depth-transformer predictor used by StreamSLMHier (Moshi-style
# step-internal AR over [text, q^{0..R-1}, dur] with a configurable position
# of the duration head). Kept side-by-side with DepthTransformerPredictor so
# the legacy StreamSLM remains untouched.
# --------------------------------------------------------------------------- #
class HierDepthTransformerPredictor(nn.Module):
    """Depth-wise AR predictor that emits (q_n, d_n) conditioned on w_{n+1}.

    Operates on n_slots = R + 2 depth positions per timestep:

        slot 0 (text)  : output discarded; lm_head(h_n) predicts w_{n+1} upstream.
        Remaining slots are ordered by `hier_ar_order`:

        hier_ar_order = "duration_last":
            slots 1..R   -> q_n^0..q_n^{R-1}
            slot  R+1    -> d_n
        hier_ar_order = "duration_first":
            slot  1      -> d_n
            slots 2..R+1 -> q_n^0..q_n^{R-1}

    Each slot's input is ``Linear_k(h_n) + Emb_{prev}(prev_token)`` where
    ``prev_token`` is the previous AR step's GT (training, teacher-forced) or
    sample (inference). Slot 0 uses a learned BOS; slot 1's "previous" is the
    text step, embedded via the Llama input embedding then projected by
    ``text_to_depformer`` so the depth transformer sees the (shared) text
    embedding rather than a fresh (V, depformer_dim) table.

    Public API:
        forward(h, w_next_emb, q_codes, dur_input)
            -> (acoustic_logits (B, T, R, K), duration_pred (B, T, dur_out_dim))
        sample(h_last, w_next_emb, temperature_aco, top_k_aco, top_p_aco,
               temperature_dur=0.0, top_k_dur=0, top_p_dur=1.0)
            -> (q_codes (B, R), d_frames (B,) long)
    """

    def __init__(
        self,
        d_hidden: int,
        d_text_emb: int,
        R: int,
        K: int,
        depformer_dim: int = 128,
        depformer_num_heads: int = 4,
        depformer_num_layers: int = 2,
        depformer_dim_feedforward: Optional[int] = None,
        depformer_multi_linear: bool = False,
        depformer_weights_per_step: bool = False,
        depformer_norm: bool = False,
        duration_loss_type: str = "regression",
        duration_num_buckets: int = 256,
        hier_ar_order: str = "duration_last",
        prev_depth_output_dropout: float = 0.0,
    ):
        super().__init__()
        if hier_ar_order not in ("duration_first", "duration_last"):
            raise ValueError(f"unknown hier_ar_order: {hier_ar_order}")
        if duration_loss_type not in ("regression", "classification"):
            raise ValueError(f"unknown duration_loss_type: {duration_loss_type}")
        StreamingTransformer = _import_moshi_streaming_transformer()

        self.R = R
        self.K = K
        self.depformer_dim = depformer_dim
        self.multi_linear = depformer_multi_linear
        self.hier_ar_order = hier_ar_order
        self.duration_loss_type = duration_loss_type
        self.duration_num_buckets = duration_num_buckets
        # Depth slots: text + R q-codebooks + 1 duration.
        self.n_slots = R + 2

        self.prev_depth_output_dropout_p = float(prev_depth_output_dropout)
        self.prev_depth_output_dropout = (
            nn.Dropout(p=self.prev_depth_output_dropout_p)
            if self.prev_depth_output_dropout_p > 0.0
            else nn.Identity()
        )

        if depformer_dim_feedforward is None:
            depformer_dim_feedforward = 4 * depformer_dim

        # Per-slot (or shared) projection of the LLM hidden state into depformer_dim.
        n_in = self.n_slots if depformer_multi_linear else 1
        self.dep_in = nn.ModuleList(
            [nn.Linear(d_hidden, depformer_dim, bias=False) for _ in range(n_in)]
        )

        # Projection of the (shared Llama) next-text embedding into depformer_dim.
        # The actual embedding lookup happens upstream in StreamSLMHier so the
        # (V, d_hidden) table stays weight-tied with the Llama backbone.
        self.text_to_depformer = nn.Linear(d_text_emb, depformer_dim, bias=False)

        # Per-codebook embeddings of previously-emitted q codes. For
        # duration_last we need R tables (q^{R-1} feeds the duration slot);
        # for duration_first we only need R-1 (q^{R-1} feeds nothing).
        n_qemb = R if hier_ar_order == "duration_last" else R - 1
        self.dep_emb = nn.ModuleList(
            [nn.Embedding(K, depformer_dim) for _ in range(n_qemb)]
        )

        # Duration input embedding (only used in duration_first, where the
        # predicted/GT d_n is fed into the q0 slot).
        self.dur_emb: Optional[nn.Embedding] = None
        self.dur_in_reg: Optional[nn.Sequential] = None
        if hier_ar_order == "duration_first":
            if duration_loss_type == "classification":
                self.dur_emb = nn.Embedding(duration_num_buckets, depformer_dim)
            else:
                self.dur_in_reg = nn.Sequential(
                    nn.Linear(1, depformer_dim),
                    nn.GELU(),
                    nn.Linear(depformer_dim, depformer_dim, bias=False),
                )

        # Slot-0 BOS for the text "input slot" (its output is discarded but
        # the slot itself participates in depformer attention).
        self.dep_bos = nn.Parameter(torch.zeros(depformer_dim))

        # Depth transformer (Moshi StreamingTransformer).
        kwargs = dict(
            d_model=depformer_dim,
            num_heads=depformer_num_heads,
            num_layers=depformer_num_layers,
            dim_feedforward=depformer_dim_feedforward,
            causal=True,
            positional_embedding="sin",
            context=None,
        )
        if depformer_weights_per_step:
            kwargs["weights_per_step"] = self.n_slots
            kwargs["gating"] = "silu"
        self.depformer = StreamingTransformer(**kwargs)

        # Per-slot output norms + heads. Slot 0 is unused (text upstream),
        # acoustic linears index codebooks 0..R-1, duration head reads its
        # own slot.
        if depformer_norm:
            self.dep_norms = nn.ModuleList(
                [nn.LayerNorm(depformer_dim) for _ in range(self.n_slots)]
            )
        else:
            self.dep_norms = nn.ModuleList(
                [nn.Identity() for _ in range(self.n_slots)]
            )
        self.linears = nn.ModuleList(
            [nn.Linear(depformer_dim, K, bias=False) for _ in range(R)]
        )
        dur_out_dim = 1 if duration_loss_type == "regression" else duration_num_buckets
        self.duration_head = nn.Linear(depformer_dim, dur_out_dim)

        # Moshi-style trunc_normal init (std = 1 / sqrt(in)).
        for m in [*self.dep_in, *self.linears, self.duration_head,
                  self.text_to_depformer]:
            nn.init.trunc_normal_(m.weight, std=1.0 / (m.in_features ** 0.5))
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
        for emb in self.dep_emb:
            nn.init.trunc_normal_(emb.weight, std=1.0 / (emb.embedding_dim ** 0.5))
        if self.dur_emb is not None:
            nn.init.trunc_normal_(self.dur_emb.weight,
                                  std=1.0 / (self.dur_emb.embedding_dim ** 0.5))
        if self.dur_in_reg is not None:
            for sub in self.dur_in_reg:
                if isinstance(sub, nn.Linear):
                    nn.init.trunc_normal_(sub.weight,
                                          std=1.0 / (sub.in_features ** 0.5))
                    if sub.bias is not None:
                        nn.init.zeros_(sub.bias)
        nn.init.trunc_normal_(self.dep_bos, std=1.0 / (depformer_dim ** 0.5))

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _dep_in_for(self, slot: int) -> nn.Linear:
        return self.dep_in[slot] if self.multi_linear else self.dep_in[0]

    def _embed_duration(self, dur_input: torch.Tensor) -> torch.Tensor:
        """Embed the (GT or predicted) duration into depformer_dim.

        Shape: dur_input is (..., ). Classification = long bucket index;
        regression = float log1p(frames).
        """
        if self.dur_emb is not None:
            return self.dur_emb(dur_input.long())
        assert self.dur_in_reg is not None
        x = dur_input.to(torch.float32).unsqueeze(-1)
        return self.dur_in_reg(x)

    # ------------------------------------------------------------------ #
    # Build per-slot inputs (training: teacher-forced full sequence).
    # ------------------------------------------------------------------ #
    def _depth_inputs(
        self,
        h: torch.Tensor,                # (..., d_hidden)
        w_next_emb: torch.Tensor,       # (..., d_text_emb)
        q_codes: torch.Tensor,          # (..., R) long
        dur_input: Optional[torch.Tensor],  # (...,) long/float; only used for dur_first
        up_to_k: Optional[int] = None,
    ) -> torch.Tensor:
        K_steps = self.n_slots if up_to_k is None else up_to_k
        text_in_emb = self.text_to_depformer(w_next_emb)        # (..., depformer_dim)
        dur_in_emb = None
        if self.hier_ar_order == "duration_first":
            if dur_input is None:
                raise ValueError(
                    "duration_first variant requires dur_input (GT or sample)"
                )
            dur_in_emb = self._embed_duration(dur_input)

        outs = []
        for k in range(K_steps):
            h_proj = self._dep_in_for(k)(h)
            if k == 0:
                token_in = self.dep_bos.expand_as(h_proj)
            elif self.hier_ar_order == "duration_last":
                if k == 1:
                    token_in = text_in_emb
                elif k <= self.R:
                    # slot k (k>=2) reads q_n^{k-2}
                    token_in = self.dep_emb[k - 2](q_codes[..., k - 2])
                    token_in = self.prev_depth_output_dropout(token_in)
                else:  # k == R + 1, duration slot
                    token_in = self.dep_emb[self.R - 1](q_codes[..., self.R - 1])
                    token_in = self.prev_depth_output_dropout(token_in)
            else:  # duration_first
                if k == 1:
                    token_in = text_in_emb
                elif k == 2:
                    token_in = dur_in_emb       # type: ignore[assignment]
                else:
                    # slot k (k>=3) reads q_n^{k-3}
                    token_in = self.dep_emb[k - 3](q_codes[..., k - 3])
                    token_in = self.prev_depth_output_dropout(token_in)
            outs.append(h_proj + token_in)
        return torch.stack(outs, dim=-2)        # (..., K_steps, depformer_dim)

    # ------------------------------------------------------------------ #
    # Training: teacher-forced over (B, T, ...) tensors.
    # ------------------------------------------------------------------ #
    def forward(
        self,
        h: torch.Tensor,              # (B, T, d_hidden)
        w_next_emb: torch.Tensor,     # (B, T, d_text_emb)
        q_codes: torch.Tensor,        # (B, T, R) long
        dur_input: Optional[torch.Tensor] = None,  # (B, T) long/float; required for dur_first
    ):
        B, T, _ = h.shape
        z = self._depth_inputs(h, w_next_emb, q_codes, dur_input)
        z_flat = z.reshape(B * T, self.n_slots, self.depformer_dim)
        u = self.depformer(z_flat)
        u = u.reshape(B, T, self.n_slots, self.depformer_dim)

        # Decode per-slot outputs.
        if self.hier_ar_order == "duration_last":
            aco_offset = 1                   # slots 1..R
            dur_slot = self.R + 1
        else:                                # duration_first
            aco_offset = 2                   # slots 2..R+1
            dur_slot = 1

        aco_logits_list = []
        for k in range(self.R):
            slot = aco_offset + k
            u_k = self.dep_norms[slot](u[..., slot, :])
            aco_logits_list.append(self.linears[k](u_k))
        aco_logits = torch.stack(aco_logits_list, dim=-2)  # (B, T, R, K)

        u_dur = self.dep_norms[dur_slot](u[..., dur_slot, :])
        dur_pred = self.duration_head(u_dur)               # (B, T, dur_out_dim)
        return aco_logits, dur_pred

    # ------------------------------------------------------------------ #
    # Inference: depth-AR over slots 0..n_slots-1 at a single time step.
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample(
        self,
        h_last: torch.Tensor,         # (B, d_hidden)
        w_next_emb: torch.Tensor,     # (B, d_text_emb)  -- already embedded
        temperature_aco: float,
        top_k_aco: int,
        top_p_aco: float,
        temperature_dur: float = 0.0,
        top_k_dur: int = 0,
        top_p_dur: float = 1.0,
    ):
        B = h_last.size(0)
        device = h_last.device

        out_q = torch.zeros(B, self.R, dtype=torch.long, device=device)
        d_frames = torch.zeros(B, dtype=torch.long, device=device)

        # In duration_first, the predicted d_n is fed back into slot 2's input.
        # We re-encode the sampled frame count into the model's input format
        # (bucket index for classification, log1p(frames) for regression) so
        # the AR feedback matches the discrete d_frames we return.
        dur_input_for_slot2: Optional[torch.Tensor] = None

        text_in_emb = self.text_to_depformer(w_next_emb)         # (B, depformer_dim)

        z_list = []
        for slot in range(self.n_slots):
            h_proj = self._dep_in_for(slot)(h_last)              # (B, depformer_dim)
            if slot == 0:
                token_in = self.dep_bos.expand_as(h_proj)
            elif self.hier_ar_order == "duration_last":
                if slot == 1:
                    token_in = text_in_emb
                elif slot <= self.R:
                    token_in = self.dep_emb[slot - 2](out_q[:, slot - 2])
                else:                                            # duration slot
                    token_in = self.dep_emb[self.R - 1](out_q[:, self.R - 1])
            else:  # duration_first
                if slot == 1:
                    token_in = text_in_emb
                elif slot == 2:
                    assert dur_input_for_slot2 is not None
                    token_in = self._embed_duration(dur_input_for_slot2)
                else:
                    token_in = self.dep_emb[slot - 3](out_q[:, slot - 3])
            z_list.append(h_proj + token_in)

            z_prefix = torch.stack(z_list, dim=1)                # (B, slot+1, D)
            u = self.depformer(z_prefix)                         # (B, slot+1, D)
            u_slot = self.dep_norms[slot](u[:, slot])

            if slot == 0:
                continue                                         # text slot: discard

            if self.hier_ar_order == "duration_last":
                if slot <= self.R:
                    k = slot - 1
                    logits = self.linears[k](u_slot)
                    out_q[:, k] = _sample(logits, temperature_aco,
                                          top_k_aco, top_p_aco)
                else:                                            # duration slot
                    dur_pred = self.duration_head(u_slot)
                    d_frames = self._dur_to_frames(
                        dur_pred, temperature_dur, top_k_dur, top_p_dur,
                    )
            else:                                                # duration_first
                if slot == 1:                                    # duration slot
                    dur_pred = self.duration_head(u_slot)
                    d_frames = self._dur_to_frames(
                        dur_pred, temperature_dur, top_k_dur, top_p_dur,
                    )
                    dur_input_for_slot2 = self._frames_to_dur_input(d_frames)
                else:
                    k = slot - 2
                    logits = self.linears[k](u_slot)
                    out_q[:, k] = _sample(logits, temperature_aco,
                                          top_k_aco, top_p_aco)

        return out_q, d_frames

    # ------------------------------------------------------------------ #
    # Inference-side conversions between (head output) <-> (frame count) <->
    # (next-slot input embedding format).
    # ------------------------------------------------------------------ #
    def _dur_to_frames(
        self,
        dur_pred: torch.Tensor,        # (B, dur_out_dim)
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:
        if self.duration_loss_type == "classification":
            if temperature <= 0:
                bucket = dur_pred.argmax(dim=-1)
            else:
                bucket = _sample(dur_pred, temperature, top_k, top_p)
            return bucket.clamp(min=0,
                                max=self.duration_num_buckets - 1).long()
        d_log = dur_pred.squeeze(-1)                        # (B,)
        return torch.expm1(d_log).round().clamp(min=1).long()

    def _frames_to_dur_input(self, d_frames: torch.Tensor) -> torch.Tensor:
        """Re-encode a sampled frame count into the predictor's input format."""
        if self.duration_loss_type == "classification":
            return d_frames.clamp(min=0,
                                  max=self.duration_num_buckets - 1).long()
        return torch.log1p(d_frames.to(torch.float32))
