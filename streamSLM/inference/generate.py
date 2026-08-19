"""Auto-regressive generation for StreamSLM (symmetric-delay convention).

Produces a sequence of (subword w_n, acoustic codes q_n, duration d_n) tuples
from a trained checkpoint. Audio reconstruction lives in `reconstruct.py` and
consumes the .units.pt this script writes.

Symmetric-delay (MusicGen-style next-step AR) convention. The model's
``_delay_shift_audio`` front-pads the audio stream by ``D = model.delay`` so
audio_input at LLM position n is ``q[n - D]`` (pad for n < D). With acoustic
targets right-shifted by ``D-1`` in training (``aco_label_mask`` over
[D-1, L_i+D-2]), the head at hidden h_n predicts:

    text head  -> w_{n+1}             (HF causal-LM shift; same as before)
    aco head   -> q_{n - (D-1)}       (NEW: one-step-ahead of the audio input)
    duration   -> d_{n - (D-1)}

Inference loop. At iteration with current length ``L = w.numel()`` (top of
loop, before append), we forward the prefix, sample (w_next, q_next, d_next)
from logits at position L-1, then:

  1. Backfill at index ``L - D`` — this is the audio slot whose
     ground-truth value q_next now estimates. Skip when L-D < 0 or
     L-D < n_real_q (don't overwrite the speech prompt's real codes).
  2. Append w_next to w, and zero-placeholder slots to q / d.

Stop conditions (bottom of loop):

  * EOS sampled at slot n_eos_pos = L (the old length before append): keep
    iterating D-1 more times so we backfill the last real audio q_{N-1}
    (predicted from h at pos n_eos_pos + D - 2). Break when
    ``w.numel() >= n_eos_pos + D``; trim to ``end = n_eos_pos`` (drop EOS
    and the D-1 trailing placeholders).
  * No EOS: break when ``w.numel() >= T_p + max_new_tokens + D``; trim to
    ``end = w.numel() - D``.

INCOMPATIBILITY NOTE. Checkpoints trained under the old asymmetric-delay
convention (aco at h_n predicts q_n, no EOS extension) cannot be used with
this generator — the sampled q would correspond to a different slot than
the backfill index here.

We do not use HF KV-cache: the input at each step depends on the *fused*
embedding, not a single token, so we re-feed the full prefix each step.
This is fine for moderate-length validation outputs; swap in cached
generation later if it becomes a hot path.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from streamSLM.config import ModelConfig, TokenizerConfig
from streamSLM.model.slm import StreamSLM, load_lm_tokenizer
from streamSLM.model.slm_hier import StreamSLMHier
from streamSLM.units import SubwordUnits


# --------------------------------------------------------------------------- #
# Sampling helpers
# --------------------------------------------------------------------------- #
def _top_k_top_p(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """Filter a 1-D logits vector by top-k / nucleus, returning logits in-place."""
    if top_k > 0:
        v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
        logits = torch.where(logits < v[..., -1:], torch.full_like(logits, -float("inf")), logits)
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        keep = probs <= top_p
        keep[..., 0] = True  # always keep the top
        sorted_logits = torch.where(keep, sorted_logits, torch.full_like(sorted_logits, -float("inf")))
        logits = torch.empty_like(logits).scatter_(-1, sorted_idx, sorted_logits)
    return logits


def _sample_categorical(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    """logits: (..., V) -> (...,) long sample."""
    if temperature <= 0:
        return logits.argmax(dim=-1)
    z = _top_k_top_p(logits / temperature, top_k=top_k, top_p=top_p)
    probs = F.softmax(z, dim=-1)
    flat = probs.reshape(-1, probs.size(-1))
    out = torch.multinomial(flat, num_samples=1).squeeze(-1)
    return out.reshape(probs.shape[:-1])


# --------------------------------------------------------------------------- #
# Core generation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def generate(
    model: nn.Module,
    prompt_ids: torch.Tensor,            # (T_p,) long  -- mandatory prompt; >= 1 token (e.g. BOS)
    prompt_q: Optional[torch.Tensor],    # (T_p, R) long or None  (None = text-only prompt)
    prompt_d: Optional[torch.Tensor],    # (T_p,) long or None
    max_new_tokens: int = 256,
    eos_id: Optional[int] = None,
    temperature_text: float = 1.0,
    top_k_text: int = 0,
    top_p_text: float = 1.0,
    temperature_aco: float = 1.0,
    top_k_aco: int = 0,
    top_p_aco: float = 1.0,
    device: Optional[torch.device] = None,
    teacher_rvq: Optional[nn.Module] = None,
    allowed_text_token_ids: Optional[torch.Tensor] = None,
    redo_final_aco: int = 0,
) -> SubwordUnits:
    """Greedy / sampled AR generation of (w_n, q_n, d_n).

    Text-only prompts (`prompt_q is None`) are force-fed one subword at a
    time so every non-PAD position sees in-distribution acoustic context
    (sampled by the model itself). Feeding the entire prompt with all-zero
    q/d in one batch is OOD vs training and produces unintelligible audio.

    Supports both :class:`StreamSLM` (parallel heads off h_n) and
    :class:`StreamSLMHier` (step-internal depth-AR conditioned on w_{n+1});
    we detect the hier path via ``model.is_hier_ar`` and re-order the
    sampling so the text head fires first.
    """
    device = device or next(model.parameters()).device
    R = model.R
    if allowed_text_token_ids is not None:
        allowed_text_token_ids = allowed_text_token_ids.to(
            device=device, dtype=torch.long).view(-1)
    # Built lazily on first iteration once we know text_logits' actual width
    # (Llama-3 includes special-token rows beyond tok.vocab_size; the head's
    # output dim is what matters here).
    allowed_text_mask: Optional[torch.Tensor] = None
    is_continuous = getattr(model, "acoustic_target", "rvq") == "continuous"
    is_hier = getattr(model, "is_hier_ar", False)
    if is_continuous and teacher_rvq is None:
        raise ValueError(
            "model.acoustic_target='continuous' requires teacher_rvq "
            "(load via _load_teacher_residual_vq) so the predicted "
            "continuous feature can be quantised back into RVQ codes "
            "for the AR feedback loop and downstream reconstruct."
        )
    if is_hier and is_continuous:
        raise ValueError(
            "StreamSLMHier is discrete-only; acoustic_target='continuous' "
            "is not supported."
        )

    prompt_ids = prompt_ids.to(device).long().view(-1)
    T_p = prompt_ids.numel()
    text_only = prompt_q is None
    D = int(model.delay)

    if text_only:
        # Seed with the first prompt token only; pos 0 hits the learned PAD
        # branch in _delay_shift_audio, which is exactly what training saw.
        w = prompt_ids[:1].clone()
        q = torch.zeros(1, R, dtype=torch.long, device=device)
        d = torch.zeros(1, dtype=torch.long, device=device)
        n_real_q = 0
    else:
        w = prompt_ids.clone()
        q = prompt_q.to(device).long()
        d = prompt_d.to(device).long()
        assert q.shape == (T_p, R) and d.shape == (T_p,), \
            "prompt_q/prompt_d must align with prompt_ids"
        # Real q/d are provided for all T_p prompt positions; do not
        # overwrite them with model samples. Optionally, drop the last
        # `redo_final_aco` acoustic positions so the model re-predicts
        # them (those frames often encode end-of-utterance silence and
        # bias the continuation toward silence/EOS).
        k_redo = max(0, min(int(redo_final_aco), T_p))
        n_real_q = T_p - k_redo
        if k_redo > 0:
            q[n_real_q:] = 0
            d[n_real_q:] = 0

    model.eval()
    target_text_total = T_p + max_new_tokens
    n_eos_pos: Optional[int] = None
    end: int = 0

    while True:
        L = w.numel()
        attn = torch.ones((1, L), dtype=torch.bool, device=device)
        out = model(
            w.unsqueeze(0), q.unsqueeze(0), d.unsqueeze(0), attn,
        )
        text_logits = out["text_logits"][0, -1]              # (V,)
        h_last = out["hidden"][:, -1]                         # (1, d_hidden)

        # Decide next text token (force-feed remaining prompt for text-only,
        # else sample from the text head). For the hier model this *must*
        # happen before the acoustic+duration sample so we can teacher-feed
        # w_{n+1} into slot 1 of the depth transformer.
        force_feed = text_only and L < T_p
        if force_feed:
            w_next = prompt_ids[L]
        else:
            if allowed_text_token_ids is not None:
                V = text_logits.size(-1)
                if allowed_text_mask is None or allowed_text_mask.numel() != V:
                    allowed_text_mask = torch.zeros(V, dtype=torch.bool, device=device)
                    keep = allowed_text_token_ids[allowed_text_token_ids < V]
                    allowed_text_mask[keep] = True
                    if eos_id is not None and 0 <= int(eos_id) < V:
                        allowed_text_mask[int(eos_id)] = True
                text_logits = text_logits.masked_fill(~allowed_text_mask, -float("inf"))
            w_next = _sample_categorical(
                text_logits, temperature_text, top_k_text, top_p_text
            )                                                 # ()

        # Sample q_{L-D} (current-step acoustic), d_{L-D} (current-step
        # duration). Under symmetric delay, aco/dur heads at hidden pos L-1
        # estimate the slot D-1 *behind* the audio input at that position
        # (which is q[L-1-D]); the predicted slot is L-D.
        if is_continuous:
            # Continuous-acoustic model: regress to a 256-dim teacher
            # pre-quantizer feature, then push that feature through the teacher's
            # ResidualVQ to recover (R,) codebook indices. This is the
            # "quantize-then-decode" path: the AR feedback embedding stays
            # code-compatible (the model was trained on RVQ-coded inputs),
            # and the .units.pt we save downstream feeds reconstruct.py
            # without any further changes.
            feat_last = out["acoustic_feat_pred"][0, -1]      # (D,)
            _, q_idx, _ = teacher_rvq(
                feat_last.to(torch.float32).view(1, 1, -1)
            )                                                 # (1, 1, R)
            q_next = q_idx[0, 0].long()                       # (R,)
            dur_pred = out["duration_pred"][0, -1]
            d_next_frames = torch.clamp(
                torch.expm1(dur_pred).round(), min=1
            ).to(torch.long)
        elif is_hier:
            # Step-internal depth-AR. The predictor returns d already in
            # frame units (regression: round(expm1) clamp>=1; classification:
            # bucket clamp into [0, num_buckets-1]).
            q_codes_b, d_frames_b = model.sample_acoustic_duration(
                h_last, w_next.view(1),
                temperature_aco=temperature_aco,
                top_k_aco=top_k_aco,
                top_p_aco=top_p_aco,
            )
            q_next = q_codes_b[0]                             # (R,)
            d_next_frames = d_frames_b[0]                     # ()
        else:
            # The predictor handles flat multihead vs depth-wise AR sampling.
            q_next = model.acoustic_predictor.sample(
                h_last, temperature_aco, top_k_aco, top_p_aco
            )[0]                                              # (R,)
            dur_pred = out["duration_pred"][0, -1]            # ()
            # Duration head regresses log(1+frames); convert back, clamp >= 1.
            d_next_frames = torch.clamp(
                torch.expm1(dur_pred).round(), min=1
            ).to(torch.long)                                  # ()

        # Symmetric-delay backfill at index L-D. Skip when the slot is in
        # the speech-prompt region (idx < n_real_q) or before the start
        # (idx < 0, i.e. the model hasn't seen enough audio context yet).
        idx_back = L - D
        if idx_back >= 0 and idx_back >= n_real_q:
            q[idx_back] = q_next
            d[idx_back] = d_next_frames

        # Append a placeholder slot for the next step.
        w = torch.cat([w, w_next.view(1)], dim=0)
        q = torch.cat([q, torch.zeros(1, R, dtype=torch.long, device=device)], dim=0)
        d = torch.cat([d, torch.zeros(1, dtype=torch.long, device=device)], dim=0)

        # First EOS sampled: mark n_eos_pos (slot where EOS landed = old L),
        # but keep iterating D-1 more steps so the last real q_{N-1} gets
        # backfilled (predicted at hidden pos n_eos_pos + D - 2).
        if (eos_id is not None and not force_feed
                and n_eos_pos is None and int(w_next) == int(eos_id)):
            n_eos_pos = L

        # Stop conditions (bottom of loop, after the append).
        if n_eos_pos is not None:
            if w.numel() >= n_eos_pos + D:
                end = n_eos_pos
                break
        else:
            if w.numel() >= target_text_total + D:
                end = w.numel() - D
                break

    return SubwordUnits(
        subword_ids=w[:end].cpu(),
        q_codes=q[:end].cpu(),
        duration_frames=d[:end].cpu(),
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _load_streamslm(checkpoint: str, device: torch.device) -> Tuple[nn.Module, TokenizerConfig, ModelConfig]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved_args = ckpt.get("args", {})
    tok_cfg = TokenizerConfig(
        token_type=saved_args.get("token_type", "rvq"),
        rvq_num_quantizers=saved_args.get("rvq_num_quantizers", 16),
        rvq_codebook_size=saved_args.get("rvq_codebook_size", 512),
        text_tokenizer=saved_args.get("text_tokenizer", "llama"),
    )
    dep_ff = saved_args.get("depformer_dim_feedforward", 0)
    model_cfg = ModelConfig(
        backbone=saved_args.get("backbone", "meta-llama/Llama-3.2-1B"),
        delay=saved_args.get("delay", 1),
        duration_loss=saved_args.get("duration_loss", "l1"),
        acoustic_predictor_type=saved_args.get(
            "acoustic_predictor_type", "multihead"
        ),
        depformer_dim=saved_args.get("depformer_dim", 128),
        depformer_num_heads=saved_args.get("depformer_num_heads", 4),
        depformer_num_layers=saved_args.get("depformer_num_layers", 2),
        depformer_dim_feedforward=(dep_ff or None),
        depformer_multi_linear=saved_args.get("depformer_multi_linear", False),
        depformer_weights_per_step=saved_args.get("depformer_weights_per_step", False),
        depformer_norm=saved_args.get("depformer_norm", False),
        fusion_mode=saved_args.get("fusion_mode", "gated_softmax"),
        audio_emb_init_scale=saved_args.get("audio_emb_init_scale", "auto"),
        duration_head_input=saved_args.get("duration_head_input", "llm"),
        duration_loss_type=saved_args.get("duration_loss_type", "regression"),
        duration_num_buckets=saved_args.get("duration_num_buckets", 256),
        feed_duration_to_depformer=saved_args.get("feed_duration_to_depformer", False),
        acoustic_target=saved_args.get("acoustic_target", "rvq"),
        acoustic_feat_dim=saved_args.get("acoustic_feat_dim", 256),
        acoustic_feat_loss=saved_args.get("acoustic_feat_loss", "l1"),
        acoustic_layer_mix=saved_args.get("acoustic_layer_mix", "last"),
        model_arch=saved_args.get("model_arch", "streamslm"),
        hier_ar_order=saved_args.get("hier_ar_order", "duration_last"),
    )
    model_cls = StreamSLMHier if model_cfg.model_arch == "streamslm_hier" else StreamSLM
    model = model_cls(model_cfg, tok_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tok_cfg, model_cfg


def _load_teacher_residual_vq(
    teacher_ckpt: str,
    num_quantizers: int,
    codebook_size: int,
    feat_dim: int,
    device: torch.device,
    codebook_dim: Optional[int] = None,
) -> nn.Module:
    """Load just the teacher's ResidualVQ submodule (RVQ teachers).

    Mirror of ``_load_teacher_residual_vq`` but for plain ResidualVQ
    teachers (model_tokenizer). Continuous-acoustic SLMs distilled
    from an RVQ teacher need this quantiser at inference to map their
    256-d feature predictions back to (R,) codebook indices.
    """
    from vector_quantize_pytorch import ResidualVQ  # type: ignore

    rvq_kwargs = dict(
        dim=feat_dim,
        num_quantizers=num_quantizers,
        codebook_size=codebook_size,
        kmeans_init=True,
        kmeans_iters=100,
        threshold_ema_dead_code=2,
    )
    if codebook_dim is not None:
        rvq_kwargs["codebook_dim"] = codebook_dim
    rvq = ResidualVQ(**rvq_kwargs)
    ckpt = torch.load(teacher_ckpt, map_location="cpu", weights_only=False)
    sd = ckpt.get("hubert_state_dict", ckpt.get("model", ckpt))
    if any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
    sub = {k[len("residual_vq."):]: v
           for k, v in sd.items() if k.startswith("residual_vq.")}
    if not sub:
        raise RuntimeError(
            f"teacher checkpoint {teacher_ckpt} contains no residual_vq.* keys"
        )
    missing, unexpected = rvq.load_state_dict(sub, strict=False)
    if unexpected:
        print(f"[teacher_rvq] unexpected keys: {unexpected}")
    return rvq.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="StreamSLM .pt (saved by train.py)")
    ap.add_argument("--prompt", default="", help="Optional text prompt to seed generation")
    ap.add_argument("--prompt_units", default="",
                    help="Optional .units.pt to prepend as a speech-continuation prompt. "
                         "Takes priority over --prompt; truncated to "
                         "--prompt_max_subwords if set.")
    ap.add_argument("--prompt_max_subwords", type=int, default=0,
                    help="If --prompt_units is set, keep only the first N subwords (0 = all).")
    ap.add_argument("--out", required=True, help="Path for generated SubwordUnits .pt")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature_text", type=float, default=0.9)
    ap.add_argument("--top_p_text", type=float, default=0.95)
    ap.add_argument("--top_k_text", type=int, default=0)
    # Acoustic head is deterministic by default (temp=0 -> argmax in _sample).
    # Text head stays stochastic; pass --temperature_aco>0 to re-enable sampling.
    ap.add_argument("--temperature_aco", type=float, default=0.0)
    ap.add_argument("--top_p_aco", type=float, default=1.0)
    ap.add_argument("--top_k_aco", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--teacher_checkpoint", default="",
                    help="StreamAlign teacher .pt; required for "
                         "acoustic_target='continuous' models so the "
                         "regressed feature can be re-quantised back into "
                         "codes for AR feedback + downstream reconstruct.")
    ap.add_argument("--bf16", action="store_true",
                    help="Wrap model forward in torch.autocast(bfloat16). "
                         "Required when the Llama backbone uses Flash "
                         "Attention 2 (FA2 only supports fp16/bf16). "
                         "Params stay fp32; mirrors training behavior.")
    ap.add_argument("--allowed_text_token_ids", default="",
                    help="Optional path to a .pt tensor of allowed LM-tokenizer "
                         "ids (1-D long). When set, non-allowed ids are masked "
                         "to -inf before text sampling, restricting the model "
                         "to vocabulary seen in the training cache (avoids "
                         "Llama backbone leakage of uppercase/punctuation).")
    ap.add_argument("--redo_final_aco", type=int, default=0,
                    help="Drop the last K acoustic positions of the speech "
                         "prompt and let the model re-predict them. Mitigates "
                         "trailing-silence/EOS bias from the prompt's final "
                         "acoustic frames. Typically set to model.delay.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tok_cfg, model_cfg = _load_streamslm(args.checkpoint, device)
    lm_tok = load_lm_tokenizer(tok_cfg)

    teacher_rvq = None
    if model_cfg.acoustic_target == "continuous":
        if not args.teacher_checkpoint:
            raise SystemExit(
                "checkpoint trained with acoustic_target='continuous' but "
                "--teacher_checkpoint was not supplied (needed to re-quantise "
                "the predicted continuous feature)."
            )
        print(f"[load] teacher residual_vq from {args.teacher_checkpoint}")
        teacher_rvq = _load_teacher_residual_vq(
            args.teacher_checkpoint,
            num_quantizers=tok_cfg.rvq_num_quantizers,
            codebook_size=tok_cfg.rvq_codebook_size,
            feat_dim=model_cfg.acoustic_feat_dim,
            device=device,
        )

    if args.prompt_units:
        prefix = SubwordUnits.load(args.prompt_units)
        if args.prompt_max_subwords > 0:
            keep = min(args.prompt_max_subwords, prefix.num_subwords)
            prefix = SubwordUnits(
                subword_ids=prefix.subword_ids[:keep],
                q_codes=prefix.q_codes[:keep],
                duration_frames=prefix.duration_frames[:keep],
            )
        print(f"[prompt] units prefix: N={prefix.num_subwords} "
              f"R={prefix.num_quantizers} frames={int(prefix.duration_frames.sum())}")
        prompt_ids = prefix.subword_ids.long()
        prompt_q = prefix.q_codes.long()
        prompt_d = prefix.duration_frames.long()
    elif args.prompt:
        prompt_ids = lm_tok(args.prompt, return_tensors="pt", add_special_tokens=True).input_ids[0]
        prompt_q = None
        prompt_d = None
    else:
        bos = lm_tok.bos_token_id if lm_tok.bos_token_id is not None else lm_tok.eos_token_id
        prompt_ids = torch.tensor([bos], dtype=torch.long)
        prompt_q = None
        prompt_d = None

    allowed_text_token_ids = None
    if args.allowed_text_token_ids:
        allowed_text_token_ids = torch.load(
            args.allowed_text_token_ids, map_location="cpu", weights_only=True
        ).long().view(-1)
        # BOS is also added in-loop; EOS is teacher-forced via eos_id.
        if lm_tok.bos_token_id is not None:
            allowed_text_token_ids = torch.cat([
                allowed_text_token_ids,
                torch.tensor([int(lm_tok.bos_token_id)], dtype=torch.long),
            ]).unique()
        print(f"[vocab_mask] {allowed_text_token_ids.numel()} allowed ids "
              f"(from {args.allowed_text_token_ids})")

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.bf16 and device.type == "cuda"
        else contextlib.nullcontext()
    )
    with amp_ctx:
        units = generate(
            model,
            prompt_ids=prompt_ids,
            prompt_q=prompt_q,
            prompt_d=prompt_d,
            max_new_tokens=args.max_new_tokens,
            eos_id=lm_tok.eos_token_id,
            temperature_text=args.temperature_text,
            top_k_text=args.top_k_text,
            top_p_text=args.top_p_text,
            temperature_aco=args.temperature_aco,
            top_k_aco=args.top_k_aco,
            top_p_aco=args.top_p_aco,
            device=device,
            teacher_rvq=teacher_rvq,
            allowed_text_token_ids=allowed_text_token_ids,
            redo_final_aco=args.redo_final_aco,
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    units.save(args.out)
    text = lm_tok.decode(units.subword_ids.tolist())
    text_path = Path(args.out).with_suffix(".txt")
    text_path.write_text(text)
    print(f"[generate] wrote {args.out}")
    print(f"[generate] wrote {text_path}")
    print(f"[generate] N={units.subword_ids.numel()}  total_frames={int(units.duration_frames.sum())}")
    print(f"[generate] text: {text!r}")


if __name__ == "__main__":
    main()
