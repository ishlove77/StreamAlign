#!/usr/bin/env python3
"""StreamSLM training loop.

Single-process or single-node multi-GPU (DDP via torchrun). Designed first
for the validation run on a small subset; scales to LibriSpeech 960h + Emilia
400h once manifests are extracted.

Usage (single GPU, validation):
    sr 1 48 --qos=q-low python -m streamSLM.train.train \
        --manifest cache/streamSLM_units/libritts/train-clean-100/manifest_shard0_of1.csv \
        --out_dir checkpoints/slm/validation \
        --max_steps 5000

Usage (DDP):
    sr 4 48 --qos=q-low torchrun --nproc_per_node=4 -m streamSLM.train.train ...
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import timedelta
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from streamSLM.config import ModelConfig, TokenizerConfig, TrainConfig
from streamSLM.data.dataset import (
    PackedSubwordUnitsDataset,
    PackingCollator,
    PadCollator,
    SubwordUnitsDataset,
)
from streamSLM.model.slm import StreamSLM, load_lm_tokenizer
from streamSLM.model.slm_hier import StreamSLMHier


# --------------------------------------------------------------------------- #
# DDP helpers
# --------------------------------------------------------------------------- #
def _maybe_init_dist():
    """Init torch.distributed if launched via torchrun; else single-process."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Long timeout: rank-0 ckpt read + ref_model load can take 20+ min on cold
        # NFS, while other ranks wait at the first collective. Default 10 min
        # watchdog kills the run before rank 0 starts broadcasting.
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=1))
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        return rank, world, local_rank, True
    return 0, 1, 0, False


def _is_main(rank: int) -> bool:
    return rank == 0


def _load_teacher_residual_vq(
    teacher_ckpt: str,
    num_quantizers: int,
    codebook_size: int,
    feat_dim: int,
    device: torch.device,
) -> nn.Module:
    """Load just the teacher's ResidualVQ submodule for validation re-quantisation.

    Mirrors streamSLM/inference/generate.py:_load_teacher_residual_vq. Used to
    map continuous-acoustic predictions back to (R,) codebook indices so we
    can compute per-codebook accuracy at val time. Eval-only, no grads.
    """
    from vector_quantize_pytorch import ResidualVQ  # type: ignore
    rvq = ResidualVQ(
        dim=feat_dim,
        num_quantizers=num_quantizers,
        codebook_size=codebook_size,
        kmeans_init=True,
        kmeans_iters=100,
        threshold_ema_dead_code=2,
    )
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
        print(f"[teacher_rvq] unexpected keys: {unexpected}", flush=True)
    return rvq.to(device).eval()


# --------------------------------------------------------------------------- #
# Optim / schedule
# --------------------------------------------------------------------------- #
def _build_optimizer(model: nn.Module, lr: float, weight_decay: float = 0.01):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or n.endswith(".bias") or "norm" in n.lower() or "embed" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if os.environ.get("STREAMSLM_OPT_8BIT", "0") == "1":
        from bitsandbytes.optim import AdamW8bit
        print("[optim] using bitsandbytes AdamW8bit (env STREAMSLM_OPT_8BIT=1)",
              flush=True)
        return AdamW8bit(groups, lr=lr, betas=(0.9, 0.95))
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))


def _lr_at(step: int, base_lr: float, warmup: int, total: int, min_ratio: float = 0.1,
           step_at: int = -1, step_to: float = 0.0) -> float:
    """Linear warmup then cosine decay to base_lr * min_ratio.
    If step_at > 0 and step >= step_at, returns step_to (constant) instead."""
    if step_at > 0 and step >= step_at:
        return step_to
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    if step >= total:
        return base_lr * min_ratio
    progress = (step - warmup) / max(total - warmup, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_ratio + (1.0 - min_ratio) * cosine)


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--manifest", required=True, nargs="+",
                    help="One or more manifest CSVs / globs / dirs produced by "
                         "extract_tokens.py (combined into a single dataset).")
    ap.add_argument("--min_subwords", type=int, default=4)
    ap.add_argument("--max_subwords", type=int, default=1024)
    # model
    ap.add_argument("--backbone", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--ref_model", default="meta-llama/Llama-3.2-1B",
                    help="Frozen text-only reference LLM for KL distillation "
                         "(TASTE stage-2). Empty string disables.")
    ap.add_argument("--text_tokenizer", choices=["llama", "qwen3"], default="llama")
    ap.add_argument("--token_type", choices=["rvq"], default="rvq")
    ap.add_argument("--rvq_num_quantizers", type=int, default=16)
    ap.add_argument("--rvq_codebook_size", type=int, default=512)
    ap.add_argument("--delay", type=int, default=1)
    ap.add_argument("--loss_w_text", type=float, default=1.0)
    ap.add_argument("--loss_w_acoustic", type=float, default=1.0)
    ap.add_argument("--loss_w_duration", type=float, default=0.1)
    ap.add_argument("--text_kl_weight", type=float, default=0.9,
                    help="Blend for text loss: kl_w*KL + (1-kl_w)*CE. "
                         "Default 0.9 matches TASTE stage-2. Set to 0.0 to "
                         "drop KL entirely; the trainer will then skip "
                         "loading --ref_model so its memory and fwd cost "
                         "are recovered.")
    ap.add_argument("--duration_loss", choices=["l1", "mse"], default="l1")
    # Residual-VQ predictor: legacy flat head ("multihead") or Moshi-style
    # depth-wise AR transformer ("depth_transformer"). See predictors.py.
    ap.add_argument("--acoustic_predictor_type",
                    choices=["multihead", "depth_transformer", "moss_local_depth"],
                    default="multihead")
    ap.add_argument("--depformer_dim", type=int, default=128)
    ap.add_argument("--depformer_num_heads", type=int, default=4)
    ap.add_argument("--depformer_num_layers", type=int, default=2)
    ap.add_argument("--depformer_dim_feedforward", type=int, default=0,
                    help="0 -> use 4 * depformer_dim.")
    ap.add_argument("--depformer_multi_linear", action="store_true", default=False)
    ap.add_argument("--depformer_weights_per_step", action="store_true", default=False)
    ap.add_argument("--depformer_norm", action="store_true", default=False)
    ap.add_argument("--prev_depth_output_dropout", type=float, default=0.0,
                    help="Dropout on the previous-codebook embedding fed into "
                         "the next depth-layer prediction (AR coupling branch "
                         "only). Does NOT touch dep_in(h) nor BOS at k=0. "
                         "Consumed only when "
                         "--acoustic_predictor_type=depth_transformer.")
    # MOSS-TTS-Local depth predictor knobs (consumed only when
    # --acoustic_predictor_type=moss_local_depth). Defaults match
    # MossTTSLocalConfig.
    ap.add_argument("--moss_local_hidden", type=int, default=1536)
    ap.add_argument("--moss_local_num_heads", type=int, default=16)
    ap.add_argument("--moss_local_num_layers", type=int, default=4)
    ap.add_argument("--moss_local_ffn", type=int, default=8960)
    ap.add_argument("--moss_local_bridge_ffn", type=int, default=2048,
                    help="SwiGLU MLP bridge ffn dim (MOSS's "
                         "additional_mlp_ffn_hidden_size).")
    # Audio<->text fusion + audio-embedding init (ablation knobs).
    ap.add_argument("--fusion_mode", choices=["gated_softmax", "sum"],
                    default="gated_softmax",
                    help="LLM-input fusion: legacy gated softmax (default) or "
                         "Moshi-style direct sum (Linear(audio) + text, no gate).")
    ap.add_argument("--audio_emb_init_scale", default="auto",
                    help='Per-codebook acoustic embedding init std. "auto" '
                         '=> 0.02/sqrt(R) (legacy). Float string overrides '
                         '(e.g. "0.02" for Moshi-like single-codebook scale).')
    ap.add_argument("--rvq_loss_weights", default="",
                    help="Comma-separated length-R per-codebook training "
                         "weights (e.g. '4,1,1,...,1'). Empty = uniform. Val "
                         "loss_acoustic is always uniform for fair compare.")
    # Duration ↔ acoustic coupling (Exp A / Exp B).
    ap.add_argument("--duration_head_input",
                    choices=["llm", "depformer_last"], default="llm",
                    help="Source of duration-head features. 'llm' (default): "
                         "LLM hidden state h. 'depformer_last': depth "
                         "transformer's u^{R-1} output (Exp A; requires "
                         "acoustic_predictor_type=depth_transformer).")
    ap.add_argument("--duration_loss_type",
                    choices=["regression", "classification"], default="regression",
                    help="'regression': L1/MSE on log1p(frames). "
                         "'classification': CE over duration_num_buckets "
                         "linear frame buckets (Exp B).")
    ap.add_argument("--duration_num_buckets", type=int, default=256,
                    help="Number of linear frame buckets when "
                         "duration_loss_type=classification. bucket = "
                         "clamp(frames, 0, N-1).")
    ap.add_argument("--feed_duration_to_depformer", action="store_true",
                    default=False,
                    help="Embed duration bucket and add to every depth-"
                         "position input z^k (Exp B). Requires "
                         "depth_transformer + classification.")
    # Continuous-acoustic head (replaces per-codebook CE with regression to a
    # teacher pre-quantizer feature).
    ap.add_argument("--acoustic_target",
                    choices=["rvq", "continuous"], default="rvq",
                    help="'rvq' (default): per-codebook CE on q_codes. "
                         "'continuous': regress LLM hidden -> teacher pre-quantizer "
                         "feature (cache must contain pre_quant_feat).")
    ap.add_argument("--acoustic_feat_dim", type=int, default=256,
                    help="Continuous-acoustic head output dim. Must match "
                         "cached pre_quant_feat last dim (256 for StreamAlign).")
    ap.add_argument("--acoustic_feat_loss",
                    choices=["l1", "mse", "cosine", "mse_l1"], default="l1",
                    help="Loss for continuous acoustic regression. mse_l1 = "
                         "0.5*MSE + 0.5*L1 (TASTE-style equal-weight average).")
    ap.add_argument("--acoustic_layer_mix",
                    choices=["last", "weighted"], default="last",
                    help="Hidden-state source for the acoustic + duration "
                         "heads. 'last' = out.hidden_states[-1] (legacy). "
                         "'weighted' = TASTE-style softmax mix over all "
                         "(num_hidden_layers + 1) hidden states. Text head "
                         "is unaffected (always reads out.logits).")
    # Model architecture variant. Default is the legacy StreamSLM; switch to
    # "streamslm_hier" for the Moshi-style step-internal depth-AR (conditions
    # q_n / d_n on the already-sampled w_{n+1}).
    ap.add_argument("--model_arch",
                    choices=["streamslm", "streamslm_hier"], default="streamslm",
                    help="Which model class to instantiate. 'streamslm' "
                         "(default, legacy) has parallel heads off h_n. "
                         "'streamslm_hier' moves (q_n, d_n) into a depth "
                         "transformer that conditions on the next text "
                         "token w_{n+1}.")
    ap.add_argument("--hier_ar_order",
                    choices=["duration_first", "duration_last"],
                    default="duration_last",
                    help="Ordering of the duration slot vs the R acoustic "
                         "slots in the hier depth-AR. Ignored when "
                         "--model_arch=streamslm.")
    ap.add_argument("--teacher_rvq_ckpt", default="",
                    help="Optional StreamAlign teacher .pt. When set AND "
                         "acoustic_target='continuous', validation re-quantises "
                         "the predicted feature through the teacher's "
                         "ResidualVQ to recover (B,T,R) indices and reports "
                         "per-codebook accuracy vs ground-truth q_codes "
                         "(layers l0/l4/l8/l15). Loaded once on rank 0.")
    ap.add_argument("--teacher_rvq_acc_layers", default="0,4,8,15",
                    help="Comma-separated codebook layer indices to log "
                         "accuracy for under continuous + teacher_rvq_ckpt. "
                         "Layers >= R are ignored.")
    # train
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    # Sequence packing: concatenate utts into B=1 long sequences with
    # position_ids resets to eliminate padding waste. Requires FA2 (or SDPA);
    # the model's forward switches to attention_mask=None when position_ids
    # is provided so create_causal_mask routes through the packed-sequence
    # path. Validation always runs unpacked for comparable metrics.
    ap.add_argument("--packing", action="store_true", default=False,
                    help="Pack utterances into B=1 long sequences. Overrides "
                         "--batch_size for the train loader (set to 1).")
    ap.add_argument("--pack_max_tokens", type=int, default=4096,
                    help="Max packed-sequence length per pack. Effective "
                         "subword budget per fwd is this value (no padding).")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr_step_at", type=int, default=-1,
                    help="If >0, switch lr to --lr_step_to once step >= this.")
    ap.add_argument("--lr_step_to", type=float, default=0.0,
                    help="Target lr after --lr_step_at. Constant from then on.")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=1000)
    ap.add_argument("--max_steps", type=int, default=50_000)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--no_bf16", dest="bf16", action="store_false")
    ap.add_argument("--save_every", type=int, default=2_000)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--val_fraction", type=float, default=0.005,
                    help="Fraction of training data held out as validation. "
                         "Set to 0 to disable val. Ignored when --val_manifest "
                         "is provided.")
    ap.add_argument("--val_manifest", nargs="*", default=None,
                    help="One or more manifest CSVs / globs / dirs for a "
                         "fixed validation set (e.g. LibriSpeech dev-clean). "
                         "When provided, training data is taken in full from "
                         "--manifest and validation uses --val_manifest "
                         "instead of a random val_fraction split.")
    ap.add_argument("--val_every", type=int, default=500,
                    help="Run validation every N optimizer steps.")
    ap.add_argument("--val_max_batches", type=int, default=50,
                    help="Cap validation passes per eval to bound runtime.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", default="", help="Path to .pt to resume from")
    ap.add_argument("--resume_model_only", action="store_true",
                    help="Warm-start: load only model weights from --resume "
                         "(strict=False), with a fresh optimizer and "
                         "start_step=0. Use to fork from a different-arch "
                         "ckpt; matched-shape keys load, the rest reset.")
    ap.add_argument("--eval_only", action="store_true",
                    help="Load --resume, run validation once, print metrics, exit. "
                         "Use --val_max_batches to bound runtime; pass 0 = full set.")
    args = ap.parse_args()

    rank, world, local_rank, is_dist = _maybe_init_dist()
    torch.manual_seed(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    t_init_start = time.perf_counter()

    def _stamp(label: str, t0: float) -> None:
        if _is_main(rank):
            print(f"[init] {label} {time.perf_counter() - t0:.1f}s", flush=True)

    out_dir = Path(args.out_dir)
    tb_writer: Optional[SummaryWriter] = None
    if _is_main(rank):
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)
        tb_writer = SummaryWriter(log_dir=str(out_dir / "tb"))

    # ----- configs
    tok_cfg = TokenizerConfig(
        token_type=args.token_type,
        rvq_num_quantizers=args.rvq_num_quantizers,
        rvq_codebook_size=args.rvq_codebook_size,
        text_tokenizer=args.text_tokenizer,
    )
    rvq_w_list = None
    if args.rvq_loss_weights.strip():
        rvq_w_list = [float(x) for x in args.rvq_loss_weights.split(",")]
    model_cfg = ModelConfig(
        backbone=args.backbone,
        delay=args.delay,
        duration_loss=args.duration_loss,
        loss_w_text=args.loss_w_text,
        loss_w_acoustic=args.loss_w_acoustic,
        loss_w_duration=args.loss_w_duration,
        acoustic_predictor_type=args.acoustic_predictor_type,
        depformer_dim=args.depformer_dim,
        depformer_num_heads=args.depformer_num_heads,
        depformer_num_layers=args.depformer_num_layers,
        depformer_dim_feedforward=(args.depformer_dim_feedforward or None),
        depformer_multi_linear=args.depformer_multi_linear,
        depformer_weights_per_step=args.depformer_weights_per_step,
        depformer_norm=args.depformer_norm,
        prev_depth_output_dropout=args.prev_depth_output_dropout,
        moss_local_hidden=args.moss_local_hidden,
        moss_local_num_heads=args.moss_local_num_heads,
        moss_local_num_layers=args.moss_local_num_layers,
        moss_local_ffn=args.moss_local_ffn,
        moss_local_bridge_ffn=args.moss_local_bridge_ffn,
        fusion_mode=args.fusion_mode,
        audio_emb_init_scale=args.audio_emb_init_scale,
        rvq_loss_weights=rvq_w_list,
        duration_head_input=args.duration_head_input,
        duration_loss_type=args.duration_loss_type,
        duration_num_buckets=args.duration_num_buckets,
        feed_duration_to_depformer=args.feed_duration_to_depformer,
        acoustic_target=args.acoustic_target,
        acoustic_feat_dim=args.acoustic_feat_dim,
        acoustic_feat_loss=args.acoustic_feat_loss,
        acoustic_layer_mix=args.acoustic_layer_mix,
        text_kl_weight=args.text_kl_weight,
        model_arch=args.model_arch,
        hier_ar_order=args.hier_ar_order,
    )

    # ----- data
    t_data = time.perf_counter()
    lm_tok = load_lm_tokenizer(tok_cfg)
    pad_id = lm_tok.pad_token_id
    if pad_id is None:
        pad_id = lm_tok.eos_token_id
    if _is_main(rank):
        print(f"[data] pad_token_id={pad_id} ({lm_tok.pad_token!r})")

    full_ds = SubwordUnitsDataset(
        args.manifest,
        min_subwords=args.min_subwords,
        max_subwords=args.max_subwords,
        return_pre_quant_feat=(args.acoustic_target == "continuous"),
    )

    # Validation set: either a fixed external manifest (--val_manifest, e.g.
    # dev-clean) or a random val_fraction split of the training manifests.
    # Validation always runs unpacked (PadCollator), independently of
    # --packing, so val metrics stay comparable across configs.
    val_ds = None
    if args.val_manifest:
        train_unpacked = full_ds
        val_ds = SubwordUnitsDataset(
            args.val_manifest,
            min_subwords=args.min_subwords,
            max_subwords=args.max_subwords,
            return_pre_quant_feat=(args.acoustic_target == "continuous"),
        )
    elif args.val_fraction and args.val_fraction > 0:
        # Deterministic train/val split (same across ranks since seeded identically).
        n_total = len(full_ds)
        n_val = max(1, int(round(n_total * args.val_fraction)))
        gen = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(n_total, generator=gen).tolist()
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        train_unpacked = torch.utils.data.Subset(full_ds, train_idx)
        val_ds = torch.utils.data.Subset(full_ds, val_idx)
    else:
        train_unpacked = full_ds
    if _is_main(rank):
        n_val = len(val_ds) if val_ds is not None else 0
        val_src = "manifest" if args.val_manifest else (
            "split" if args.val_fraction > 0 else "none")
        print(f"[data] {len(full_ds)} utterances after length filter "
              f"(train={len(train_unpacked)} val={n_val} val_src={val_src})")
    _stamp("data setup", t_data)

    if args.packing:
        # Build the packed train dataset from the same manifest list. We don't
        # apply the val_fraction split to the packed dataset (it would re-read
        # the CSVs); instead we accept that with --val_fraction>0 some val
        # utts will leak into the packed train stream. For benchmarks /
        # external --val_manifest the train/val sets are disjoint already.
        if args.val_fraction and args.val_fraction > 0 and not args.val_manifest:
            if _is_main(rank):
                print("[data] WARNING: --packing + --val_fraction>0 may overlap "
                      "train/val; prefer --val_manifest for clean splits.",
                      flush=True)
        packed_ds = PackedSubwordUnitsDataset(
            args.manifest,
            min_subwords=args.min_subwords,
            max_subwords=args.max_subwords,
            pack_max_tokens=args.pack_max_tokens,
            return_pre_quant_feat=(args.acoustic_target == "continuous"),
            seed=args.seed,
            world=world,
            rank=rank,
            delay=args.delay,
            eos_token_id=lm_tok.eos_token_id,
            pad_token_id=pad_id,
        )
        if _is_main(rank):
            print(f"[data] packing on: pack_max_tokens={args.pack_max_tokens} "
                  f"utts={packed_ds.num_utterances()} (B forced to 1)",
                  flush=True)
        sampler = None
        loader = DataLoader(
            packed_ds,
            batch_size=1,
            shuffle=False,
            sampler=None,
            num_workers=args.num_workers,
            collate_fn=PackingCollator(),
            pin_memory=True,
            drop_last=False,
            persistent_workers=args.num_workers > 0,
        )
    else:
        sampler = (
            DistributedSampler(train_unpacked, num_replicas=world, rank=rank, shuffle=True)
            if is_dist else None
        )
        loader = DataLoader(
            train_unpacked,
            batch_size=args.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=PadCollator(
                pad_token_id=pad_id,
                eos_token_id=lm_tok.eos_token_id,
                delay=args.delay,
            ),
            pin_memory=True,
            drop_last=True,
            persistent_workers=args.num_workers > 0,
        )

    val_loader = None
    if val_ds is not None and _is_main(rank):
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=max(1, args.num_workers // 2),
            collate_fn=PadCollator(
                pad_token_id=pad_id,
                eos_token_id=lm_tok.eos_token_id,
                delay=args.delay,
            ),
            pin_memory=True,
            drop_last=False,
            persistent_workers=False,
        )

    # ----- model
    # When --resume (full mode) is set, the ckpt's state_dict overwrites the
    # backbone weights anyway — skip the 2.4 GB from_pretrained NFS read and
    # build an empty scaffold from config instead. For --resume_model_only we
    # still load HF weights because strict=False may leave some backbone
    # tensors un-restored from the (possibly forked-arch) ckpt.
    skip_backbone_weights = bool(args.resume) and not args.resume_model_only
    t_model = time.perf_counter()
    model_cls = StreamSLMHier if args.model_arch == "streamslm_hier" else StreamSLM
    model = model_cls(
        model_cfg, tok_cfg, skip_backbone_weights=skip_backbone_weights,
    ).to(device)
    if _is_main(rank):
        n_params = sum(p.numel() for p in model.parameters())
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        bb_src = "from_config (skip HF weights, ckpt restores)" if skip_backbone_weights else "from_pretrained"
        arch_tag = args.model_arch + (
            f"({args.hier_ar_order})" if args.model_arch == "streamslm_hier" else ""
        )
        print(f"[model] arch={arch_tag} params={n_params/1e6:.1f}M trainable={n_train/1e6:.1f}M "
              f"backbone={args.backbone} delay={args.delay} bb={bb_src}")
    _stamp("model construct", t_model)

    if is_dist:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    optimizer = _build_optimizer(
        model.module if is_dist else model, args.lr, args.weight_decay
    )

    # ----- frozen ref_model for TASTE stage-2 KL distillation
    # Loaded AFTER optimizer so its params are never iterated; mirrors
    # TASTE-SpokenLM/run.py:451-456 + TasteTrainer.set_ref_model.
    # Skip the load entirely when --text_kl_weight 0 — the model's
    # compute_loss collapses text_loss to plain CE in that branch, so the
    # ref_model would just burn ~5 GB and ~half the per-step FLOPs for
    # nothing.
    ref_model = None
    t_ref = time.perf_counter()
    if args.ref_model and args.text_kl_weight > 0:
        from transformers import LlamaForCausalLM
        from streamSLM.model.slm import _maybe_apply_liger
        _maybe_apply_liger(args.ref_model)
        ref_model = LlamaForCausalLM.from_pretrained(
            args.ref_model,
            torch_dtype=torch.bfloat16,
            attn_implementation=os.environ.get(
                "STREAMSLM_ATTN_IMPL", "flash_attention_2"
            ),
        )
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
        ref_model = ref_model.to(device)
        if _is_main(rank):
            n_ref = sum(p.numel() for p in ref_model.parameters())
            n_ref_train = sum(p.numel() for p in ref_model.parameters() if p.requires_grad)
            print(f"[ref] {args.ref_model} bf16 params={n_ref/1e6:.1f}M "
                  f"trainable={n_ref_train/1e6:.1f}M (must be 0) "
                  f"text_kl_weight={args.text_kl_weight}")
    elif _is_main(rank):
        if args.ref_model and args.text_kl_weight == 0:
            print(f"[ref] skipping {args.ref_model} load "
                  f"(--text_kl_weight=0 drops KL term)")
        else:
            print("[ref] disabled (no ref_model)")
    _stamp("ref_model load", t_ref)

    # ----- amp
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float32
    use_amp = args.bf16 and torch.cuda.is_available()

    # ----- resume
    # Rank-0 reads the (8 GB-ish) ckpt from NFS once; model params + optim
    # state then broadcast to other ranks via NCCL. Cuts NFS read pressure
    # by N for N-rank DDP — important when several trainings launch in
    # parallel and contend for the same NFS bandwidth.
    start_step = 0
    if args.resume:
        t_resume = time.perf_counter()
        mode = "model-only fork" if args.resume_model_only else "full"
        if _is_main(rank):
            print(f"[resume] loading {args.resume}  mode={mode}", flush=True)

        inner = model.module if is_dist else model

        ckpt = None
        if not is_dist or rank == 0:
            t_read = time.perf_counter()
            ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
            if _is_main(rank):
                print(f"[resume] rank0 read ckpt in "
                      f"{time.perf_counter() - t_read:.1f}s", flush=True)

        if not is_dist or rank == 0:
            if args.resume_model_only:
                result = inner.load_state_dict(ckpt["model"], strict=False)
                if _is_main(rank):
                    missing = list(result.missing_keys)
                    unexpected = list(result.unexpected_keys)
                    print(f"[resume] strict=False: missing={len(missing)} unexpected={len(unexpected)}")
                    if missing:
                        print(f"[resume] missing[:5]={missing[:5]}")
                    if unexpected:
                        print(f"[resume] unexpected[:5]={unexpected[:5]}")
            else:
                inner.load_state_dict(ckpt["model"])
                optimizer.load_state_dict(ckpt["optim"])
                start_step = int(ckpt.get("step", 0))

        if is_dist:
            t_bcast = time.perf_counter()
            # Broadcast model params + buffers from rank 0 → others. Params
            # are already on GPU (model.to(device) at construction), so NCCL
            # broadcast is direct.
            for p in inner.parameters():
                dist.broadcast(p.data, src=0)
            for b in inner.buffers():
                # Some buffers (e.g. integer counters) may not be float.
                # NCCL handles fp32/fp16/bf16/int directly; pass through.
                dist.broadcast(b.data, src=0)

            if not args.resume_model_only:
                # Broadcast start_step.
                step_t = torch.tensor([start_step], dtype=torch.long, device=device)
                dist.broadcast(step_t, src=0)
                start_step = int(step_t.item())

                # Broadcast optimizer state (CPU tensors; pickled + tunneled
                # via NCCL byte tensor — temporarily ~5–7 GB on GPU, OK on
                # 48 GB cards).
                obj = [optimizer.state_dict() if rank == 0 else None]
                dist.broadcast_object_list(obj, src=0)
                if rank != 0:
                    optimizer.load_state_dict(obj[0])
                obj = None  # release the temporary state dict
            if _is_main(rank):
                print(f"[resume] broadcast (params+buffers"
                      f"{'' if args.resume_model_only else '+optim'}) in "
                      f"{time.perf_counter() - t_bcast:.1f}s", flush=True)

        ckpt = None  # release rank-0 ckpt mem
        _stamp("resume total", t_resume)

    # ----- val helper
    # Lazy-loaded teacher RVQ for continuous-acoustic re-quantisation in val.
    # Only rank 0 ever runs validation, so loaded inside run_validation().
    teacher_rvq_state: dict = {"loaded": False, "module": None, "layers": ()}

    @torch.no_grad()
    def run_validation() -> Optional[dict]:
        if val_loader is None:
            return None
        was_training = model.training
        model.eval()
        sums = {"loss": 0.0, "loss_text": 0.0, "loss_text_ce": 0.0, "loss_text_kl": 0.0,
                "loss_acoustic": 0.0, "loss_duration": 0.0}
        acc_counts = {"text": [0, 0], "dur": [0, 0]}
        aco_layers_seen: list[int] = []
        IGNORE = -100
        n_batches = 0
        for vb_idx, vb in enumerate(val_loader):
            if args.val_max_batches and vb_idx >= args.val_max_batches:
                break
            sw = vb["subword_ids"].to(device, non_blocking=True)
            qc = vb["q_codes"].to(device, non_blocking=True)
            df = vb["duration_frames"].to(device, non_blocking=True)
            am = vb["attention_mask"].to(device, non_blocking=True)
            text_lm = vb["text_label_mask"].to(device, non_blocking=True).to(torch.bool)
            aco_lm = vb["aco_label_mask"].to(device, non_blocking=True).to(torch.bool)
            pqf = (
                vb["pre_quant_feat"].to(device, non_blocking=True)
                if "pre_quant_feat" in vb else None
            )
            inner_ = model.module if is_dist else model
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                out_ = model(sw, qc, df, am)
                losses_ = inner_.compute_loss(
                    out_, sw, qc, df, am,
                    ref_model=ref_model, pre_quant_feat=pqf,
                    text_label_mask=text_lm, aco_label_mask=aco_lm,
                )
            for k in sums:
                sums[k] += float(losses_[k].detach())

            # Symmetric-delay: text_label_mask is over *prediction* positions
            # [0, L_i-1]. text_logits at pos n predicts stored token at n+1, so
            # accuracy is gated on text_lm[:, :T-1] and compared against sw[..., 1:].
            T_full = sw.size(1)
            text_logits = out_["text_logits"][..., :-1, :]
            shift_mask = text_lm[..., : T_full - 1]
            text_labels = sw[..., 1:].masked_fill(~shift_mask, IGNORE)
            text_pred = text_logits.argmax(dim=-1)
            text_valid = (text_labels != IGNORE)
            acc_counts["text"][0] += int(((text_pred == text_labels) & text_valid).sum().item())
            acc_counts["text"][1] += int(text_valid.sum().item())

            # Symmetric-delay: acoustic targets are right-shifted by D-1 so the
            # model at position n predicts q_{n-(D-1)}. Gate via aco_label_mask.
            aco_mask_b = aco_lm
            shift_pad = inner_.delay - 1
            if shift_pad > 0:
                qc_shifted = torch.nn.functional.pad(qc, (0, 0, shift_pad, 0))[:, :T_full, :]
                df_shifted = torch.nn.functional.pad(df, (shift_pad, 0))[:, :T_full]
            else:
                qc_shifted = qc
                df_shifted = df
            if "acoustic_logits" in out_:
                aco_pred = out_["acoustic_logits"].argmax(dim=-1)  # (B, T, R)
                R_dim = aco_pred.size(-1)
                if not aco_layers_seen:
                    aco_layers_seen = list(range(R_dim))
                    for r in aco_layers_seen:
                        acc_counts[f"aco_l{r}"] = [0, 0]
                for r in aco_layers_seen:
                    correct_r = ((aco_pred[..., r] == qc_shifted[..., r]) & aco_mask_b)
                    acc_counts[f"aco_l{r}"][0] += int(correct_r.sum().item())
                    acc_counts[f"aco_l{r}"][1] += int(aco_mask_b.sum().item())
            elif (
                "acoustic_feat_pred" in out_
                and args.teacher_rvq_ckpt
                and inner_.acoustic_target == "continuous"
            ):
                # Continuous-acoustic eval: re-quantise the predicted feature
                # through the teacher's ResidualVQ to recover (B,T,R) indices,
                # then compute per-codebook accuracy vs ground-truth q_codes.
                if not teacher_rvq_state["loaded"]:
                    print(
                        f"[teacher_rvq] loading {args.teacher_rvq_ckpt} "
                        f"(R={inner_.R} C={args.rvq_codebook_size} "
                        f"feat_dim={inner_.acoustic_feat_dim})",
                        flush=True,
                    )
                    t_rvq = time.perf_counter()
                    teacher_rvq_state["module"] = _load_teacher_residual_vq(
                        args.teacher_rvq_ckpt,
                        num_quantizers=inner_.R,
                        codebook_size=args.rvq_codebook_size,
                        feat_dim=inner_.acoustic_feat_dim,
                        device=device,
                    )
                    requested = [
                        int(s) for s in args.teacher_rvq_acc_layers.split(",")
                        if s.strip()
                    ]
                    teacher_rvq_state["layers"] = tuple(
                        r for r in requested if 0 <= r < inner_.R
                    )
                    teacher_rvq_state["loaded"] = True
                    print(
                        f"[teacher_rvq] loaded in "
                        f"{time.perf_counter() - t_rvq:.1f}s "
                        f"layers_logged={list(teacher_rvq_state['layers'])}",
                        flush=True,
                    )
                for r in teacher_rvq_state["layers"]:
                    acc_counts.setdefault(f"aco_l{r}", [0, 0])
                feat_pred = out_["acoustic_feat_pred"].float()  # (B, T, D)
                B_, T_, D_ = feat_pred.shape
                _q, q_idx, _commit = teacher_rvq_state["module"](
                    feat_pred.reshape(B_ * T_, D_)
                )
                q_idx = q_idx.reshape(B_, T_, -1)  # (B, T, R)
                for r in teacher_rvq_state["layers"]:
                    correct_r = ((q_idx[..., r] == qc_shifted[..., r]) & aco_mask_b)
                    acc_counts[f"aco_l{r}"][0] += int(correct_r.sum().item())
                    acc_counts[f"aco_l{r}"][1] += int(aco_mask_b.sum().item())

            dur_pred = out_["duration_pred"]
            if inner_.duration_loss_type == "classification":
                dur_pred_int = dur_pred.argmax(dim=-1)
                dur_target_int = df_shifted.clamp(
                    min=0, max=inner_.duration_num_buckets - 1
                ).long()
            else:
                dur_pred_int = torch.expm1(dur_pred.float()).clamp(min=0).round().long()
                dur_target_int = df_shifted.long()
            acc_counts["dur"][0] += int(
                ((dur_pred_int == dur_target_int) & aco_mask_b).sum().item()
            )
            acc_counts["dur"][1] += int(aco_mask_b.sum().item())

            n_batches += 1
        if was_training:
            model.train()
        if n_batches == 0:
            return None
        out = {k: v / n_batches for k, v in sums.items()}
        for k, (c, n) in acc_counts.items():
            out[f"acc_{k}"] = (c / n) if n > 0 else 0.0
        return out

    if args.eval_only:
        if _is_main(rank):
            ckpt_tag = Path(args.resume).name if args.resume else "<no-resume>"
            vstats = run_validation()
            if vstats is None:
                print(f"[eval_only] {ckpt_tag}: no val batches", flush=True)
            else:
                aco_layer_idxs = sorted(
                    int(k[len("acc_aco_l"):]) for k in vstats if k.startswith("acc_aco_l")
                )
                aco_acc_parts = [f"l{r}={vstats[f'acc_aco_l{r}']:.4f}" for r in aco_layer_idxs]
                aco_acc_str = " ".join(aco_acc_parts) if aco_acc_parts else "n/a"
                print(
                    f"[eval_only {ckpt_tag} step={start_step}] "
                    f"loss={vstats['loss']:.4f} "
                    f"text={vstats['loss_text']:.4f} "
                    f"(ce={vstats['loss_text_ce']:.4f} kl={vstats['loss_text_kl']:.4f}) "
                    f"aco={vstats['loss_acoustic']:.4f} "
                    f"dur={vstats['loss_duration']:.4f}  "
                    f"acc: text={vstats['acc_text']:.4f} "
                    f"aco({aco_acc_str}) "
                    f"dur={vstats['acc_dur']:.4f}",
                    flush=True,
                )
        if is_dist:
            dist.barrier()
            dist.destroy_process_group()
        return

    # ----- train loop
    _stamp("init total", t_init_start)
    model.train()
    step = start_step
    accum = 0
    t0 = time.time()
    running = {"loss": 0.0, "loss_text": 0.0, "loss_text_ce": 0.0, "loss_text_kl": 0.0,
               "loss_acoustic": 0.0, "loss_duration": 0.0}
    running_n = 0
    running_gn_sum = 0.0
    running_gn_max = 0.0
    running_gn_n = 0
    nonfinite_skipped = 0
    epoch = 0
    optimizer.zero_grad(set_to_none=True)

    while step < args.max_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            subword_ids = batch["subword_ids"].to(device, non_blocking=True)
            q_codes = batch["q_codes"].to(device, non_blocking=True)
            duration_frames = batch["duration_frames"].to(device, non_blocking=True)
            attn = batch["attention_mask"].to(device, non_blocking=True)
            text_label_mask = batch["text_label_mask"].to(device, non_blocking=True).to(torch.bool)
            aco_label_mask = batch["aco_label_mask"].to(device, non_blocking=True).to(torch.bool)
            pos_ids = (
                batch["position_ids"].to(device, non_blocking=True)
                if "position_ids" in batch else None
            )
            pre_quant_feat = (
                batch["pre_quant_feat"].to(device, non_blocking=True)
                if "pre_quant_feat" in batch else None
            )

            inner = model.module if is_dist else model
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = model(
                    subword_ids, q_codes, duration_frames, attn,
                    position_ids=pos_ids,
                )
                losses = inner.compute_loss(
                    outputs, subword_ids, q_codes, duration_frames, attn,
                    ref_model=ref_model, pre_quant_feat=pre_quant_feat,
                    position_ids=pos_ids,
                    text_label_mask=text_label_mask,
                    aco_label_mask=aco_label_mask,
                )
                loss = losses["loss"] / args.grad_accum

            loss.backward()
            accum += 1

            for k in running:
                running[k] += float(losses[k].detach())
            running_n += 1

            if accum < args.grad_accum:
                continue

            # ---- optimizer step
            for g in optimizer.param_groups:
                g["lr"] = _lr_at(step, args.lr, args.warmup_steps, args.max_steps,
                                 step_at=args.lr_step_at, step_to=args.lr_step_to)
            if args.grad_clip > 0:
                gn = torch.nn.utils.clip_grad_norm_(
                    (model.module if is_dist else model).parameters(), args.grad_clip
                )
            else:
                gn = torch.nn.utils.clip_grad_norm_(
                    (model.module if is_dist else model).parameters(), float("inf")
                )
            gn_val = float(gn) if torch.isfinite(gn) else float("inf")
            if not torch.isfinite(gn):
                nonfinite_skipped += 1
                optimizer.zero_grad(set_to_none=True)
                accum = 0
                step += 1
                if _is_main(rank):
                    print(f"[step {step:>6d}] non-finite grad_norm — skipping optimizer step",
                          flush=True)
                continue
            running_gn_sum += gn_val
            running_gn_max = max(running_gn_max, gn_val)
            running_gn_n += 1
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            accum = 0

            # ---- log
            if _is_main(rank) and step % args.log_every == 0:
                dt = time.time() - t0
                rate = running_n / max(dt, 1e-6)
                lr_now = optimizer.param_groups[0]["lr"]
                avg = {k: running[k] / running_n for k in running}
                gn_avg = (running_gn_sum / running_gn_n) if running_gn_n > 0 else 0.0
                gn_max = running_gn_max
                msg = (f"[step {step:>6d}] "
                       f"loss={avg['loss']:.4f} "
                       f"text={avg['loss_text']:.4f} "
                       f"(ce={avg['loss_text_ce']:.4f} kl={avg['loss_text_kl']:.4f}) "
                       f"aco={avg['loss_acoustic']:.4f} "
                       f"dur={avg['loss_duration']:.4f} "
                       f"gn(avg/max)={gn_avg:.2f}/{gn_max:.2f} "
                       f"skip={nonfinite_skipped} "
                       f"lr={lr_now:.2e} {rate:.1f}fwd/s")
                print(msg, flush=True)
                if tb_writer is not None:
                    tb_writer.add_scalar("loss/total", avg["loss"], step)
                    tb_writer.add_scalar("loss/text", avg["loss_text"], step)
                    tb_writer.add_scalar("loss/text_ce", avg["loss_text_ce"], step)
                    tb_writer.add_scalar("loss/text_kl", avg["loss_text_kl"], step)
                    tb_writer.add_scalar("loss/acoustic", avg["loss_acoustic"], step)
                    tb_writer.add_scalar("loss/duration", avg["loss_duration"], step)
                    tb_writer.add_scalar("train/lr", lr_now, step)
                    tb_writer.add_scalar("train/fwd_per_sec", rate, step)
                    tb_writer.add_scalar("train/grad_norm_avg", gn_avg, step)
                    tb_writer.add_scalar("train/grad_norm_max", gn_max, step)
                    tb_writer.add_scalar("train/nonfinite_skipped", nonfinite_skipped, step)
                running = {k: 0.0 for k in running}
                running_n = 0
                running_gn_sum = 0.0
                running_gn_max = 0.0
                running_gn_n = 0
                t0 = time.time()

            # ---- validate
            if (
                _is_main(rank)
                and val_loader is not None
                and args.val_every > 0
                and step % args.val_every == 0
            ):
                vstats = run_validation()
                if vstats is not None:
                    aco_layer_idxs = sorted(
                        int(k[len("acc_aco_l"):]) for k in vstats
                        if k.startswith("acc_aco_l")
                    )
                    aco_acc_parts = [
                        f"l{r}={vstats[f'acc_aco_l{r}']:.3f}"
                        for r in aco_layer_idxs
                    ]
                    aco_acc_str = " ".join(aco_acc_parts) if aco_acc_parts else "n/a"
                    print(
                        f"[val   {step:>6d}] "
                        f"loss={vstats['loss']:.4f} "
                        f"text={vstats['loss_text']:.4f} "
                        f"(ce={vstats['loss_text_ce']:.4f} kl={vstats['loss_text_kl']:.4f}) "
                        f"aco={vstats['loss_acoustic']:.4f} "
                        f"dur={vstats['loss_duration']:.4f}  "
                        f"acc: text={vstats['acc_text']:.3f} "
                        f"aco({aco_acc_str}) "
                        f"dur={vstats['acc_dur']:.3f}",
                        flush=True,
                    )
                    if tb_writer is not None:
                        tb_writer.add_scalar("val/loss_total", vstats["loss"], step)
                        tb_writer.add_scalar("val/loss_text", vstats["loss_text"], step)
                        tb_writer.add_scalar("val/loss_text_ce", vstats["loss_text_ce"], step)
                        tb_writer.add_scalar("val/loss_text_kl", vstats["loss_text_kl"], step)
                        tb_writer.add_scalar("val/loss_acoustic", vstats["loss_acoustic"], step)
                        tb_writer.add_scalar("val/loss_duration", vstats["loss_duration"], step)
                        tb_writer.add_scalar("val/acc_text", vstats["acc_text"], step)
                        tb_writer.add_scalar("val/acc_dur", vstats["acc_dur"], step)
                        for r in (0, 4, 8):
                            key = f"acc_aco_l{r}"
                            if key in vstats:
                                tb_writer.add_scalar(f"val/acc_aco_l{r}", vstats[key], step)

            # ---- save
            if _is_main(rank) and (step % args.save_every == 0 or step == args.max_steps):
                ckpt_path = out_dir / f"step_{step:08d}.pt"
                torch.save({
                    "step": step,
                    "model": (model.module if is_dist else model).state_dict(),
                    "optim": optimizer.state_dict(),
                    "args": vars(args),
                }, ckpt_path)
                latest = out_dir / "latest.pt"
                if latest.exists() or latest.is_symlink():
                    latest.unlink()
                latest.symlink_to(ckpt_path.name)
                print(f"[save] {ckpt_path}", flush=True)

            if step >= args.max_steps:
                break
        epoch += 1

    if _is_main(rank) and torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"[mem] peak_alloc={peak:.2f} GiB", flush=True)

    if is_dist:
        dist.barrier()
        dist.destroy_process_group()
    if _is_main(rank):
        if tb_writer is not None:
            tb_writer.close()
        print("[done]")


if __name__ == "__main__":
    main()
