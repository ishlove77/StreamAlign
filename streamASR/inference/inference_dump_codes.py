#!/usr/bin/env python3
"""Prod-style RVQ inference + dump per-layer code histograms.

Wraps inference_stream.py: monkey-patches the RVQ forward to
collect ``indices`` from every call, then prints codebook-usage stats per
layer at the end. Stats: unique codes used, fraction of codebook entries
covered, top-10 / bottom-10 frequencies, entropy.

Output also saved to <output_dir>/codebook_usage.json next to the flacs.
"""
import importlib.util
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROD_RVQ = os.path.join(_HERE, "inference_stream.py")
_spec = importlib.util.spec_from_file_location("inference_stream", _PROD_RVQ)
_orig = importlib.util.module_from_spec(_spec)
sys.modules["inference_stream"] = _orig
_spec.loader.exec_module(_orig)


# Container shared across model instances
_collected = {"counts": None, "codebook_size": None, "num_layers": None}


def _patch_rvq_forward(rvq_module):
    """Wrap the ResidualVQ.forward so we can record indices per layer."""
    if getattr(rvq_module, "_codedump_patched", False):
        return
    orig_forward = rvq_module.forward

    def patched(x):
        out = orig_forward(x)
        try:
            # Convention used by ResidualVQ adapter: (z_q, codes, _latents, commit, codebook)
            if isinstance(out, (tuple, list)) and len(out) >= 2:
                idx = out[1]
            else:
                return out
            # idx: (B, T, R)
            if idx is None or not torch.is_tensor(idx):
                return out
            if idx.dim() < 2:
                return out
            if idx.dim() == 2:
                idx = idx.unsqueeze(-1)
            num_layers = idx.shape[-1]
            if _collected["counts"] is None:
                _collected["counts"] = [Counter() for _ in range(num_layers)]
                _collected["num_layers"] = num_layers
            arr = idx.detach().cpu().numpy().reshape(-1, num_layers)  # (N, R)
            for i in range(num_layers):
                _collected["counts"][i].update(arr[:, i].tolist())
        except Exception as e:
            print(f"[codedump] hook failed: {e}", flush=True)
        return out

    rvq_module.forward = patched
    rvq_module._codedump_patched = True


# Hook into model construction. The prod_rvq wrapper exposes _RVQModel; patch
# its __init__ so every fresh instance gets the recorder attached after init.
_orig_init = _orig._RVQModel.__init__


def _patched_init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    rvq = getattr(self, "residual_vq", None)
    if rvq is not None:
        # Record codebook size from env (must match training/loading config)
        cb = int(os.environ.get("RVQ_CODEBOOK_SIZE", "512"))
        _collected["codebook_size"] = cb
        _patch_rvq_forward(rvq)


_orig._RVQModel.__init__ = _patched_init


def _summarize_and_save(output_dir):
    """Dump per-rank counts immediately, then (rank0 only) try to merge across
    ranks if rank files exist for ranks 0..world_size-1."""
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    counts = _collected["counts"]
    cb = _collected["codebook_size"] or 0
    if counts is None:
        print(f"[codedump rank{rank}] no codes collected — skipping summary")
        return
    # Per-rank dump
    os.makedirs(output_dir, exist_ok=True)
    rank_path = os.path.join(output_dir, f"codebook_usage.rank{rank}.json")
    rank_payload = {
        "codebook_size": cb,
        "num_layers": len(counts),
        "rank": rank,
        "layers": [
            {"layer": i, "counts": dict(c)} for i, c in enumerate(counts)
        ],
    }
    with open(rank_path, "w") as f:
        json.dump(rank_payload, f)
    print(f"[codedump rank{rank}] wrote {rank_path} (n_layers={len(counts)})")

    if rank != 0:
        return
    # Wait briefly for other ranks (best effort), then merge
    if world > 1:
        import time
        for _ in range(60):  # up to 60s wait
            ranks_done = [
                i for i in range(world)
                if os.path.exists(os.path.join(output_dir, f"codebook_usage.rank{i}.json"))
            ]
            if len(ranks_done) == world: break
            time.sleep(1)
    merged = [Counter() for _ in range(len(counts))]
    for r in range(world):
        rp = os.path.join(output_dir, f"codebook_usage.rank{r}.json")
        if not os.path.exists(rp): continue
        with open(rp) as f:
            d = json.load(f)
        for layer_info in d.get("layers", []):
            i = layer_info["layer"]
            for code, n in layer_info["counts"].items():
                merged[i][int(code)] += int(n)
    counts = merged
    summary = {"codebook_size": cb, "num_layers": len(counts), "layers": []}
    print("\n" + "=" * 78)
    print(f"Codebook usage  |  codebook_size={cb}  num_layers={len(counts)}")
    print("=" * 78)
    print(f"{'layer':>5s} {'unique':>8s} {'frac':>8s} {'entropy':>9s} {'top1_freq':>10s} {'min_freq':>9s}")
    print("-" * 78)
    for i, c in enumerate(counts):
        used = len(c)
        total_calls = sum(c.values())
        if total_calls == 0:
            continue
        probs = np.array(list(c.values()), dtype=np.float64) / total_calls
        ent = float(-np.sum(probs * np.log2(probs + 1e-12)))
        top1 = max(c.values())
        bot1 = min(c.values()) if used else 0
        frac = used / max(1, cb)
        layer_info = {
            "layer": i,
            "unique_used": used,
            "fraction_used": frac,
            "total_uses": total_calls,
            "entropy_bits": ent,
            "max_entropy_bits": float(np.log2(cb)) if cb > 0 else 0.0,
            "top1_freq": top1,
            "min_freq_among_used": bot1,
        }
        summary["layers"].append(layer_info)
        print(f"{i:>5d} {used:>8d} {frac:>8.3f} {ent:>9.3f} {top1:>10d} {bot1:>9d}")
    out_path = os.path.join(output_dir, "codebook_usage.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[codedump] saved {out_path}")


if __name__ == "__main__":
    args = _orig._orig.parse_args()
    if args.output_split is None:
        args.output_split = args.split
    if args.world_size > 1:
        local_rank_env = os.environ.get("LOCAL_RANK")
        if local_rank_env is not None:
            _orig._orig.run(int(local_rank_env), args)
        else:
            torch.multiprocessing.spawn(_orig._orig.run, nprocs=args.world_size, args=(args,))
    else:
        _orig._orig.run(0, args)
    # After single-rank inference, summarize
    _summarize_and_save(args.output_dir)
