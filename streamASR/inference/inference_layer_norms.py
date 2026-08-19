#!/usr/bin/env python3
"""Prod-style RVQ inference + record per-layer L2 norms of q_i.

Hooks each ``residual_vq.layers[i].forward`` and records the mean L2 norm of
the layer's quantized output ``q_i`` over (B, T). Aggregates across all
utterances and ranks (each rank dumps its own JSON; rank0 merges).
"""
import importlib.util
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROD_RVQ = os.path.join(_HERE, "inference_stream.py")
_spec = importlib.util.spec_from_file_location("inference_stream", _PROD_RVQ)
_orig = importlib.util.module_from_spec(_spec)
sys.modules["inference_stream"] = _orig
_spec.loader.exec_module(_orig)


_state = {"sums": None, "counts": None, "num_layers": None}


def _patch_layer_forwards(rvq_module):
    """Wrap each layer's forward to record ||q_i||_2 (averaged over B*T)."""
    layers = getattr(rvq_module, "layers", None)
    if layers is None:
        return
    n = len(layers)
    if _state["sums"] is None:
        _state["sums"] = [0.0] * n
        _state["counts"] = [0] * n
        _state["num_layers"] = n
    for i, layer in enumerate(layers):
        if getattr(layer, "_norm_dump_patched", False):
            continue
        orig = layer.forward
        idx_local = i

        def make(orig, idx_local):
            def patched(x, *a, **kw):
                out = orig(x, *a, **kw)
                try:
                    if isinstance(out, (tuple, list)) and len(out) >= 1:
                        q = out[0]
                    else:
                        q = out
                    if torch.is_tensor(q) and q.dim() >= 2:
                        # L2 norm per (batch, time) frame, then mean
                        flat = q.detach()
                        if flat.dim() == 3:
                            norms = flat.float().norm(dim=-1)  # (B, T)
                        else:
                            norms = flat.float().norm(dim=-1)
                        val = float(norms.mean().item())
                        cnt = int(norms.numel())
                        _state["sums"][idx_local] += val * cnt
                        _state["counts"][idx_local] += cnt
                except Exception as e:
                    print(f"[norm-dump] layer {idx_local} hook failed: {e}", flush=True)
                return out

            return patched

        layer.forward = make(orig, idx_local)
        layer._norm_dump_patched = True


_orig_init = _orig._RVQModel.__init__


def _patched_init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    rvq = getattr(self, "residual_vq", None)
    if rvq is not None:
        _patch_layer_forwards(rvq)


_orig._RVQModel.__init__ = _patched_init


def _summarize_and_save(output_dir):
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    sums = _state["sums"]; counts = _state["counts"]
    if sums is None:
        print(f"[norm-dump rank{rank}] no data — skipping")
        return
    os.makedirs(output_dir, exist_ok=True)
    rank_path = os.path.join(output_dir, f"layer_norms.rank{rank}.json")
    payload = {
        "rank": rank,
        "num_layers": len(sums),
        "sums": sums,
        "counts": counts,
    }
    with open(rank_path, "w") as f:
        json.dump(payload, f)
    print(f"[norm-dump rank{rank}] wrote {rank_path}")
    if rank != 0:
        return
    # rank0: wait briefly for others, then merge
    if world > 1:
        import time
        for _ in range(60):
            done = [
                i for i in range(world)
                if os.path.exists(os.path.join(output_dir, f"layer_norms.rank{i}.json"))
            ]
            if len(done) == world: break
            time.sleep(1)
    merged_sums = list(sums)
    merged_counts = list(counts)
    for r in range(world):
        if r == 0: continue
        rp = os.path.join(output_dir, f"layer_norms.rank{r}.json")
        if not os.path.exists(rp): continue
        with open(rp) as f:
            d = json.load(f)
        for i in range(len(merged_sums)):
            if i < len(d["sums"]):
                merged_sums[i] += d["sums"][i]
                merged_counts[i] += d["counts"][i]
    avg = [merged_sums[i] / max(1, merged_counts[i]) for i in range(len(merged_sums))]
    print("\n" + "=" * 60)
    print(f"Per-layer mean ||q_i||_2  (codebook, distilled R=32 model)")
    print("=" * 60)
    print(f"{'layer':>6s} {'mean_norm':>11s} {'count':>10s}")
    print("-" * 60)
    for i, (a, c) in enumerate(zip(avg, merged_counts)):
        print(f"{i:>6d} {a:>11.4f} {c:>10d}")
    out = os.path.join(output_dir, "layer_norms.json")
    with open(out, "w") as f:
        json.dump({"avg": avg, "counts": merged_counts}, f, indent=2)
    print(f"\n[norm-dump] saved {out}")


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
    _summarize_and_save(args.output_dir)
