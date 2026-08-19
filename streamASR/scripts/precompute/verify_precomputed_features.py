#!/usr/bin/env python3
"""Validate the .speech_tokens.pt / .spk_emb.pt cache produced by
``precompute_speech_tokens.py``.

Checks (per LibriTTS split):
  1. coverage: how many .wav files have BOTH .pt sidecars
  2. shape/dtype sanity on a random sample (default 64 files)
  3. that the dataset+collate path actually loads them and yields the
     keys preprocess_batch consumes (`speech_units`, `spk_emb`)
"""

import os, sys, glob, random, argparse
import torch

_STREAMASR_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _STREAMASR_ROOT not in sys.path:
    sys.path.insert(0, _STREAMASR_ROOT)

from utils.data_utils_cosyvoice import _libri_feature_paths

_LIBRI_ROOT = "/home/datasets/LibriTTS"
_DEFAULT_DATASETS = ["train-clean-100", "train-clean-360", "train-other-500", "dev-clean"]


def _has_cache(wav_path: str) -> bool:
    tok, spk = _libri_feature_paths(wav_path)
    return os.path.exists(tok) and os.path.exists(spk)


def coverage(splits):
    print("Coverage:")
    for ds in splits:
        wavs = glob.glob(os.path.join(_LIBRI_ROOT, ds, "*", "*", "*.wav"))
        both = sum(1 for w in wavs if _has_cache(w))
        pct = 100.0 * both / max(1, len(wavs))
        print(f"  {ds:>20s}: {both:>6d} / {len(wavs):>6d}  ({pct:5.1f} %)")


def sanity_sample(splits, n):
    rng = random.Random(0)
    pool = []
    for ds in splits:
        for w in glob.glob(os.path.join(_LIBRI_ROOT, ds, "*", "*", "*.wav")):
            if _has_cache(w):
                pool.append(w)
    if not pool:
        print("Sanity sample: no cached files found.")
        return False
    sample = rng.sample(pool, min(n, len(pool)))
    print(f"\nSanity sample (n={len(sample)}):")
    bad = 0
    for w in sample:
        tok_p, spk_p = _libri_feature_paths(w)
        try:
            tok = torch.load(tok_p, map_location="cpu")
            spk = torch.load(spk_p, map_location="cpu")
        except Exception as e:
            print(f"  LOAD-ERROR {w}: {e}")
            bad += 1; continue
        ok_tok = tok.dim() == 1 and tok.dtype in (torch.int32, torch.int64) and tok.numel() > 0
        ok_spk = spk.numel() == 192 and spk.dtype == torch.float32
        if not (ok_tok and ok_spk):
            print(f"  SHAPE-BAD {w}: tok={tuple(tok.shape)}/{tok.dtype} "
                  f"spk={tuple(spk.shape)}/{spk.dtype}")
            bad += 1
    print(f"  pass: {len(sample) - bad}/{len(sample)}, bad: {bad}")
    return bad == 0


def collate_smoketest(splits, n):
    """End-to-end check: dataset → collate yields ``speech_units`` + ``spk_emb``."""
    from utils.data_utils_cosyvoice import LibriTTSDataset, unified_collate_fn
    paths = []
    for ds in splits:
        paths.extend(glob.glob(os.path.join(_LIBRI_ROOT, ds, "*", "*", "*.wav")))
    paths = [p for p in paths
             if os.path.exists(p + ".speech_tokens.pt")
             and os.path.exists(p + ".spk_emb.pt")][:max(n, 4)]
    if len(paths) < 2:
        print("\nCollate smoketest: not enough cached files."); return False
    ds = LibriTTSDataset(paths, use_precomputed_features=True)
    items = [ds[i] for i in range(min(len(ds), n))]
    batch = unified_collate_fn(items)
    print("\nCollate smoketest:")
    print(f"  batch keys: {sorted(batch.keys())}")
    for k in ("speech_units", "spk_emb"):
        if k not in batch:
            print(f"  MISSING key {k!r}"); return False
    print(f"  speech_units: {tuple(batch['speech_units'].shape)} {batch['speech_units'].dtype}")
    print(f"  spk_emb:      {tuple(batch['spk_emb'].shape)} {batch['spk_emb'].dtype}")
    if batch["spk_emb"].shape[-1] != 192:
        print("  spk_emb dim != 192"); return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=_DEFAULT_DATASETS)
    ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args()

    coverage(args.datasets)
    ok1 = sanity_sample(args.datasets, args.n)
    ok2 = collate_smoketest(args.datasets, n=8)
    print("\nRESULT:", "PASS" if (ok1 and ok2) else "FAIL")


if __name__ == "__main__":
    main()
