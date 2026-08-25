#!/usr/bin/env python3
"""Precompute and cache CosyVoice speech tokens + speaker embeddings.

For every audio file under <data_root>/<split>/*/*/*.<ext>, saves:
  <cache_root>[<cache_subdir>/]<rel>.speech_tokens.pt  -- shape (T_tok,) int64
  <cache_root>[<cache_subdir>/]<rel>.spk_emb.pt        -- shape (D,)    float32

Defaults target LibriTTS so existing invocations keep working. To preprocess
a different LibriVox-derived corpus that shares LibriTTS split names (e.g.
LibriSpeech), pass --cache_subdir to namespace the output and avoid clobbering
the LibriTTS cache. See the LibriSpeech launch script for an example.

Skips files that already have both caches. Pass --overwrite to redo.
Shard with --rank / --world_size for parallelism.
"""

import os, sys, glob, argparse, time
import torch
import torchaudio.functional as F_audio
from hyperpyyaml import load_hyperpyyaml

_STREAMASR_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_COSYVOICE_ROOT = os.environ.get(
    "COSYVOICE_ROOT", os.path.join(_STREAMASR_ROOT, "third_party", "CosyVoice")
)

for p in [
    os.path.join(_STREAMASR_ROOT, "third_party", "Matcha-TTS"),
    os.path.join(_COSYVOICE_ROOT, "third_party", "Matcha-TTS"),
    _COSYVOICE_ROOT,
    _STREAMASR_ROOT,
]:
    if p not in sys.path:
        sys.path.insert(0, p)

from cosyvoice.cli.frontend import CosyVoiceFrontEnd

_LIBRI_ROOT = os.environ.get("LIBRITTS_ROOT", "/data/LibriTTS")
_DATASETS   = ["train-clean-100", "train-clean-360", "train-other-500",
               "dev-clean", "dev-other"]
_MODEL_DIR  = os.path.join(_COSYVOICE_ROOT, "pretrained_models", "Fun-CosyVoice3-0.5B")
_MAX_WAV_SECONDS = 30.0
_SAMPLE_RATE     = 16_000

# Default writable cache root. Per-dataset --cache_subdir is appended below.
_DEFAULT_CACHE_ROOT = os.path.join(_STREAMASR_ROOT, "cache", "cosyvoice_features")


def cache_paths(cache_root: str, data_root: str, wav_path: str):
    """Return (tokens_path, spk_emb_path) under the writable cache root,
    mirroring wav_path's location relative to data_root."""
    rel = os.path.relpath(wav_path, data_root)
    base = os.path.join(cache_root, rel)
    return base + ".speech_tokens.pt", base + ".spk_emb.pt"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rank",       type=int, default=0)
    p.add_argument("--world_size", type=int, default=1)
    p.add_argument("--data_root",  type=str, default=_LIBRI_ROOT,
                   help="Root directory containing <split>/<spk>/<chap>/<utt>.<ext> "
                        "audio files. Defaults to the LibriTTS root.")
    p.add_argument("--datasets",   nargs="+", default=_DATASETS,
                   help="Splits (subdirectories of --data_root) to process.")
    p.add_argument("--ext",        type=str, default="wav",
                   help="Audio file extension to glob for (no leading dot). "
                        "Default 'wav'; use 'flac' for raw LibriSpeech.")
    p.add_argument("--cache_root", type=str, default=_DEFAULT_CACHE_ROOT,
                   help="Writable root directory for cached .pt files. "
                        "Cache layout mirrors audio paths relative to --data_root.")
    p.add_argument("--cache_subdir", type=str, default="",
                   help="Optional subdirectory under --cache_root to namespace this "
                        "dataset's cache (e.g. '_librispeech'). Required when running "
                        "a corpus whose split names collide with LibriTTS.")
    p.add_argument("--overwrite",  action="store_true",
                   help="Reprocess and overwrite existing .pt cache files.")
    return p.parse_args()


def load_frontend():
    with open(f"{_MODEL_DIR}/cosyvoice3.yaml") as f:
        configs = load_hyperpyyaml(f)
    return CosyVoiceFrontEnd(
        get_tokenizer=configs["get_tokenizer"],
        feat_extractor=configs["feat_extractor"],
        campplus_model=f"{_MODEL_DIR}/campplus.onnx",
        speech_tokenizer_model=f"{_MODEL_DIR}/speech_tokenizer_v3.onnx",
        allowed_special=configs["allowed_special"],
    )


def extract_tokens(frontend, wav: torch.Tensor) -> torch.Tensor:
    """Extract speech tokens for a (T,) waveform, chunking if >30s."""
    max_samples = int(_MAX_WAV_SECONDS * _SAMPLE_RATE)
    if wav.size(0) <= max_samples:
        units, _ = frontend._extract_speech_token(wav.unsqueeze(0))
        return units.squeeze(0).cpu()
    chunks = []
    for start in range(0, wav.size(0), max_samples):
        u, _ = frontend._extract_speech_token(
            wav[start : start + max_samples].unsqueeze(0)
        )
        chunks.append(u.squeeze(0).cpu())
    return torch.cat(chunks, dim=0)


def extract_spk_emb(frontend, wav: torch.Tensor) -> torch.Tensor:
    """Extract speaker embedding for a (T,) waveform, chunking if >30s."""
    max_samples = int(_MAX_WAV_SECONDS * _SAMPLE_RATE)
    if wav.size(0) <= max_samples:
        return frontend._extract_spk_embedding(wav.unsqueeze(0)).squeeze(0).cpu()
    embs = []
    for start in range(0, wav.size(0), max_samples):
        e = frontend._extract_spk_embedding(wav[start : start + max_samples].unsqueeze(0))
        embs.append(e.squeeze(0).cpu())
    return torch.stack(embs).mean(dim=0)


def main():
    args = parse_args()

    # Resolve dataset-specific cache root. Namespacing prevents LibriSpeech (or
    # any other corpus that shares LibriTTS split names) from clobbering the
    # existing LibriTTS cache when --cache_subdir is passed.
    cache_root = (os.path.join(args.cache_root, args.cache_subdir)
                  if args.cache_subdir else args.cache_root)
    data_root = os.path.abspath(args.data_root)
    ext = args.ext.lstrip(".")

    # Gather all audio files
    all_wavs = []
    for ds in args.datasets:
        pattern = os.path.join(data_root, ds, "*", "*", f"*.{ext}")
        all_wavs.extend(glob.glob(pattern))
    all_wavs.sort()
    print(f"data_root={data_root}  cache_root={cache_root}  ext=.{ext}")
    print(f"Total {ext} files: {len(all_wavs)}")

    # Shard across processes
    my_wavs = all_wavs[args.rank :: args.world_size]

    # Filter to only files missing at least one cache (unless --overwrite).
    if args.overwrite:
        todo = list(my_wavs)
    else:
        todo = []
        for p in my_wavs:
            tok_path, spk_path = cache_paths(cache_root, data_root, p)
            if not (os.path.exists(tok_path) and os.path.exists(spk_path)):
                todo.append(p)
    print(f"Rank {args.rank}/{args.world_size}: {len(todo)} files to process "
          f"(skipping {len(my_wavs) - len(todo)} already cached)")

    if not todo:
        return

    print("Loading CosyVoice frontend...")
    frontend = load_frontend()
    print("Frontend ready.\n")

    t0 = time.time()
    for i, wav_path in enumerate(todo):
        try:
            import torchaudio
            wav, sr = torchaudio.load(wav_path)
            wav = wav.mean(0) if wav.size(0) > 1 else wav.squeeze(0)
            if sr != _SAMPLE_RATE:
                wav = F_audio.resample(wav, sr, _SAMPLE_RATE)

            tok_path, spk_path = cache_paths(cache_root, data_root, wav_path)
            os.makedirs(os.path.dirname(tok_path), exist_ok=True)

            if args.overwrite or not os.path.exists(tok_path):
                tokens = extract_tokens(frontend, wav)
                torch.save(tokens, tok_path)

            if args.overwrite or not os.path.exists(spk_path):
                spk = extract_spk_emb(frontend, wav)
                torch.save(spk, spk_path)

        except Exception as e:
            print(f"  ERROR {wav_path}: {e}")
            continue

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(todo) - i - 1) / rate
            print(f"  [{i+1}/{len(todo)}]  {rate:.1f} files/s  "
                  f"ETA {remaining/60:.0f} min")

    print(f"\nDone. Processed {len(todo)} files in {(time.time()-t0)/60:.1f} min.")


if __name__ == "__main__":
    main()
