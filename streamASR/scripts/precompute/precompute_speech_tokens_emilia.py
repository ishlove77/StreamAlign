#!/usr/bin/env python3
"""Precompute and cache CosyVoice speech tokens + speaker embeddings for Emilia.

For every Emilia .mp3 file listed in the manifest CSV, saves:
  <cache_root>/<rel>.speech_tokens.pt  -- shape (T_tok,) int64
  <cache_root>/<rel>.spk_emb.pt        -- shape (D,)    float32

where <rel> is the path of the wav relative to the Emilia data root.
Cache layout matches utils.data_utils_cosyvoice._emilia_feature_paths so the
training dataloader (EmiliaTextGridDataset, use_precomputed_features=True)
finds them without further configuration.

Skips files that already have both caches.  Files longer than 30 s
(as reported by the CSV `duration` column) are skipped, matching the
dataloader's runtime filter.

Run with multiple processes via --rank / --world_size for parallelism.
"""

import os, sys, csv, argparse, tempfile, time
import torch
import torchaudio
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

_EMILIA_ROOT = os.environ.get("EMILIA_ROOT", "/data/Emilia")
_DEFAULT_CSV = os.path.join(_EMILIA_ROOT, "emilia_en_400h.csv")
_MODEL_DIR   = os.path.join(_COSYVOICE_ROOT, "pretrained_models", "Fun-CosyVoice3-0.5B")
_MAX_WAV_SECONDS = 30.0
_SAMPLE_RATE     = 16_000

# Match utils/data_utils_cosyvoice.py: _FEATURE_CACHE_ROOT / "_emilia".
_DEFAULT_CACHE_ROOT = os.path.join(
    os.environ.get(
        "COSYVOICE_FEATURE_CACHE_ROOT",
        os.path.join(_STREAMASR_ROOT, "cache", "cosyvoice_features"),
    ),
    "_emilia",
)
_DEFAULT_CACHE_ROOT = os.environ.get(
    "COSYVOICE_FEATURE_CACHE_ROOT_EMILIA", _DEFAULT_CACHE_ROOT
)


def _resolve_data_path(path_value: str, data_folder: str) -> str:
    resolved = path_value.strip()
    for token in ("$data_root", "${data_root}", "{data_root}"):
        resolved = resolved.replace(token, data_folder)
    if os.path.isabs(resolved):
        return os.path.normpath(resolved)
    return os.path.normpath(os.path.join(data_folder, resolved))


def cache_paths(cache_root: str, wav_path: str, data_root: str):
    rel = os.path.relpath(wav_path, data_root)
    base = os.path.join(cache_root, rel)
    return base + ".speech_tokens.pt", base + ".spk_emb.pt"


def atomic_save(obj, path):
    """torch.save via tempfile + os.replace, so concurrent writers can't tear
    the on-disk file. Two racing workers each write their own .tmp; whichever
    rename runs last wins. Both contents are valid (deterministic given input)."""
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".__tmp_", suffix=".pt")
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rank",       type=int, default=0)
    p.add_argument("--world_size", type=int, default=1)
    p.add_argument("--csv",        type=str, default=_DEFAULT_CSV,
                   help="Emilia manifest CSV (ID,duration,wav,spk_id,wrd).")
    p.add_argument("--data_root",  type=str, default=_EMILIA_ROOT,
                   help="Root directory replacing $data_root in the CSV.")
    p.add_argument("--cache_root", type=str, default=_DEFAULT_CACHE_ROOT,
                   help="Writable root directory for cached .pt files. "
                        "Cache layout mirrors wav paths relative to --data_root.")
    p.add_argument("--max_seconds", type=float, default=_MAX_WAV_SECONDS,
                   help="Skip CSV rows whose `duration` exceeds this value.")
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

    all_wavs = []
    skipped_long = 0
    with open(args.csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                dur = float(row["duration"])
            except (KeyError, ValueError):
                dur = 0.0
            if dur > args.max_seconds:
                skipped_long += 1
                continue
            all_wavs.append(_resolve_data_path(row["wav"], args.data_root))
    all_wavs.sort()
    print(f"CSV: {args.csv}")
    print(f"Total wav files (≤{args.max_seconds:.0f}s): {len(all_wavs)} "
          f"(skipped {skipped_long} longer rows)")

    my_wavs = all_wavs[args.rank :: args.world_size]

    if args.overwrite:
        todo = list(my_wavs)
    else:
        todo = []
        for p in my_wavs:
            tok_path, spk_path = cache_paths(args.cache_root, p, args.data_root)
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
            wav, sr = torchaudio.load(wav_path)
            wav = wav.mean(0) if wav.size(0) > 1 else wav.squeeze(0)
            if sr != _SAMPLE_RATE:
                wav = F_audio.resample(wav, sr, _SAMPLE_RATE)

            tok_path, spk_path = cache_paths(args.cache_root, wav_path, args.data_root)
            os.makedirs(os.path.dirname(tok_path), exist_ok=True)

            if args.overwrite or not os.path.exists(tok_path):
                tokens = extract_tokens(frontend, wav)
                atomic_save(tokens, tok_path)

            if args.overwrite or not os.path.exists(spk_path):
                spk = extract_spk_emb(frontend, wav)
                atomic_save(spk, spk_path)

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
