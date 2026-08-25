#!/usr/bin/env python3
"""Extract per-subword (w_n, q_n, d_n) units from a trained StreamAlign model.

Output layout (mirrors LibriTTS/Emilia path structure under cache_root):
    <cache_root>/<rel>.units.pt        SubwordUnits (subword_ids, q_codes, duration_frames)
    <cache_root>/manifest_shard{R}_of{W}.csv

Each manifest row: rel_path, n_subwords, n_frames_total, units_pt_path

Run sharded for parallelism:
    sr 1 24 python -m streamSLM.extract.extract_tokens \
        --dataset libritts --split train-clean-100 \
        --rank 0 --world_size 8 \
        --variant rvq \
        --checkpoint weights/Streamalign-R16/rvq_teacher/epoch_22.pt \
        --cache_root cache/streamSLM_units
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from contextlib import nullcontext as _nullcontext
from pathlib import Path
from typing import List

# alignment.yaml instantiates a WandBLogger at model construction; we never log
# anything during extraction, so force wandb fully offline before any import
# pulls it in.
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")

import torch
import torch.distributed as dist  # noqa: F401  (parity w/ existing scripts)
from torch.utils.data import DataLoader, Subset

# Resolve sibling streamASR/ + CosyVoice imports the same way streamASR's own
# scripts do, so a user only needs to invoke this module from the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_STREAMASR_ROOT = os.path.join(_REPO_ROOT, "streamASR")
_COSYVOICE_ROOT = os.environ.get(
    "COSYVOICE_ROOT", os.path.join(_STREAMASR_ROOT, "third_party", "CosyVoice")
)
for p in [
    os.path.join(_STREAMASR_ROOT, "third_party", "Matcha-TTS"),
    os.path.join(_COSYVOICE_ROOT, "third_party", "Matcha-TTS"),
    _COSYVOICE_ROOT,
    _STREAMASR_ROOT,
    _REPO_ROOT,
]:
    if p not in sys.path:
        sys.path.insert(0, p)

from cosyvoice.cli.frontend import CosyVoiceFrontEnd  # type: ignore  # noqa: E402
from hyperpyyaml import load_hyperpyyaml  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from utils.data_utils_cosyvoice import (  # noqa: E402
    LibriTTSDataset, LibriSpeechFlacDataset, EmiliaTextGridDataset,
    unified_collate_fn,
    _LIBRITTS_ROOT, _LIBRISPEECH_ROOT, _EMILIA_ROOT, load_emilia_wavpaths,
)
from utils.train_utils import preprocess_batch  # noqa: E402

from streamSLM.units import SubwordUnits  # noqa: E402


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def _load_model_class(variant: str = "rvq"):
    """RVQ is the only supported quantizer."""
    if variant != "rvq":
        raise ValueError(f"unsupported variant: {variant} (only 'rvq')")
    from models.model_tokenizer import Data2VecSemanticAcousticModel as M
    return M


def _load_streamalign_model(
    variant: str, checkpoint: str, hparams: str, truthmodel_ckpt: str,
    chunk_size: int, left_context: int, device: torch.device,
):
    ModelCls = _load_model_class(variant)
    model = ModelCls(
        chunk_size=chunk_size,
        left_context=left_context,
        hparams_file=hparams,
        checkpoint_path=truthmodel_ckpt,
    )
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "hubert_state_dict" in ckpt:
        sd = ckpt["hubert_state_dict"]
    elif "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[load] missing={len(missing)} unexpected={len(unexpected)} (proceeding)")
    model.to(device).eval()
    return model


def _load_cosyvoice_frontend(model_dir: str):
    cfg_path = os.path.join(model_dir, "cosyvoice3.yaml")
    with open(cfg_path) as f:
        configs = load_hyperpyyaml(f)
    return CosyVoiceFrontEnd(
        get_tokenizer=configs["get_tokenizer"],
        feat_extractor=configs["feat_extractor"],
        campplus_model=f"{model_dir}/campplus.onnx",
        speech_tokenizer_model=f"{model_dir}/speech_tokenizer_v3.onnx",
        allowed_special=configs["allowed_special"],
    )


# --------------------------------------------------------------------------- #
# Dataset construction
# --------------------------------------------------------------------------- #
def _build_dataset(dataset: str, split: str, emilia_csv: str = ""):
    """Build the per-utterance dataset.

    Emilia mode picks one of two sources:
      - emilia_csv != ""  : the curated 400h subset CSV; `split` is just used
                            as the cache subdirectory name (e.g. "400h").
      - emilia_csv == ""  : a single Emilia subdir, e.g. "EN-B000062".
    """
    if dataset == "libritts":
        root = Path(_LIBRITTS_ROOT) / split
        wavpaths = sorted(str(p) for p in root.rglob("*.wav"))
        return LibriTTSDataset(wavpaths, use_precomputed_features=False), _LIBRITTS_ROOT
    if dataset == "librispeech":
        root = Path(_LIBRISPEECH_ROOT) / split
        wavpaths = sorted(str(p) for p in root.rglob("*.flac"))
        return LibriSpeechFlacDataset(wavpaths), _LIBRISPEECH_ROOT
    if dataset == "emilia":
        if emilia_csv:
            wavpaths = load_emilia_wavpaths(emilia_csv, _EMILIA_ROOT)
        else:
            root = Path(_EMILIA_ROOT) / split
            wavpaths = sorted(str(p) for p in root.rglob("*.mp3"))
        return EmiliaTextGridDataset(wavpaths, use_precomputed_features=False), _EMILIA_ROOT
    raise ValueError(f"unknown dataset: {dataset}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["libritts", "librispeech", "emilia"], required=True)
    ap.add_argument("--split", required=True,
                    help="LibriTTS split (e.g. train-clean-100) or Emilia subdir / "
                         "name to use as the cache subdir when --emilia_csv is set")
    ap.add_argument("--emilia_csv", default="",
                    help="If set, drive Emilia extraction from this CSV (e.g. the "
                         "emilia_en_400h.csv subset) instead of an EN-B subdir rglob.")
    ap.add_argument("--variant", choices=["rvq"], default="rvq")
    ap.add_argument("--checkpoint", required=True, help="StreamAlign student-model .pt")
    ap.add_argument("--hparams", default=
                    os.environ.get("ASR_HPARAMS", os.path.join(_STREAMASR_ROOT, "hparams", "alignment.yaml")))
    # Truth-model paired with alignment.yaml. Every current training/inference
    # script (run_inference_subset.sh, run_train_rvq_example.sh, run_exp*_*.sh,
    # run_inference_mimi_decoder.sh, …) uses char_asr_ckpt.
    # The 02-11 default in inference_core.py is stale (vocab=29 vs 73).
    ap.add_argument("--truthmodel_checkpoint_path", default=
                    os.environ.get("TRUTH_MODEL_CKPT", os.path.join(_STREAMASR_ROOT, "results", "char_asr_ckpt")))
    ap.add_argument("--chunk_size", type=int, default=16)
    ap.add_argument("--left_context", type=int, default=8)
    ap.add_argument("--cosyvoice_model_dir", default=
                    os.path.join(_COSYVOICE_ROOT, "pretrained_models", "Fun-CosyVoice3-0.5B"))
    ap.add_argument("--tokenizer", choices=["llama", "qwen3"], default="llama")
    ap.add_argument("--cache_root", required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world_size", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--max_batches", type=int, default=0,
                    help="Stop after this many batches (0 = unlimited). For smoke tests.")
    ap.add_argument("--log_every", type=int, default=5,
                    help="Print a progress line every N batches (default 5).")
    # quantizer env knobs (set before importing the model below)
    ap.add_argument("--bf16", action="store_true",
                    help="Run encoder forward under bf16 autocast (inference only).")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Filter out indices whose <cache_root>/<dataset>/<split>/<rel>.units.pt "
                         "already exists. Lets a re-launch with a different WORLD complete only "
                         "the not-yet-done utts (no GPU work spent re-extracting).")
    args = ap.parse_args()

    # Surface quantizer knobs to streamASR's model via env (read on __init__).

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] rank={args.rank}/{args.world_size} device={device} variant={args.variant} "
          f"batch_size={args.batch_size} num_workers={args.num_workers}", flush=True)

    # Tokenizer (LLM subword tokenizer used by char_data_from_constraints)
    if args.tokenizer == "qwen3":
        glob_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    else:
        glob_tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    glob_tok.pad_token = glob_tok.eos_token
    # NeMo Normalizer build is ~10min at startup. extract_subword_units only
    # uses txt_normalizer in the legacy decode path; with TextGrid constraints
    # it is dead weight, so skip it entirely.
    normalizer = None

    # Model
    model = _load_streamalign_model(
        args.variant, args.checkpoint,
        args.hparams, args.truthmodel_checkpoint_path,
        args.chunk_size, args.left_context, device,
    )
    # CosyVoice frontend (mel-spec, speech-token ONNX, CAMP++ ONNX) is unused
    # under extract_only=True. Skip the ONNX initialisations.
    frontend = None

    # Dataset, sharded by rank
    dataset, data_root = _build_dataset(args.dataset, args.split, args.emilia_csv)
    indices = list(range(args.rank, len(dataset), args.world_size))

    if args.skip_existing:
        if not hasattr(dataset, "wavpaths"):
            raise RuntimeError(
                "--skip_existing requires the dataset to expose .wavpaths "
                f"(got {type(dataset).__name__})"
            )
        cache_check_root = Path(args.cache_root) / args.dataset / args.split
        kept = []
        n_skipped_existing = 0
        for i in indices:
            wav_path = dataset.wavpaths[i]
            rel = os.path.relpath(wav_path, data_root)
            if (cache_check_root / (rel + ".units.pt")).exists():
                n_skipped_existing += 1
            else:
                kept.append(i)
        print(f"[shard {args.rank}] skip_existing: kept={len(kept)} "
              f"skipped_existing={n_skipped_existing} (orig_slice={len(indices)})",
              flush=True)
        indices = kept

    shard = Subset(dataset, indices)
    n_total = len(shard)
    total_batches = (n_total + args.batch_size - 1) // args.batch_size
    print(f"[shard {args.rank}] {n_total} / {len(dataset)} samples "
          f"({total_batches} batches at batch_size={args.batch_size})", flush=True)

    loader = DataLoader(
        shard,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=unified_collate_fn,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    # Output paths
    cache_root = Path(args.cache_root) / args.dataset / args.split
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_root / f"manifest_shard{args.rank}_of{args.world_size}.csv"
    manifest_f = open(manifest_path, "w", newline="")
    manifest = csv.writer(manifest_f)
    manifest.writerow(["rel_path", "n_subwords", "n_frames_total", "units_pt"])

    written = skipped_missing_tg = skipped_empty = 0
    t0 = time.time()
    for bi, batch in enumerate(loader):
        # Skip whole batch if NO sample has TextGrid intervals; otherwise let the
        # alignment routine drop the missing ones internally (returns empty units).
        if not any(b is not None for b in batch.get("textgrid_intervals", [])):
            skipped_missing_tg += len(batch["file_paths"])
            continue

        try:
            preproc = preprocess_batch(batch, frontend, glob_tok, device,
                                       txt_normalizer=normalizer,
                                       extract_only=True)
        except Exception as e:
            print(f"[batch {bi}] preprocess failed: {e}")
            continue

        try:
            ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
                   if args.bf16 and device.type == "cuda" else _nullcontext())
            with torch.inference_mode(), ctx:
                units_per_sample = model.extract_subword_units(
                    waveforms=preproc["waveforms"],
                    wav_mask=preproc["wav_mask"],
                    tokenizer=glob_tok,
                    txt_normalizer=normalizer,
                    char_ids_list=preproc["char_ids_list"],
                    constraints_list=preproc["constraints_list"],
                    spk_emb=preproc["spk_emb"],
                )
        except Exception as e:
            print(f"[batch {bi}] extract failed: {e}")
            continue

        for wav_path, u in zip(batch["file_paths"], units_per_sample):
            n = int(u["subword_ids"].numel())
            if n == 0:
                skipped_empty += 1
                continue
            rel = os.path.relpath(wav_path, data_root)
            out_path = cache_root / (rel + ".units.pt")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            SubwordUnits(
                subword_ids=u["subword_ids"],
                q_codes=u["q_codes"],
                duration_frames=u["duration_frames"],
                # Optional: pre-quantizer continuous feature for the continuous
                # regression objective. extract_subword_units returns it on
                # newer checkpoints; .get() guards against older callers.
                pre_quant_feat=u.get("pre_quant_feat", None),
            ).save(str(out_path))
            manifest.writerow([
                rel, n,
                int(u["duration_frames"].sum().item()),
                str(out_path),
            ])
            written += 1

        if (bi + 1) % args.log_every == 0 or bi == 0:
            elapsed = time.time() - t0
            seen = (bi + 1) * args.batch_size
            seen = min(seen, n_total)
            rate = written / max(elapsed, 1e-6)              # utt / s
            batch_rate = (bi + 1) / max(elapsed, 1e-6)       # batches / s
            remaining_batches = max(total_batches - (bi + 1), 0)
            eta_s = remaining_batches / batch_rate if batch_rate > 0 else float("inf")
            eta_str = (time.strftime("%H:%M:%S", time.gmtime(eta_s))
                       if eta_s != float("inf") else "inf")
            print(f"[shard {args.rank}/{args.world_size}] "
                  f"batch {bi+1}/{total_batches}  utt {seen}/{n_total}  "
                  f"written={written}  skipped(no_tg)={skipped_missing_tg}  "
                  f"skipped(empty)={skipped_empty}  rate={rate:.1f}utt/s  eta={eta_str}",
                  flush=True)

        if args.max_batches and (bi + 1) >= args.max_batches:
            print(f"[shard {args.rank}] hit --max_batches={args.max_batches}, stopping early")
            break

    manifest_f.close()
    elapsed = time.time() - t0
    rate = written / max(elapsed, 1e-6)
    print(f"[shard {args.rank}/{args.world_size}] DONE written={written}  "
          f"skipped(no_tg)={skipped_missing_tg}  skipped(empty)={skipped_empty}  "
          f"in {elapsed:.0f}s ({rate:.1f}utt/s)  ->  {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
