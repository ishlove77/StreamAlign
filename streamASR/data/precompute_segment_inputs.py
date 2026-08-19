#!/usr/bin/env python3
"""Precompute and cache _segment_pass inputs for all training/validation data.

Runs the frozen encoder, acoustic head, and RNN-T alignment once and saves
per-sample tensors to disk.  The cached files can then be used by
train_segment_pass.py to train only the segment_pass portion of the model.

Saved per sample (one .pt file each):
    z_raw           : (T_speech, 256)
    char_alignment  : (n_char, T_speech)  bool
    word_alignment  : (n_word, T_speech)  bool
    char_data       : dict
    gt_char_indices : (n_char,)           long
    spk_emb         : (192,)
    waveform        : (T_wav,)            — full waveform for loss
    crop_start      : int
    file_path       : str
"""

import os
import sys
import glob
import argparse
import hashlib

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml
from transformers import AutoTokenizer

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", message=".*StreamingMediaDecoder.*")
warnings.filterwarnings("ignore", message=".*TorchCodec.*")

_STREAMASR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STREAMASR_ROOT)

matcha_path = os.path.join(_STREAMASR_ROOT, "third_party", "Matcha-TTS")
if matcha_path not in sys.path:
    sys.path.append(matcha_path)

from utils.data_utils import LibriTTSDataset, unified_collate_fn
from utils.train_utils import (
    is_main_process,
    load_filtered_state,
    preprocess_batch,
    process_batch,
)
from models.model import Data2VecSemanticAcousticModel

sys.path.insert(0, "/home/CosyVoice")
from cosyvoice.cli.frontend import CosyVoiceFrontEnd


###############################################################################
# Argument parsing
###############################################################################

def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute segment_pass inputs and cache to disk."
    )
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Root directory to write cached .pt files.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--chunk_size", type=int, default=4)
    parser.add_argument("--left_context", type=int, default=32)
    parser.add_argument(
        "--hparams", type=str,
        default="/home/streamalign/streamASR/hparams/alignment.yaml",
    )
    parser.add_argument(
        "--checkpoint_path", type=str,
        default="/home/streamalign/streamASR/results/char_asr_ckpt",
    )
    parser.add_argument(
        "--resume_path", type=str, default=None,
        help="Student model checkpoint to load (if you want acoustic_head weights).",
    )
    parser.add_argument("--splits", type=str, default="train,val",
                        help="Comma-separated splits to process: train, val, or both.")
    return parser.parse_args()


###############################################################################
# Distributed setup
###############################################################################

def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}"), local_rank


###############################################################################
# Per-sample saver
###############################################################################

def _path_to_key(file_path: str) -> str:
    """Deterministic 16-char hex key derived from the file path."""
    return hashlib.md5(file_path.encode()).hexdigest()[:16]


@torch.no_grad()
def precompute_split(
    wavpaths, split_name, args, model, frontend, glob_tok, device, local_rank
):
    out_dir = os.path.join(args.output_dir, split_name)
    os.makedirs(out_dir, exist_ok=True)

    dataset = LibriTTSDataset(wavpaths)
    sampler = DistributedSampler(dataset, shuffle=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=unified_collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    it = tqdm(loader, desc=f"[Rank {local_rank}] {split_name}") if is_main_process() else loader
    saved = skipped = 0

    for batch_data in it:
        if batch_data is None:
            continue

        try:
            preprocessed = preprocess_batch(batch_data, frontend, glob_tok, device)
        except Exception as e:
            print(f"[Rank {local_rank}] preprocess_batch failed: {e}")
            continue

        # Skip batches with no TextGrid (char_ids_list all-None → constrained
        # alignment impossible; with batch_size=1 this is always fatal).
        char_ids_list = preprocessed.get("char_ids_list", [])
        if not any(c is not None for c in char_ids_list):
            skipped += 1
            continue

        try:
            model_out = process_batch(
                model, preprocessed, device, detach=True, tokenizer=glob_tok
            )
        except Exception as e:
            print(f"[Rank {local_rank}] process_batch failed: {e}")
            continue

        if model_out is None:
            skipped += 1
            continue

        fwd = model_out["forward_output"]
        char_alignment = model_out.get("char_alignment")  # (B, n_char, T)
        word_alignment = model_out.get("word_alignment")  # (B, n_word, T)
        z_raw_batch    = fwd.get("z_raw")                 # (B, T, 256)

        if char_alignment is None or word_alignment is None or z_raw_batch is None:
            skipped += 1
            continue

        B = z_raw_batch.size(0)
        file_paths  = preprocessed["file_paths"]
        waveforms   = preprocessed["waveforms"]   # (B, T_wav) on device
        crop_starts = preprocessed["crop_starts"] # list[int]
        spk_emb_batch = preprocessed["spk_emb"]  # (B, 192)

        # char_data was built inside preprocess_batch
        char_data_batch = preprocessed.get("char_data", [{}] * B)

        # gt_char_indices: build from model forward output
        # It is computed inside Data2VecSemanticAcousticModel.forward() and is
        # NOT returned in model_out directly.  Re-derive it from char_data.
        from utils.text_utils import CHAR_TO_IDX as _C2I
        SPACE_IDX = _C2I.get(" ", 0)

        for b in range(B):
            if not char_alignment[b].any() or not word_alignment[b].any():
                skipped += 1
                continue

            fp = file_paths[b]
            key = _path_to_key(fp)
            out_path = os.path.join(out_dir, f"{key}.pt")

            # Skip already-computed files (allows resuming)
            if os.path.exists(out_path):
                saved += 1
                continue

            cd = char_data_batch[b]
            char_indices = cd.get("char_indices", [])
            n_chars = len(char_indices)

            gt_char_indices = torch.full((n_chars,), SPACE_IDX, dtype=torch.long)
            for i, ci in enumerate(char_indices):
                gt_char_indices[i] = ci

            sample = {
                "z_raw":           z_raw_batch[b].cpu(),           # (T, 256)
                "char_alignment":  char_alignment[b].cpu(),        # (n_char, T)
                "word_alignment":  word_alignment[b].cpu(),        # (n_word, T)
                "char_data":       {k: list(v) if hasattr(v, '__iter__') and not isinstance(v, str) else v
                                    for k, v in cd.items()},
                "gt_char_indices": gt_char_indices,                # (n_char,)
                "spk_emb":         spk_emb_batch[b].cpu(),        # (192,)
                "waveform":        waveforms[b, :preprocessed["wav_lens"][b].long().item()].cpu(),
                "file_path":       fp,
            }
            torch.save(sample, out_path)
            saved += 1

    if is_main_process():
        print(f"[{split_name}] Rank {local_rank}: saved {saved}, skipped {skipped}")


###############################################################################
# Entry point
###############################################################################

def main():
    args = parse_args()
    device, local_rank = setup_distributed()
    torch.backends.cudnn.benchmark = True

    # Build model (encoder + acoustic_head only needed; segment_pass not needed here)
    model_dir = "/home/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"

    with open(f"{model_dir}/cosyvoice3.yaml", "r") as f:
        configs = load_hyperpyyaml(f)
    frontend = CosyVoiceFrontEnd(
        get_tokenizer=configs["get_tokenizer"],
        feat_extractor=configs["feat_extractor"],
        campplus_model=f"{model_dir}/campplus.onnx",
        speech_tokenizer_model=f"{model_dir}/speech_tokenizer_v3.onnx",
        allowed_special=configs["allowed_special"],
    )

    glob_tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    glob_tok.pad_token = glob_tok.eos_token

    model = Data2VecSemanticAcousticModel(
        chunk_size=args.chunk_size,
        left_context=args.left_context,
        hparams_file=args.hparams,
        checkpoint_path=args.checkpoint_path,
    ).to(device)

    if args.resume_path and os.path.isfile(args.resume_path):
        if is_main_process():
            print(f"Loading student checkpoint: {args.resume_path}")
        ckpt = torch.load(args.resume_path, map_location=device)
        load_filtered_state(model, ckpt["hubert_state_dict"], "student_model")

    model.eval()

    splits = [s.strip() for s in args.splits.split(",")]

    for split in splits:
        wavpaths = []
        if split == "train":
            for dataset in ["train-clean-100", "train-clean-360", "train-other-500"]:
                pattern = f"/home/datasets/LibriTTS/{dataset}/*/*/*.wav"
                wavpaths.extend(glob.glob(pattern))
        elif split == "val":
            for dataset in ["dev-clean"]:
                pattern = f"/home/datasets/LibriTTS/{dataset}/*/*/*.wav"
                wavpaths.extend(glob.glob(pattern))
        else:
            raise ValueError(f"Unknown split: {split!r}. Use 'train' or 'val'.")

        if is_main_process():
            print(f"Processing split '{split}': {len(wavpaths)} wav files.")

        precompute_split(wavpaths, split, args, model, frontend, glob_tok, device, local_rank)
        dist.barrier()

    dist.destroy_process_group()
    if is_main_process():
        print("Precomputation complete.")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
