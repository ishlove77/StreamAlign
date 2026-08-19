#!/usr/bin/env python3
"""
compare_flag_latency.py

Compares per-word streaming latency between flag=False and flag=True modes.

When flag=True, chunks that produce no new text immediately commit all
accumulated words (the "none_text" early-commit heuristic).
When flag=False (default), those chunks leave the last word tentative.

The script runs both modes on every file, compares:
  - Aggregate latency statistics (mean, std, p50, p90)
  - WER/CER for each mode
  - Per-file transcription differences (saved to text.txt if any differ)

Usage
-----
python compare_flag_latency.py \
    --hparams_file /path/to/chunk_streaming_word_fastemit.yaml \
    --checkpoint   /path/to/CKPT+... \
    --input_dir    /path/to/LibriSpeech \
    --split test-clean \
    [--max_files 100]
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import csv
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List

import jiwer
import numpy as np
import torch
from tqdm import tqdm

# Allow imports from streamASR root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.checkpoints import Checkpointer
from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig
from speechbrain.inference.ASR import StreamingASR

from measure_latency_word import (
    SAMPLE_RATE,
    WordCommitInfo,
    parse_textgrid_words,
    stream_file_with_timing,
    compute_file_latencies,
    load_boundary_classifier,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare streaming latency: flag=False vs flag=True"
    )
    p.add_argument("--hparams_file",
                   default="/home/streamalign/streamASR/hparams/chunk_streaming_word_fastemit.yaml")
    p.add_argument("--checkpoint", type=str,
                   default="/home/streamalign/streamASR/train/results/conformer_transducer_char/word_fastemit/save/word_asr_ckpt")
    p.add_argument("--chunk_size", type=int, default=4)
    p.add_argument("--left_context", type=int, default=32)
    p.add_argument("--input_dir",
                   default="/home/datasets/LibriSpeech")
    p.add_argument("--split", default="test-clean")
    p.add_argument("--max_files", type=int, default=None)
    p.add_argument("--output_dir", type=str,
                   default="/home/streamalign/streamASR/evaluate/latency/compare_results",
                   help="Directory for output files (text.txt, latency CSVs).")
    p.add_argument("--csv_dir", type=str,
                   default="/home/datasets/LibriSpeech/csv")
    p.add_argument("--tokenizer_ckpt", type=str,
                   default="/home/streamalign/streamASR/train/results/conformer_transducer_char/word_fastemit/pretrained/tokenizer.ckpt")
    p.add_argument("--boundary_classifier_ckpt", type=str,
                   default="/home/streamalign/streamASR/train/results/best_model.pt")
    return p.parse_args()


def print_stats(label: str, records: List[Dict], hyps: List[str], refs: List[str]):
    """Print latency statistics and WER/CER for one mode."""
    if not records:
        print(f"\n[{label}] No data collected.")
        return

    lat_ms = np.array([r["latency_ms"] for r in records])
    model_ms = np.array([r["model_processing_ms"] for r in records])
    audio_ms = (
        np.array([r["audio_time_at_commit_s"] for r in records]) -
        np.array([r["gt_end_s"] for r in records])
    ) * 1000.0

    def _row(name, arr):
        return (f"  {name:<8s}:"
                f"  mean={arr.mean():>7.1f}"
                f"  std={arr.std():>7.1f}"
                f"  p50={np.median(arr):>7.1f}"
                f"  p90={np.percentile(arr, 90):>7.1f}")

    print(f"\n{'='*65}")
    print(f"  [{label}]  Per-word Streaming Latency (ms)")
    print(f"  Words measured: {len(lat_ms)}")
    print(f"{'='*65}")
    print(_row("total", lat_ms))
    print(_row("audio", audio_ms))
    print(_row("model", model_ms))

    if hyps and refs:
        wer = jiwer.wer(refs, hyps) * 100
        cer = jiwer.cer(refs, hyps) * 100
        print(f"  WER : {wer:.2f}%")
        print(f"  CER : {cer:.2f}%")
    print(f"{'='*65}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model (shared between both runs) ────────────────────────────
    print("Loading StreamingASR ...")
    with open(args.hparams_file, encoding="utf-8") as f:
        hparams = load_hyperpyyaml(f)
    asr_model = StreamingASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts={"device": str(device)},
    )
    torch.set_grad_enabled(False)

    # ── Load checkpoint ──────────────────────────────────────────────────
    ckpt_dir = Path(args.checkpoint)
    if ckpt_dir.is_file():
        ckpt_dir = ckpt_dir.parent
    model_ckpt = ckpt_dir / "model.ckpt"
    norm_ckpt = ckpt_dir / "normalizer.ckpt"

    if model_ckpt.exists():
        print(f"Loading model from: {model_ckpt}")
        state = torch.load(model_ckpt, map_location=device)
        missing, unexpected = hparams["model"].load_state_dict(state, strict=False)
        if missing:
            print(f"  Missing keys: {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected[:3]}{'...' if len(unexpected) > 3 else ''}")
    else:
        recoverables = {"model": hparams["model"], "normalizer": hparams["normalize"]}
        checkpointer = Checkpointer(str(ckpt_dir.parent), recoverables=recoverables)
        checkpointer.recover_if_possible()

    if norm_ckpt.exists():
        print(f"Loading normalizer from: {norm_ckpt}")
        norm_state = torch.load(norm_ckpt, map_location=device)
        normalize = hparams["modules"].get("normalize")
        if normalize and isinstance(norm_state, dict) and "glob_mean" in norm_state:
            normalize.glob_mean = norm_state["glob_mean"]
            normalize.glob_std = norm_state["glob_std"]
            normalize.count = norm_state.get("count", 1)

    hparams["model"].eval()
    print("Model loaded!")

    # ── Load BoundaryClassifier ──────────────────────────────────────────
    boundary_classifier = None
    if args.boundary_classifier_ckpt:
        print(f"Loading BoundaryClassifier from: {args.boundary_classifier_ckpt}")
        boundary_classifier = load_boundary_classifier(args.boundary_classifier_ckpt, device)
        print("  BoundaryClassifier loaded.\n")

    # ── Load tokenizer ───────────────────────────────────────────────────
    tokenizer = getattr(asr_model.hparams, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "Load"):
        tok_loaded = getattr(tokenizer, "vocab_size", lambda: 0)() > 0
        if not tok_loaded:
            candidates = []
            if args.tokenizer_ckpt:
                candidates.append(args.tokenizer_ckpt)
            ckpt_dir_tok = Path(args.checkpoint)
            candidates.append(str(ckpt_dir_tok.parent.parent / "pretrained" / "tokenizer.ckpt"))
            for cand in candidates:
                if os.path.exists(cand):
                    print(f"Loading tokenizer from: {cand}")
                    tokenizer.Load(cand)
                    break
            else:
                raise FileNotFoundError(f"tokenizer.ckpt not found. Searched: {candidates}")

    dynchunktrain_config = DynChunkTrainConfig(
        chunk_size=args.chunk_size,
        left_context_size=args.left_context,
    )
    chunk_frames = asr_model.get_chunk_size_frames(dynchunktrain_config)
    chunk_ms = chunk_frames / SAMPLE_RATE * 1000.0
    print(f"Streaming: chunk_size={args.chunk_size} ({chunk_ms:.0f} ms), "
          f"left_context={args.left_context}\n")

    # ── Load CSV references ──────────────────────────────────────────────
    csv_dir = Path(args.csv_dir) if args.csv_dir else Path(args.input_dir) / "csv"
    csv_path = csv_dir / f"{args.split}.csv"
    id_to_ref: Dict[str, str] = {}
    if csv_path.exists():
        with open(csv_path, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                id_to_ref[row["ID"]] = row["wrd"].lower()
        print(f"Loaded {len(id_to_ref)} references from {csv_path}")
    else:
        print(f"Warning: CSV not found at {csv_path} — WER/CER skipped.")

    # ── Gather wav files ─────────────────────────────────────────────────
    split_dir = Path(args.input_dir) / args.split
    wav_files = sorted(split_dir.glob("**/*.wav"))
    if args.max_files:
        wav_files = wav_files[:args.max_files]
    print(f"Found {len(wav_files)} .wav files in '{args.split}'.\n")

    # ── Run both modes ───────────────────────────────────────────────────
    # flag=False: default (none_text heuristic disabled)
    # flag=True:  none_text heuristic enabled (commit all when no new text)
    records_off: List[Dict] = []    # flag=False
    records_on: List[Dict] = []     # flag=True
    hyps_off: List[str] = []
    hyps_on: List[str] = []
    refs_off: List[str] = []
    refs_on: List[str] = []

    # Per-file transcription comparison
    diff_lines: List[str] = []
    skipped = 0

    for wav_path in tqdm(wav_files, desc="Comparing flag=False vs flag=True"):
        tg_path = wav_path.with_suffix(".TextGrid")
        if not tg_path.exists():
            skipped += 1
            continue

        gt_intervals = parse_textgrid_words(str(tg_path))
        if not gt_intervals:
            skipped += 1
            continue

        try:
            # --- flag=False ---
            wc_off = stream_file_with_timing(
                asr_model, str(wav_path), dynchunktrain_config,
                boundary_classifier=boundary_classifier,
                device=device, flag=False,
            )
            rec_off = compute_file_latencies(wc_off, gt_intervals)
            records_off.extend(rec_off)
            hyp_off = " ".join(w.word.lower() for w in wc_off)

            # --- flag=True ---
            wc_on = stream_file_with_timing(
                asr_model, str(wav_path), dynchunktrain_config,
                boundary_classifier=boundary_classifier,
                device=device, flag=True,
            )
            rec_on = compute_file_latencies(wc_on, gt_intervals)
            records_on.extend(rec_on)
            hyp_on = " ".join(w.word.lower() for w in wc_on)

            # CSV reference
            csv_ref = id_to_ref.get(wav_path.stem)
            if csv_ref is not None:
                hyps_off.append(hyp_off if hyp_off else "")
                hyps_on.append(hyp_on if hyp_on else "")
                refs_off.append(csv_ref)
                refs_on.append(csv_ref)

            # Compare transcriptions
            if hyp_off != hyp_on:
                diff_lines.append(f"FILE: {wav_path.name}")
                diff_lines.append(f"  REF      : {csv_ref if csv_ref else '(no ref)'}")
                diff_lines.append(f"  flag=False: {hyp_off}")
                diff_lines.append(f"  flag=True : {hyp_on}")
                diff_lines.append("")

        except Exception as exc:
            print(f"\n  Error [{wav_path.name}]: {exc}")
            traceback.print_exc()
            skipped += 1

    # ── Print results ────────────────────────────────────────────────────
    print(f"\nSkipped files: {skipped}")
    print(f"Files evaluated: {len(hyps_off)}")

    print_stats("flag=False (default)", records_off, hyps_off, refs_off)
    print_stats("flag=True  (none_text heuristic)", records_on, hyps_on, refs_on)

    # ── Side-by-side comparison ──────────────────────────────────────────
    if records_off and records_on:
        lat_off = np.array([r["latency_ms"] for r in records_off])
        lat_on = np.array([r["latency_ms"] for r in records_on])
        print(f"\n{'='*65}")
        print(f"  Latency delta (flag=True - flag=False)")
        print(f"{'='*65}")
        print(f"  mean:  {lat_on.mean() - lat_off.mean():>+7.1f} ms")
        print(f"  p50 :  {np.median(lat_on) - np.median(lat_off):>+7.1f} ms")
        print(f"  p90 :  {np.percentile(lat_on, 90) - np.percentile(lat_off, 90):>+7.1f} ms")
        if hyps_off and hyps_on:
            wer_off = jiwer.wer(refs_off, hyps_off) * 100
            wer_on = jiwer.wer(refs_on, hyps_on) * 100
            print(f"  WER :  {wer_off:.2f}% -> {wer_on:.2f}%  (delta {wer_on - wer_off:>+.2f}%)")
        print(f"{'='*65}")

    # ── Write text.txt if transcriptions differ ──────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if diff_lines:
        text_path = out_dir / "text.txt"
        with open(text_path, "w", encoding="utf-8") as fh:
            fh.write(f"Transcription differences: flag=False vs flag=True\n")
            fh.write(f"Total files with differences: {sum(1 for l in diff_lines if l.startswith('FILE:'))}\n")
            fh.write(f"{'='*65}\n\n")
            for line in diff_lines:
                fh.write(line + "\n")
        print(f"\nTranscription differences saved to: {text_path}")
        print(f"  {sum(1 for l in diff_lines if l.startswith('FILE:'))} files had different transcriptions.")
    else:
        print("\nNo transcription differences between flag=False and flag=True.")

    # ── Save per-word CSVs ───────────────────────────────────────────────
    fieldnames = ["word", "gt_end_s", "audio_time_at_commit_s",
                  "model_processing_ms", "total_output_time_s", "latency_ms"]
    for label, recs in [("flag_false", records_off), ("flag_true", records_on)]:
        csv_out = out_dir / f"latency_{label}.csv"
        with open(csv_out, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(recs)
        print(f"  Saved: {csv_out}")


if __name__ == "__main__":
    main()
