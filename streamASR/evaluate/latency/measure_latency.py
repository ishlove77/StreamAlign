#!/usr/bin/env python3
"""
measure_latency.py

Measures per-word streaming latency using StreamingCharModel
(SpeechBrain Conformer Transducer, character-level) with chunk-based streaming.

Latency definition
------------------
For each correctly recognized word W the latency is:

    latency = (audio_time_at_commit + model_processing_time) − gt_end_time_W

Components
  audio_time_at_commit     : cumulative audio time (s) at the end of the
                             step in which word W first appears in the
                             streaming hypothesis.
                             = min((step_index + 1) × chunk_frames / SAMPLE_RATE,
                                   actual_file_duration)
                             Each step advances the audio clock by exactly
                             chunk_frames / SAMPLE_RATE seconds.

  model_processing_time    : wall-clock time (s) for that chunk's
                             transcribe_chunk call, CUDA-synchronised.

  gt_end_time_W            : word W's end time (s) from the Praat TextGrid.

"Correctly recognized" means the ASR-decoded word equals the ground-truth
word (case-insensitive), determined by word-level DP sequence alignment
between the decoded hypothesis and the TextGrid reference.

Chunk configuration
  Controlled by --chunk_size and --left_context (encoder output frames).
  chunk_frames = StreamingCharModel.get_chunk_size_frames(config)
  Each step commits chunk_frames / SAMPLE_RATE seconds of new audio.

TextGrid files are expected to sit alongside the .wav files:
  <input_dir>/<split>/<spk>/<chap>/<utt>.wav
  <input_dir>/<split>/<spk>/<chap>/<utt>.TextGrid

Usage
-----
python measure_latency.py \\
    --hparams_file /path/to/train_chunk_streaming.yaml \\
    --checkpoint   /path/to/checkpoint_dir \\
    --input_dir /home/datasets/LibriSpeech \\
    --split test-clean \\
    [--chunk_size 16] \\
    [--left_context 8] \\
    [--max_files 100] \\
    [--output_csv latency_records.csv]
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import csv
import itertools
import logging
import os
import sys
import traceback
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jiwer
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

# Allow imports from streamASR root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig
from speechbrain.utils.streaming import split_fixed_chunks
from models.model import StreamingCharModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16_000


# ---------------------------------------------------------------------------
# TextGrid parser  (words tier only)
# ---------------------------------------------------------------------------
@dataclass
class WordInterval:
    word: str
    start: float
    end: float


def parse_textgrid_words(path: str) -> List[WordInterval]:
    """Return non-silence intervals from the 'words' tier of a TextGrid."""
    intervals: List[WordInterval] = []
    in_words_tier = False
    in_interval = False
    cur_xmin = cur_xmax = cur_text = None

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if 'name = "words"' in line:
                in_words_tier = True
                continue
            if in_words_tier and line.startswith("name =") and '"words"' not in line:
                break
            if not in_words_tier:
                continue
            if line.startswith("intervals ["):
                in_interval = True
                cur_xmin = cur_xmax = cur_text = None
            elif in_interval:
                if line.startswith("xmin ="):
                    cur_xmin = float(line.split("=", 1)[1].strip())
                elif line.startswith("xmax ="):
                    cur_xmax = float(line.split("=", 1)[1].strip())
                elif line.startswith("text ="):
                    raw = line.split("=", 1)[1].strip().strip('"')
                    cur_text = raw
                    if cur_xmin is not None and cur_xmax is not None and cur_text:
                        intervals.append(WordInterval(cur_text, cur_xmin, cur_xmax))
                    in_interval = False

    return intervals


# ---------------------------------------------------------------------------
# Word-level DP sequence alignment
# ---------------------------------------------------------------------------
def align_sequences(hyp: List[str], ref: List[str]) -> List[Tuple[int, int]]:
    """
    Standard DP edit-distance alignment between hyp and ref (case-insensitive).

    Returns
    -------
    list of (hyp_idx, ref_idx)
        Index pairs where hyp[i].lower() == ref[j].lower()  (correct matches).
    """
    n, m = len(hyp), len(ref)
    INF = n + m + 1

    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if hyp[i - 1].lower() == ref[j - 1].lower() else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    matches: List[Tuple[int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if hyp[i - 1].lower() == ref[j - 1].lower() else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                if cost == 0:
                    matches.append((i - 1, j - 1))
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1

    return list(reversed(matches))


# ---------------------------------------------------------------------------
# WordCommitInfo – timing recorded for each word in the final hypothesis
# ---------------------------------------------------------------------------
@dataclass
class WordCommitInfo:
    word: str
    audio_time_at_commit: float   # seconds: end of the chunk where word first appeared
    model_processing_time: float  # seconds: wall-clock for that chunk's stream step
    #
    # total_output_time = audio_time_at_commit + model_processing_time
    # latency_W         = total_output_time    − gt_end_time_W


# ---------------------------------------------------------------------------
# StreamingCharModel chunk-based streaming with per-word timing
# ---------------------------------------------------------------------------
@torch.no_grad()
def stream_file_with_timing(
    asr_model: StreamingCharModel,
    wav_path: str,
    dynchunktrain_config: DynChunkTrainConfig,
) -> List[WordCommitInfo]:
    """
    Stream *wav_path* through StreamingCharModel in chunk-based streaming mode.

    For each step the decoded output is inspected. Every word that appears
    for the first time in this step is assigned:
        audio_time_at_commit  =  min((step + 1) × chunk_frames / SAMPLE_RATE,
                                     actual_file_duration)
        model_processing_time =  wall-clock seconds for this step's
                                 transcribe_chunk call (CUDA-synchronised).

    Each chunk emits incremental characters; words are tracked by splitting
    the accumulated hypothesis on whitespace.

    Parameters
    ----------
    asr_model : StreamingCharModel
        Loaded StreamingCharModel in eval mode.
    wav_path : str
    dynchunktrain_config : DynChunkTrainConfig
        Streaming configuration (chunk_size, left_context_size).

    Returns
    -------
    list of WordCommitInfo, one entry per word in the final hypothesis.
    """
    info = torchaudio.info(wav_path)
    actual_audio_dur: float = info.num_frames / info.sample_rate

    # Load audio as mono float32
    waveform, sr = torchaudio.load(wav_path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    waveform = waveform.mean(dim=0)  # [time]

    chunk_frames = asr_model.get_chunk_size_frames(dynchunktrain_config)
    context = asr_model.make_streaming_context(dynchunktrain_config)

    # Split audio into fixed-size chunks
    batch = waveform.unsqueeze(0)  # [1, time]
    audio_chunks = list(split_fixed_chunks(batch, chunk_frames))

    # Flush chunks: zero-padded to drain the streaming feature extractor
    final_chunk_count = (
        asr_model.hparams["fea_streaming_extractor"]
        .get_recommended_final_chunk_count(chunk_frames)
    )
    flush_chunks = [torch.zeros((1, chunk_frames))] * final_chunk_count

    running_text = ""
    prev_word_count = 0
    word_commits: List[WordCommitInfo] = []
    last_audio_time_end = 0.0
    last_model_time = 0.0

    for step_num, chunk in enumerate(itertools.chain(audio_chunks, flush_chunks)):
        # Pad last real chunk to full chunk_frames if shorter
        actual_samples = chunk.size(-1)
        if actual_samples < chunk_frames:
            chunk = F.pad(chunk, (0, chunk_frames - actual_samples))
            chunk_len = torch.tensor([actual_samples / chunk_frames])
        else:
            chunk_len = None  # defaults to ones([batch]) inside transcribe_chunk

        t_start = time.perf_counter()
        chunk_output = asr_model.transcribe_chunk(context, chunk, chunk_len)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_end = time.perf_counter()
        # transcribe_chunk returns a list of strings (one per batch item)
        new_text = chunk_output[0] if chunk_output else ""
        running_text += new_text
        print(running_text)
        if running_text:
            print(running_text[-1]==" ")
        model_time = t_end - t_start

        # Audio clock: real chunks advance by chunk_frames/SAMPLE_RATE;
        # flush chunks are capped at actual duration.
        audio_time_end = min(
            (step_num + 1) * chunk_frames / SAMPLE_RATE,
            actual_audio_dur,
        )
        last_audio_time_end = audio_time_end
        last_model_time = model_time

        # A word is complete only when the model has emitted a trailing space
        # (i.e., the next word has begun). Commit only finalized words so that
        # partial character sequences like "HO" are not recorded before the
        # full word "HOPED " is flushed.
        curr_words = running_text.strip().split() if running_text.strip() else []
        # Only words followed by a trailing space are finalized
        if running_text and running_text[-1] == " ":
            finalized = len(curr_words)
        else:
            finalized = max(0, len(curr_words) - 1)

        for i in range(prev_word_count, finalized):
            word_commits.append(WordCommitInfo(
                word=curr_words[i],
                audio_time_at_commit=audio_time_end,
                model_processing_time=model_time,
            ))
        prev_word_count = finalized

    # Commit the final word (no trailing space after last word in stream)
    curr_words = running_text.strip().split() if running_text.strip() else []
    for i in range(prev_word_count, len(curr_words)):
        word_commits.append(WordCommitInfo(
            word=curr_words[i],
            audio_time_at_commit=last_audio_time_end,
            model_processing_time=last_model_time,
        ))

    return word_commits


# ---------------------------------------------------------------------------
# Per-file latency computation
# ---------------------------------------------------------------------------
def compute_file_latencies(
    word_commits: List[WordCommitInfo],
    gt_intervals: List[WordInterval],
) -> List[Dict]:
    """
    Align decoded hypothesis words against TextGrid ground truth.

    For each correctly recognized word (hyp == ref, case-insensitive):

        total_output_time = audio_time_at_commit + model_processing_time
        latency           = total_output_time  −  gt_word_end_time

    Parameters
    ----------
    word_commits : list of WordCommitInfo  (one entry per hypothesis word)
    gt_intervals : list of WordInterval from TextGrid (non-silence only)

    Returns
    -------
    list of dicts with keys:
        word, gt_end_s, audio_time_at_commit_s,
        model_processing_ms, total_output_time_s, latency_ms
    """
    hyp_words = [wc.word for wc in word_commits]
    ref_words = [iv.word for iv in gt_intervals]
    ref_end_times = [iv.end for iv in gt_intervals]

    matches = align_sequences(hyp_words, ref_words)

    records = []
    for hyp_i, ref_j in matches:
        audio_t = word_commits[hyp_i].audio_time_at_commit
        model_t = word_commits[hyp_i].model_processing_time
        total_t = audio_t + model_t
        latency_s = total_t - ref_end_times[ref_j]

        records.append({
            "word":                   ref_words[ref_j],
            "gt_end_s":               ref_end_times[ref_j],
            "audio_time_at_commit_s": audio_t,
            "model_processing_ms":    model_t * 1000.0,
            "total_output_time_s":    total_t,
            "latency_ms":             latency_s * 1000.0,
        })
    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Measure per-word streaming latency — "
            "StreamingCharModel (SpeechBrain Conformer Transducer), chunk-based streaming."
        )
    )
    p.add_argument("--hparams_file",
                   default="/home/streamalign/streamASR/hparams/chunk_streaming_char.yaml",
                   help="Path to SpeechBrain hparams YAML for StreamingCharModel.")
    p.add_argument("--checkpoint", type=str, default="/home/streamalign/streamASR/train/results/conformer_transducer_char/char_asr/save/char_asr_ckpt",
                   help="Path to checkpoint directory or file (optional).")
    p.add_argument("--chunk_size", type=int, default=8,
                   help="DynChunkTrain chunk size in encoder output frames.")
    p.add_argument("--left_context", type=int, default=32,
                   help="DynChunkTrain left context size in encoder output frames.")
    p.add_argument("--input_dir",
                   default="/home/datasets/LibriSpeech",
                   help="Root LibriSpeech directory containing split sub-folders.")
    p.add_argument("--split", default="test-clean",
                   help="Dataset split name (must have TextGrid files alongside .wav).")
    p.add_argument("--max_files", type=int, default=None,
                   help="Limit number of .wav files processed (for quick testing).")
    p.add_argument("--output_csv", type=str, default=None,
                   help="Optional CSV file path to save per-word latency records.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load StreamingCharModel ───────────────────────────────────────────
    print("Loading StreamingCharModel …")
    asr_model = StreamingCharModel(
        hparams_file=args.hparams_file,
        checkpoint_path=args.checkpoint,
        device=str(device),
    )
    asr_model = asr_model.to(device).eval()
    torch.set_grad_enabled(False)

    dynchunktrain_config = DynChunkTrainConfig(
        chunk_size=args.chunk_size,
        left_context_size=args.left_context,
    )
    chunk_frames = asr_model.get_chunk_size_frames(dynchunktrain_config)
    chunk_ms = chunk_frames / SAMPLE_RATE * 1000.0
    print(
        f"Streaming config: chunk_size={args.chunk_size} enc-frames "
        f"({chunk_ms:.0f} ms per step), left_context={args.left_context} enc-frames\n"
    )

    # ── Gather wav files ──────────────────────────────────────────────────
    split_dir = Path(args.input_dir) / args.split
    wav_files = sorted(split_dir.glob("**/*.wav"))
    if args.max_files:
        wav_files = wav_files[: args.max_files]
    print(f"Found {len(wav_files)} .wav files in '{args.split}'.\n")

    all_records: List[Dict] = []
    all_hyps: List[str] = []
    all_refs: List[str] = []
    skipped = 0
    for wav_path in tqdm(wav_files, desc="Measuring latency"):
        tg_path = wav_path.with_suffix(".TextGrid")
        if not tg_path.exists():
            skipped += 1
            continue

        gt_intervals = parse_textgrid_words(str(tg_path))
        if not gt_intervals:
            skipped += 1
            continue
        ref_text = " ".join(iv.word.lower() for iv in gt_intervals)
        try:
            word_commits = stream_file_with_timing(asr_model, str(wav_path), dynchunktrain_config)
            records = compute_file_latencies(word_commits, gt_intervals)
            
            all_records.extend(records)
            hyp_text = " ".join(w.word.lower() for w in word_commits)
            ref_text = " ".join(iv.word.lower() for iv in gt_intervals)
            all_hyps.append(hyp_text if hyp_text else "")
            all_refs.append(ref_text)
        except Exception as exc:
            print(f"\n  Error [{wav_path.name}]: {exc}")
            traceback.print_exc()
            skipped += 1

    # ── Statistics ────────────────────────────────────────────────────────
    print(f"\nSkipped files (no TextGrid / error): {skipped}")
    print(f"Files evaluated : {len(all_hyps)}")
    print(f"Correctly recognized words measured: {len(all_records)}\n")

    if all_hyps:
        print(all_refs)
        print(all_hyps)
        wer = jiwer.wer(all_refs, all_hyps) * 100
        cer = jiwer.cer(all_refs, all_hyps) * 100
        print(f"WER : {wer:.2f}%")
        print(f"CER : {cer:.2f}%\n")

    if not all_records:
        print("No data collected — cannot compute latency statistics.")
        return

    lat_ms   = np.array([r["latency_ms"]            for r in all_records])
    model_ms = np.array([r["model_processing_ms"]   for r in all_records])
    audio_ms = (
        np.array([r["audio_time_at_commit_s"] for r in all_records]) -
        np.array([r["gt_end_s"]               for r in all_records])
    ) * 1000.0

    def _row(label, arr):
        return (f"║  {label:<8s}:"
                f"  mean={arr.mean():>7.1f}"
                f"  std={arr.std():>7.1f}"
                f"  p50={np.median(arr):>7.1f}"
                f"  p90={np.percentile(arr, 90):>7.1f}  ║")

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          Per-word Streaming Latency  (ms)                    ║")
    print("║  StreamingCharModel — chunk-based streaming                  ║")
    print("║  latency = audio_latency + model_processing_time             ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Words measured : {len(lat_ms)}".ljust(63) + "║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(_row("total",  lat_ms))
    print(_row("audio",  audio_ms))
    print(_row("model",  model_ms))
    print("╚══════════════════════════════════════════════════════════════╝")

    if args.output_csv:
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["word", "gt_end_s", "audio_time_at_commit_s",
                            "model_processing_ms", "total_output_time_s",
                            "latency_ms"],
            )
            writer.writeheader()
            writer.writerows(all_records)
        print(f"\nPer-word records saved to: {out_path}")


if __name__ == "__main__":
    main()
