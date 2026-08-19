#!/usr/bin/env python3
"""
measure_latency_word.py

Measures per-word streaming latency using StreamingASR
(SpeechBrain Conformer Transducer, SentencePiece/BPE word-piece level)
with chunk-based streaming.

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
  chunk_frames = StreamingASR.get_chunk_size_frames(config)
  Each step commits chunk_frames / SAMPLE_RATE seconds of new audio.

TextGrid files are expected to sit alongside the .wav files:
  <input_dir>/<split>/<spk>/<chap>/<utt>.wav
  <input_dir>/<split>/<spk>/<chap>/<utt>.TextGrid

Usage
-----
python measure_latency_word.py \\
    --hparams_file /path/to/chunk_streaming_word.yaml \\
    --checkpoint   /home/streamalign/streamASR/train/results/best_model.pt \\
    --input_dir /home/datasets/LibriSpeech \\
    --split test-clean \\
    [--chunk_size 4] \\
    [--left_context 32] \\
    [--max_files 100] \\
    [--output_csv latency_records_word.csv]
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

from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.checkpoints import Checkpointer
from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig
from speechbrain.utils.streaming import split_fixed_chunks
from speechbrain.inference.ASR import StreamingASR

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
# BoundaryClassifier helpers
# ---------------------------------------------------------------------------

def load_boundary_classifier(ckpt_path: str, device: torch.device):
    """Load a trained BoundaryClassifier from a checkpoint file."""
    from models.boundary_classifier import BoundaryClassifier
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    hp    = ckpt["hparams"]
    model = BoundaryClassifier(
        joint_dim  = hp["joint_dim"],
        hidden_dim = hp.get("hidden_dim", 512),
        num_layers = hp.get("num_layers", 3),
        dropout    = hp.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def compute_pred_feat(
    context,
    device: torch.device,
) -> torch.Tensor:
    """
    Extract the proj_dec output stored in the streaming context.

    The greedy searcher's decode_network_lst is [emb, dec, proj_dec], so
    _forward_PN runs the full emb → LSTM → proj_dec chain.  After each call
    to transducer_greedy_decode_streaming the context stores:

        context.decoder_context.hidden = (out_PN, rnn_hidden)
            out_PN    : Tensor [batch, 1, joint_dim]  — proj_dec output for
                        the token that will be used in the next blank decision
            rnn_hidden: LSTM (h, c) state

    We return out_PN[:, -1, :] directly — no re-computation needed.

    Returns
    -------
    pred_feat : Tensor [1, joint_dim]
    """
    hidden_state = context.decoder_context.hidden   # (out_PN, rnn_hidden)
    out_pn = hidden_state[0]                        # [batch, 1, joint_dim]
    return out_pn[:, -1, :].to(device)              # [1, joint_dim]


# ---------------------------------------------------------------------------
# StreamingASR chunk-based streaming with per-word timing
# ---------------------------------------------------------------------------
@torch.no_grad()
def stream_file_with_timing(
    asr_model: StreamingASR,
    wav_path: str,
    dynchunktrain_config: DynChunkTrainConfig,
    boundary_classifier=None,
    device: Optional[torch.device] = None,
    flag: bool = False,
) -> List[WordCommitInfo]:
    """
    Stream *wav_path* through StreamingASR (word-piece/BPE) in
    chunk-based streaming mode.

    For each step the decoded output is inspected. Every word that appears
    for the first time in this step is assigned:
        audio_time_at_commit  =  min((step + 1) × chunk_frames / SAMPLE_RATE,
                                     actual_file_duration)
        model_processing_time =  wall-clock seconds for this step's
                                 transcribe_chunk call (CUDA-synchronised).

    The SentencePiece tokenizer emits incremental tokens that are joined
    into text via `tokenizer_decode_streaming`; words are tracked by
    splitting the accumulated hypothesis on whitespace.

    Parameters
    ----------
    asr_model : StreamingASR
        Loaded StreamingASR (with SentencePiece tokenizer from
        chunk_streaming_word.yaml) in eval mode.
    wav_path : str
    dynchunktrain_config : DynChunkTrainConfig
        Streaming configuration (chunk_size, left_context_size).
    boundary_classifier : BoundaryClassifier, optional
        If provided, used at each chunk boundary to decide whether the last
        emitted subword ends a complete word.  Replaces the default heuristic
        (treat last word as tentative whenever new tokens arrived).
    device : torch.device, optional
        Device for boundary classifier tensors.  Inferred if not given.

    Returns
    -------
    list of WordCommitInfo, one entry per word in the final hypothesis.
    """
    if device is None:
        device = next(iter(asr_model.mods.parameters())).device
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
        asr_model.hparams.fea_streaming_extractor
        .get_recommended_final_chunk_count(chunk_frames)
    )
    flush_chunks = [torch.zeros((1, chunk_frames))] * final_chunk_count

    running_text = ""
    committed_count = 0   # words fully committed (all but the last tentative word)
    word_commits: List[WordCommitInfo] = []
    last_audio_time_end = 0.0
    last_model_time = 0.0

    # ── BoundaryClassifier setup ─────────────────────────────────────────────
    # We capture the proj_enc output during transcribe_chunk via a forward hook.
    # This gives us enc_feat = encoder projection at the last frame of the chunk,
    # which is the same tensor the greedy decoder uses for the blank decision.
    enc_proj_buf: List[torch.Tensor] = []
    hook_handle = None
    if boundary_classifier is not None:
        hook_handle = asr_model.mods["proj_enc"].register_forward_hook(
            lambda _m, _i, out: enc_proj_buf.append(out.detach())
        )
    for step_num, chunk in enumerate(itertools.chain(audio_chunks, flush_chunks)):
        # Pad last real chunk to full chunk_frames if shorter
        actual_samples = chunk.size(-1)
        if actual_samples < chunk_frames:
            chunk = F.pad(chunk, (0, chunk_frames - actual_samples))
            chunk_len = torch.tensor([actual_samples / chunk_frames])
        else:
            chunk_len = None  # defaults to ones([batch]) inside transcribe_chunk

        enc_proj_buf.clear()   # capture only this chunk's proj_enc output

        t_start = time.perf_counter()
        chunk_output = asr_model.transcribe_chunk(context, chunk, chunk_len)

        new_text = chunk_output[0] if chunk_output else ""
        running_text += new_text

        curr_words = running_text.replace("\u2581", " ").split() if running_text.strip() else []
        # ── Word boundary decision ────────────────────────────────────────────
        # With boundary_classifier: call the model on (enc_feat, pred_feat) at
        # the chunk boundary, where:
        #   enc_feat  = proj_enc output at the last encoder frame (captured via hook)
        #   pred_feat = re-running one LSTM step from context.decoder_context
        #               reproduces the exact pred_proj used for the blank decision
        #
        # is_boundary=True  → last word is complete, commit it now
        # is_boundary=False → last word may still grow in the next chunk, keep tentative
        #
        # Without boundary_classifier: fall back to the original heuristic
        # (no new text ↔ all words frozen; new text ↔ last word still tentative).
        is_boundary: Optional[bool] = None
        if boundary_classifier is not None:
            enc_proj = enc_proj_buf[0]                              # [..., T_chunk, joint_dim]
            enc_feat = enc_proj.reshape(-1, enc_proj.shape[-1])[-1].unsqueeze(0)  # [1, joint_dim]
            pred_feat = compute_pred_feat(context, device)                           # [1, joint_dim]
            logits = boundary_classifier(enc_feat.to(device), pred_feat)          # [1, 2]
            is_boundary = logits.argmax(-1).item() == 1

        # CUDA sync and wall-clock end are measured after the boundary classifier
        # so that model_time accounts for the full per-chunk inference cost.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_end = time.perf_counter()
        model_time = t_end - t_start

        # Audio clock: real chunks advance by chunk_frames/SAMPLE_RATE;
        # flush chunks are capped at actual duration.
        audio_time_end = min(
            (step_num + 1) * chunk_frames / SAMPLE_RATE,
            actual_audio_dur,
        )
        last_audio_time_end = audio_time_end
        last_model_time = model_time

        if flag and len(new_text)==0 : 
            none_text = True
        else:
            none_text = False

        if is_boundary is not None or none_text:
            confirmed_count = len(curr_words) if (is_boundary or none_text) else max(len(curr_words) - 1, 0)
        else:
            # Original heuristic: no new tokens → all frozen; new tokens → last tentative
            confirmed_count = max(len(curr_words) - 1, 0)

        for i in range(committed_count, confirmed_count):
            word_commits.append(WordCommitInfo(
                word=curr_words[i],
                audio_time_at_commit=audio_time_end,
                model_processing_time=model_time,
            ))
        committed_count = max(committed_count, confirmed_count)

    if hook_handle is not None:
        hook_handle.remove()

    # After all chunks: commit the last (now fully complete) word(s).
    curr_words = running_text.replace("\u2581", " ").split() if running_text.strip() else []
    for i in range(committed_count, len(curr_words)):
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
            "StreamingASR (SpeechBrain Conformer Transducer, SentencePiece/BPE), chunk-based streaming."
        )
    )
    p.add_argument("--hparams_file",
                   default="/home/streamalign/streamASR/hparams/chunk_streaming_word_fastemit.yaml",
                   help="Path to SpeechBrain hparams YAML for StreamingASR (word-piece/BPE).")
    p.add_argument("--checkpoint", type=str,
                   default="/home/streamalign/streamASR/results/conformer_transducer_char/word_fastemit/save/word_asr_ckpt",
                   help="Path to checkpoint directory or file (optional).")
    p.add_argument("--chunk_size", type=int, default=4,
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
    p.add_argument("--csv_dir", type=str, default="/home/datasets/LibriSpeech/csv",
                   help="Directory containing LibriSpeech CSV files (e.g. …/LibriSpeech/csv). "
                        "Defaults to <input_dir>/csv. Used for WER/CER reference text.")
    p.add_argument("--tokenizer_ckpt", type=str, default="/home/streamalign/streamASR/train/results/conformer_transducer_char/word_fastemit/pretrained/tokenizer.ckpt",
                   help="Explicit path to tokenizer.ckpt. If omitted, the script "
                        "searches <checkpoint>/../pretrained/tokenizer.ckpt and "
                        "the pretrain_folder declared in the hparams YAML.")
    p.add_argument("--boundary_classifier_ckpt", type=str, default="/home/streamalign/streamASR/train/results/boundary_classifier_0416/save/best_precision_model.pt",
                   help="Path to a trained BoundaryClassifier checkpoint "
                        "(best_model.pt produced by train/train_boundary_classifier.py). "
                        "When provided, the boundary model replaces the default heuristic "
                        "that treats the last word of each chunk as always tentative. "
                        "The model predicts at every chunk boundary whether the last "
                        "emitted subword ends a complete word, allowing earlier commits.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load StreamingASR (word-piece/BPE via chunk_streaming_word.yaml) ──
    print("Loading StreamingASR (word-piece/BPE) …")
    with open(args.hparams_file, encoding="utf-8") as f:
        hparams = load_hyperpyyaml(f)
    asr_model = StreamingASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts={"device": str(device)},
    )
    torch.set_grad_enabled(False)

    # ── Load checkpoint ───────────────────────────────────────────────────
    ckpt_dir = Path(args.checkpoint)
    if ckpt_dir.is_file():
        ckpt_dir = ckpt_dir.parent
    model_ckpt = ckpt_dir / "model.ckpt"
    norm_ckpt  = ckpt_dir / "normalizer.ckpt"

    if model_ckpt.exists():
        print(f"Loading model from: {model_ckpt}")
        state = torch.load(model_ckpt, map_location=device)
        missing, unexpected = hparams["model"].load_state_dict(state, strict=False)
        if missing:
            print(f"  Missing keys (will use init): {missing}")
        if unexpected:
            print(f"  Unexpected keys (ignored): {unexpected[:3]}{'...' if len(unexpected) > 3 else ''}")
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
            normalize.glob_std  = norm_state["glob_std"]
            normalize.count     = norm_state.get("count", 1)
            print("  Loaded normalizer stats")

    hparams["model"].eval()
    print("Model loaded!")

    # ── Load BoundaryClassifier (optional) ────────────────────────────────
    boundary_classifier = None
    if args.boundary_classifier_ckpt:
        print(f"Loading BoundaryClassifier from: {args.boundary_classifier_ckpt}")
        boundary_classifier = load_boundary_classifier(args.boundary_classifier_ckpt, device)
        print("  BoundaryClassifier loaded — boundary model will decide word commits.\n")

    # ── Ensure the SentencePiece tokenizer is loaded ──────────────────────
    # _load_checkpoint resolves pretrain_folder as a relative path from the
    # training CWD, which fails when running from a different directory.
    # We resolve it here with explicit fallback candidates.
    tokenizer = getattr(asr_model.hparams, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "Load"):
        tok_loaded = getattr(tokenizer, "vocab_size", lambda: 0)() > 0
        if not tok_loaded:
            candidates = []
            if args.tokenizer_ckpt:
                candidates.append(args.tokenizer_ckpt)
            # <checkpoint_dir>/../../pretrained/tokenizer.ckpt
            ckpt_dir = Path(args.checkpoint)
            candidates.append(str(ckpt_dir.parent.parent / "pretrained" / "tokenizer.ckpt"))
            # pretrain_folder from hparams (may be relative to train/ dir)
            pretrain_folder = getattr(asr_model.hparams, "pretrain_folder", None)
            if pretrain_folder:
                hparams_dir = Path(args.hparams_file).parent
                candidates.append(str(hparams_dir / pretrain_folder / "tokenizer.ckpt"))
                candidates.append(str(Path(pretrain_folder) / "tokenizer.ckpt"))
            for cand in candidates:
                if os.path.exists(cand):
                    print(f"Loading tokenizer from: {cand}")
                    tokenizer.Load(cand)
                    break
            else:
                raise FileNotFoundError(
                    "tokenizer.ckpt not found. Pass --tokenizer_ckpt <path>.\n"
                    f"Searched: {candidates}"
                )

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

    # ── Load CSV reference transcripts for WER/CER ───────────────────────
    csv_dir = Path(args.csv_dir) if args.csv_dir else Path(args.input_dir) / "csv"
    csv_path = csv_dir / f"{args.split}.csv"
    id_to_ref: Dict[str, str] = {}
    if csv_path.exists():
        with open(csv_path, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                id_to_ref[row["ID"]] = row["wrd"].lower()
        print(f"Loaded {len(id_to_ref)} reference transcripts from {csv_path}")
    else:
        print(f"Warning: CSV not found at {csv_path} — WER/CER will be skipped.")

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
        try:
            word_commits = stream_file_with_timing(
                asr_model, str(wav_path), dynchunktrain_config,
                boundary_classifier=boundary_classifier,
                device=device,
            )
            records = compute_file_latencies(word_commits, gt_intervals)

            all_records.extend(records)
            hyp_text = " ".join(w.word.lower() for w in word_commits)
            csv_ref = id_to_ref.get(wav_path.stem)
            if csv_ref is not None:
                all_hyps.append(hyp_text if hyp_text else "")
                all_refs.append(csv_ref)
        except Exception as exc:
            print(f"\n  Error [{wav_path.name}]: {exc}")
            traceback.print_exc()
            skipped += 1

    # ── Statistics ────────────────────────────────────────────────────────
    print(f"\nSkipped files (no TextGrid / error): {skipped}")
    print(f"Files evaluated : {len(all_hyps)}")
    print(f"Correctly recognized words measured: {len(all_records)}\n")

    if all_hyps:
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
    print("║  StreamingASR — SentencePiece/BPE, chunk-based               ║")
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
