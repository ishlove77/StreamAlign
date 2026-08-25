#!/usr/bin/env python3
"""
generate_chunk_textgrids.py

Streams LibriSpeech audio through the word-level StreamingASR model
(SpeechBrain Conformer Transducer, SentencePiece/BPE) and saves the
chunk-committed word sequences as Praat TextGrid files.

Each word in the output TextGrid has its xmax set to the end time of the
streaming chunk in which the word model first committed that word.  When the
character-level training script reads these TextGrids, it assigns each word's
character tokens to the corresponding encoder chunk — exactly mirroring what
the word model emitted per chunk.

Word boundary marker (▁ / U+2581) is treated as a space: the raw
SentencePiece hypothesis is split on whitespace after replacing ▁ with ' '.

Output
------
TextGrid files are written under --output_dir (default: same directory as
each .wav file), preserving the LibriSpeech sub-directory structure:

    <output_dir>/<split>/<spk>/<chap>/<utt>.TextGrid

or, when --output_dir is omitted:

    <input_dir>/<split>/<spk>/<chap>/<utt>.TextGrid

Existing TextGrid files are skipped unless --overwrite is passed.

Usage
-----
python generate_chunk_textgrids.py \\
    --hparams_file /path/to/chunk_streaming_word_fastemit.yaml \\
    --checkpoint   /path/to/checkpoint_dir \\
    --input_dir    <LIBRISPEECH_ROOT> \\
    --splits       train-clean-100 train-clean-360 \\
    [--chunk_size  4] \\
    [--left_context 32] \\
    [--output_dir  /path/to/chunk_textgrids] \\
    [--max_files   100] \\
    [--overwrite]
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import itertools
import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_STREAMASR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig
from speechbrain.utils.streaming import split_fixed_chunks
from speechbrain.inference.ASR import StreamingASR

SAMPLE_RATE = 16_000


# ---------------------------------------------------------------------------
# DataLoader: parallel MP3/WAV decode + resample
# ---------------------------------------------------------------------------
class WavDataset(Dataset):
    """Dataset of (row_index, waveform[time]) for parallel decode via num_workers."""

    def __init__(self, wav_paths: List[str]):
        self.wav_paths = wav_paths

    def __len__(self) -> int:
        return len(self.wav_paths)

    def __getitem__(self, idx: int):
        path = self.wav_paths[idx]
        try:
            waveform, sr = torchaudio.load(path)
            if sr != SAMPLE_RATE:
                waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
            waveform = waveform.mean(dim=0)  # [time]
            return idx, waveform, ""
        except Exception as exc:  # surface decode failure to main process
            return idx, torch.zeros(0), f"{type(exc).__name__}: {exc}"


def collate_keep_list(batch):
    """Return list of (idx, waveform, err) — no padding (we pad inside the
    streaming loop chunk-by-chunk to avoid materialising T_max*B tensors)."""
    return batch


# ---------------------------------------------------------------------------
# Streaming: collect per-chunk committed words (single utterance)
# ---------------------------------------------------------------------------
@torch.no_grad()
def stream_and_get_chunk_words(
    asr_model: StreamingASR,
    wav_path: str,
    dynchunktrain_config: DynChunkTrainConfig,
) -> Tuple[List[Tuple[str, int, int]], int]:
    """
    Stream *wav_path* through the word-level StreamingASR model.

    The raw token output of each chunk is recorded verbatim: ▁ (U+2581,
    the SentencePiece word-boundary marker) is replaced with a space but
    no further splitting or merging is performed.  This preserves the exact
    per-chunk alignment produced by the word model.

    For example, if the model emits "token" in chunk 0 and "ization▁hello"
    in chunk 1, the records are ("token", 0, 4) and ("ization hello", 4, 8)
    for enc_chunk_size=4 — "tokenization" is never reconstructed.

    Flush chunks are attributed to the last real audio chunk.
    Empty chunks emit ("", frame_start, frame_end) so every step has an entry.

    Returns
    -------
    chunk_commits  : list of (chunk_text: str, frame_start: int, frame_end: int)
                     frame_start / frame_end are encoder output frame indices.
                     chunk_text may contain spaces and may begin with a space
                     when the first token of the chunk starts a new word.
    n_audio_chunks : int — number of real (non-flush) audio chunks
    """
    waveform, sr = torchaudio.load(wav_path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    waveform = waveform.mean(dim=0)  # [time]

    chunk_frames = asr_model.get_chunk_size_frames(dynchunktrain_config)
    context = asr_model.make_streaming_context(dynchunktrain_config)

    batch = waveform.unsqueeze(0)  # [1, time]
    audio_chunks = list(split_fixed_chunks(batch, chunk_frames))
    n_audio_chunks = len(audio_chunks)

    final_chunk_count = (
        asr_model.hparams.fea_streaming_extractor
        .get_recommended_final_chunk_count(chunk_frames)
    )
    flush_chunks = [torch.zeros((1, chunk_frames))] * final_chunk_count

    enc_chunk_size = dynchunktrain_config.chunk_size  # encoder output frames/chunk
    # Audio samples per one encoder output frame (hop_size × CNN subsampling).
    samples_per_enc_frame = chunk_frames / enc_chunk_size

    chunk_commits: List[Tuple[str, int, int]] = []
    last_real_chunk_idx = max(n_audio_chunks - 1, 0)

    prev_frame_end       = 0   # running end of the last real audio chunk
    last_real_frame_start = 0  # frame_start of the last real audio chunk
    last_real_frame_end   = 0  # frame_end   of the last real audio chunk

    for step_num, chunk in enumerate(itertools.chain(audio_chunks, flush_chunks)):
        actual_samples = chunk.size(-1)
        if actual_samples < chunk_frames:
            chunk = F.pad(chunk, (0, chunk_frames - actual_samples))
            chunk_len = torch.tensor([actual_samples / chunk_frames])
        else:
            chunk_len = None

        chunk_output = asr_model.transcribe_chunk(context, chunk, chunk_len)
        new_text = chunk_output[0] if chunk_output else ""

        # Replace ▁ with space; preserve leading space (signals a new word).
        # Empty new_text is kept as "" so every step has an entry.
        chunk_text = new_text.replace("\u2581", " ")

        if step_num < n_audio_chunks:
            # Real audio chunk: frame_start picks up where the previous chunk
            # ended, so boundaries chain correctly even if a prior chunk
            # produced fewer than enc_chunk_size frames.
            frame_start       = prev_frame_end
            actual_enc_frames = max(1, round(actual_samples / samples_per_enc_frame))
            frame_end         = frame_start + actual_enc_frames
            prev_frame_end        = frame_end
            last_real_frame_start = frame_start
            last_real_frame_end   = frame_end
        else:
            # Flush chunk: reuse the last real chunk's boundaries so that any
            # text flushed out is attributed to the same interval.
            frame_start = last_real_frame_start
            frame_end   = last_real_frame_end

        chunk_commits.append((chunk_text, frame_start, frame_end))

    return chunk_commits, n_audio_chunks


# ---------------------------------------------------------------------------
# Streaming: collect per-chunk committed words (BATCHED)
# ---------------------------------------------------------------------------
@torch.no_grad()
def stream_and_get_chunk_words_batched(
    asr_model: StreamingASR,
    waveforms: List[torch.Tensor],
    dynchunktrain_config: DynChunkTrainConfig,
) -> List[Tuple[List[Tuple[str, int, int]], int]]:
    """Batched analogue of :func:`stream_and_get_chunk_words`.

    Runs B utterances through one shared streaming context in lockstep over
    ``N_max + final_chunk_count`` steps where ``N_max = max_i n_audio_chunks_i``.

    Per-utterance handling
    ----------------------
    * For step < n_audio_chunks_i:  the i-th slot carries the real audio chunk
      (zero-padded if shorter than ``chunk_frames``); ``chunk_len[i]`` is set
      to the partial-chunk fraction on the last real chunk, else 1.0.
    * For step >= n_audio_chunks_i: the i-th slot carries an all-zero chunk
      with ``chunk_len[i] = 1.0`` (silence ≈ flush).  Any text emitted in
      these steps is attributed to ``last_real_frame_{start,end}[i]`` — the
      same flush-attribution policy as single-stream.

    Sorting the batch by descending duration upstream minimises the number of
    extra silence steps short utterances are exposed to.

    Returns
    -------
    list of length B; each element is ``(chunk_commits, n_audio_chunks)``
    matching the single-utterance return signature.
    """
    B = len(waveforms)
    chunk_frames = asr_model.get_chunk_size_frames(dynchunktrain_config)
    enc_chunk_size = dynchunktrain_config.chunk_size
    samples_per_enc_frame = chunk_frames / enc_chunk_size

    final_chunk_count = (
        asr_model.hparams.fea_streaming_extractor
        .get_recommended_final_chunk_count(chunk_frames)
    )

    # Per-utterance chunking metadata
    n_audio_chunks: List[int] = []
    last_chunk_samples: List[int] = []  # samples in the last (possibly partial) audio chunk
    for w in waveforms:
        T = int(w.size(0))
        if T <= 0:
            n_audio_chunks.append(1)
            last_chunk_samples.append(0)
            continue
        n_full = T // chunk_frames
        rem = T % chunk_frames
        if rem == 0:
            n_audio_chunks.append(n_full)
            last_chunk_samples.append(chunk_frames)
        else:
            n_audio_chunks.append(n_full + 1)
            last_chunk_samples.append(rem)

    N_max = max(n_audio_chunks) if n_audio_chunks else 0
    n_steps = N_max + final_chunk_count

    context = asr_model.make_streaming_context(dynchunktrain_config)

    chunk_commits: List[List[Tuple[str, int, int]]] = [[] for _ in range(B)]
    prev_frame_end = [0] * B
    last_real_frame_start = [0] * B
    last_real_frame_end = [0] * B

    for step in range(n_steps):
        chunk_buf = torch.zeros((B, chunk_frames))
        chunk_len = torch.ones((B,))
        for i in range(B):
            n_i = n_audio_chunks[i]
            if step < n_i:
                start = step * chunk_frames
                if step == n_i - 1:
                    seg_len = last_chunk_samples[i]
                else:
                    seg_len = chunk_frames
                if seg_len > 0:
                    chunk_buf[i, :seg_len] = waveforms[i][start:start + seg_len]
                if step == n_i - 1 and seg_len < chunk_frames:
                    chunk_len[i] = seg_len / chunk_frames if chunk_frames > 0 else 1.0
                # else chunk_len[i] stays at 1.0
            # else: silence flush, chunk_len[i] = 1.0 already

        chunk_output = asr_model.transcribe_chunk(context, chunk_buf, chunk_len)

        for i in range(B):
            new_text = chunk_output[i] if chunk_output else ""
            chunk_text = new_text.replace("▁", " ")
            n_i = n_audio_chunks[i]
            if step < n_i:
                frame_start = prev_frame_end[i]
                if step == n_i - 1 and last_chunk_samples[i] < chunk_frames:
                    actual_enc_frames = max(
                        1, round(last_chunk_samples[i] / samples_per_enc_frame)
                    )
                else:
                    actual_enc_frames = enc_chunk_size
                frame_end = frame_start + actual_enc_frames
                prev_frame_end[i] = frame_end
                last_real_frame_start[i] = frame_start
                last_real_frame_end[i] = frame_end
            else:
                frame_start = last_real_frame_start[i]
                frame_end = last_real_frame_end[i]
            chunk_commits[i].append((chunk_text, frame_start, frame_end))

    return list(zip(chunk_commits, n_audio_chunks))


# ---------------------------------------------------------------------------
# TextGrid writer
# ---------------------------------------------------------------------------
def write_textgrid_chunks(
    path: str,
    chunk_commits: List[Tuple[str, int]],
    n_audio_chunks: int,
    enc_chunk_size: int,
) -> None:
    """
    Write a Praat TextGrid with a single 'words' interval tier using
    encoder frame ranges instead of audio timestamps.

    xmin / xmax in each interval are encoder output frame indices
    (integers stored as floats).  Chunk ci occupies exactly one interval:
        xmin = ci * enc_chunk_size
        xmax = (ci + 1) * enc_chunk_size

    The text of each interval is the concatenation of all text fragments
    the word model emitted during that chunk (multiple flush steps
    attributed to the same chunk are joined).  ▁ has already been replaced
    with a regular space by the caller; the result may have a leading space
    if the first token of the chunk was a new-word token (▁word).

    The training script recovers the chunk index from iv.end via:
        ci = max(0, (round(iv.end) - 1) // enc_chunk_size)

    Empty chunks (no output from the word model) are written as empty-text
    intervals; the training parser ignores them.

    File-level xmax = n_audio_chunks * enc_chunk_size (total encoder frames).
    """
    # Build per-chunk data: frame_start -> [concatenated_text, frame_end].
    # frame_end is taken from the first entry for each frame_start (the real
    # audio chunk step) so that the partial last chunk's trimmed frame_end is
    # preserved.  Subsequent entries for the same frame_start (flush steps)
    # only contribute their text.
    chunk_data: Dict[int, List] = {}
    for chunk_text, frame_start, frame_end in chunk_commits:
        if frame_start not in chunk_data:
            chunk_data[frame_start] = [chunk_text, frame_end]
        else:
            chunk_data[frame_start][0] += chunk_text

    # Build one interval per chunk using exact frame boundaries
    intervals: List[Tuple[float, float, str]] = []

    for ci in range(n_audio_chunks):
        seg_start = ci * enc_chunk_size
        default_end = (ci + 1) * enc_chunk_size
        if seg_start in chunk_data:
            text, seg_end = chunk_data[seg_start]
        else:
            text, seg_end = "", default_end
        intervals.append((float(seg_start), float(seg_end), text))

    total_frames = int(intervals[-1][1]) if intervals else n_audio_chunks * enc_chunk_size

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write('File type = "ooTextFile"\n')
        f.write('Object class = "TextGrid"\n\n')
        f.write('xmin = 0\n')
        f.write(f'xmax = {total_frames}\n')
        f.write('tiers? <exists>\n')
        f.write('size = 1\n')
        f.write('item []:\n')
        f.write('    item [1]:\n')
        f.write('        class = "IntervalTier"\n')
        f.write('        name = "words"\n')
        f.write('        xmin = 0\n')
        f.write(f'        xmax = {total_frames}\n')
        f.write(f'        intervals: size = {len(intervals)}\n')
        for idx, (xmin, xmax, text) in enumerate(intervals, 1):
            f.write(f'        intervals [{idx}]:\n')
            f.write(f'            xmin = {xmin:.6f}\n')
            f.write(f'            xmax = {xmax:.6f}\n')
            f.write(f'            text = "{text}"\n')


# ---------------------------------------------------------------------------
# Model loading (shared with measure_latency_word.py)
# ---------------------------------------------------------------------------
def load_word_asr_model(
    hparams_file: str,
    checkpoint: str,
    tokenizer_ckpt: Optional[str],
    device: torch.device,
) -> StreamingASR:
    """Load StreamingASR (word-piece/BPE) from hparams + checkpoint."""
    with open(hparams_file, encoding="utf-8") as f:
        hparams = load_hyperpyyaml(f)

    asr_model = StreamingASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts={"device": str(device)},
    )
    torch.set_grad_enabled(False)

    ckpt_dir = Path(checkpoint)
    if ckpt_dir.is_file():
        ckpt_dir = ckpt_dir.parent
    model_ckpt = ckpt_dir / "model.ckpt"
    norm_ckpt  = ckpt_dir / "normalizer.ckpt"

    if model_ckpt.exists():
        print(f"Loading model from: {model_ckpt}")
        state = torch.load(model_ckpt, map_location=device)
        missing, unexpected = hparams["model"].load_state_dict(state, strict=False)
        if missing:
            print(f"  Missing keys (init): {missing[:3]}{'...' if len(missing) > 3 else ''}")
        if unexpected:
            print(f"  Unexpected keys (ignored): {unexpected[:3]}{'...' if len(unexpected) > 3 else ''}")
    else:
        from speechbrain.utils.checkpoints import Checkpointer
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

    # Ensure SentencePiece tokenizer is loaded
    tokenizer = getattr(asr_model.hparams, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "Load"):
        tok_loaded = getattr(tokenizer, "vocab_size", lambda: 0)() > 0
        if not tok_loaded:
            candidates = []
            if tokenizer_ckpt:
                candidates.append(tokenizer_ckpt)
            candidates.append(str(ckpt_dir.parent.parent / "pretrained" / "tokenizer.ckpt"))
            pretrain_folder = getattr(asr_model.hparams, "pretrain_folder", None)
            if pretrain_folder:
                hparams_dir = Path(hparams_file).parent
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

    print("Word model loaded.\n")
    return asr_model


# ---------------------------------------------------------------------------
# Output path helper
# ---------------------------------------------------------------------------
def get_output_tg_path(
    wav_path: Path,
    input_root: Path,
    output_dir: Optional[Path],
) -> Path:
    """
    Derive the output TextGrid path.

    If output_dir is given the relative path of the wav under input_root
    is mirrored under output_dir.  Otherwise the TextGrid is placed
    alongside the wav file.
    """
    if output_dir is not None:
        rel = wav_path.relative_to(input_root)
        return (output_dir / rel).with_suffix(".TextGrid")
    return wav_path.with_suffix(".TextGrid")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Generate chunk-level Praat TextGrid files from the word-level "
            "StreamingASR model for use in character-model chunk-aligned training."
        )
    )
    p.add_argument(
        "--hparams_file",
        default=os.path.join(
            _STREAMASR_ROOT, "hparams", "chunk_streaming_word_fastemit.yaml"
        ),
        help="Path to the word-model hparams YAML (SentencePiece/BPE).",
    )
    p.add_argument(
        "--checkpoint",
        default=os.environ.get("WORD_ASR_CKPT", os.path.join(
            _STREAMASR_ROOT, "train", "results",
            "conformer_transducer_char", "word_fastemit", "save", "word_asr_ckpt",
        )),
        help="Path to the word-model checkpoint directory.",
    )
    p.add_argument(
        "--input_dir",
        default=os.environ.get("LIBRISPEECH_ROOT", "/data/LibriSpeech"),
        help="Root LibriSpeech directory.",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train-clean-100"],
        help="Dataset split(s) to process (e.g. train-clean-100 train-clean-360).",
    )
    p.add_argument(
        "--chunk_size", type=int, default=4,
        help="DynChunkTrain chunk size in encoder output frames.",
    )
    p.add_argument(
        "--left_context", type=int, default=32,
        help="DynChunkTrain left context in encoder output frames.",
    )
    p.add_argument(
        "--output_dir", type=str, default=None,
        help=(
            "Directory to write TextGrid files (mirrors LibriSpeech structure). "
            "If omitted, TextGrids are written alongside each .wav file."
        ),
    )
    p.add_argument(
        "--max_files", type=int, default=None,
        help="Limit number of .wav files per split (for quick testing).",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing TextGrid files (default: skip).",
    )
    p.add_argument(
        "--tokenizer_ckpt", type=str, default=None,
        help="Explicit path to tokenizer.ckpt (optional; auto-detected otherwise).",
    )
    p.add_argument(
        "--csv_manifest", type=str, default=None,
        help="Path to Emilia-style CSV manifest (ID,duration,wav,spk_id,wrd). "
             "When provided, --splits and --input_dir are ignored.",
    )
    p.add_argument(
        "--data_root", type=str, default=None,
        help="Value to substitute for $data_root in CSV wav paths. "
             "Defaults to the parent directory of --csv_manifest.",
    )
    p.add_argument(
        "--rank", type=int, default=0,
        help="GPU rank for multi-GPU sharding (0-indexed).",
    )
    p.add_argument(
        "--world_size", type=int, default=1,
        help="Total number of GPU processes for sharding.",
    )
    p.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size for batched streaming (>1 enables the batched path "
             "with one shared streaming context per batch).",
    )
    p.add_argument(
        "--num_workers", type=int, default=4,
        help="DataLoader worker processes for parallel MP3/WAV decode + resample.",
    )
    p.add_argument(
        "--sort_by_duration", action="store_true",
        help="Sort the (post-shard) manifest by descending duration before "
             "batching, to keep batches length-homogeneous and minimise the "
             "number of silence steps short utterances are exposed to.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.world_size > 1:
        print(f"[Rank {args.rank}/{args.world_size}] Device: {device}")
    else:
        print(f"Device: {device}")

    asr_model = load_word_asr_model(
        args.hparams_file, args.checkpoint, args.tokenizer_ckpt, device
    )

    dynchunktrain_config = DynChunkTrainConfig(
        chunk_size=args.chunk_size,
        left_context_size=args.left_context,
    )
    chunk_frames = asr_model.get_chunk_size_frames(dynchunktrain_config)
    chunk_ms = chunk_frames / SAMPLE_RATE * 1000.0
    print(
        f"Streaming config: chunk_size={args.chunk_size} enc-frames "
        f"({chunk_ms:.0f} ms/step), left_context={args.left_context} enc-frames\n"
    )

    total_written = 0
    total_skipped = 0
    total_errors  = 0

    if args.csv_manifest:
        import csv as csv_module
        csv_path  = Path(args.csv_manifest)
        data_root = args.data_root or str(csv_path.parent)
        input_root = Path(data_root)
        output_dir = Path(args.output_dir) if args.output_dir else None

        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv_module.DictReader(f))
        if args.max_files:
            rows = rows[: args.max_files]
        if args.world_size > 1:
            total_before = len(rows)
            rows = rows[args.rank::args.world_size]
            print(f"[emilia] Rank {args.rank}: {len(rows)}/{total_before} utterances from {csv_path.name}")
        else:
            print(f"[emilia] {len(rows)} utterances from {csv_path.name}")

        # Filter out rows whose TextGrid already exists (unless --overwrite)
        pending_rows: List[Tuple[Path, Path]] = []
        for row in rows:
            wav_path = Path(row["wav"].replace("$data_root", data_root))
            tg_path  = get_output_tg_path(wav_path, input_root, output_dir)
            if tg_path.exists() and not args.overwrite:
                total_skipped += 1
                continue
            pending_rows.append((wav_path, tg_path))

        if args.sort_by_duration and rows and "duration" in rows[0]:
            row_by_wav = {Path(r["wav"].replace("$data_root", data_root)): r for r in rows}
            def _dur(item):
                r = row_by_wav.get(item[0])
                try:
                    return -float(r["duration"]) if r else 0.0
                except Exception:
                    return 0.0
            pending_rows.sort(key=_dur)

        tqdm_desc = f"GPU {args.rank} [emilia]" if args.world_size > 1 else "Generating TextGrids [emilia]"

        if args.batch_size > 1 and len(pending_rows) > 0:
            # ---- Batched path: DataLoader prefetch + batched streaming ----
            wav_paths = [str(wp) for wp, _ in pending_rows]
            tg_paths  = [tp for _, tp in pending_rows]
            ds = WavDataset(wav_paths)
            loader = DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate_keep_list,
                pin_memory=False,
                drop_last=False,
                persistent_workers=(args.num_workers > 0),
            )
            with tqdm(total=len(pending_rows), desc=tqdm_desc, position=args.rank) as pbar:
                for batch in loader:
                    # Drop decode failures and zero-length waveforms
                    good: List[Tuple[int, torch.Tensor]] = []
                    for idx, wav, err in batch:
                        if err:
                            print(f"\n  Decode error [{wav_paths[idx]}]: {err}")
                            total_errors += 1
                            pbar.update(1)
                            continue
                        if int(wav.size(0)) <= 0:
                            print(f"\n  Empty waveform: {wav_paths[idx]}")
                            total_errors += 1
                            pbar.update(1)
                            continue
                        good.append((idx, wav))
                    if not good:
                        continue
                    waveforms = [w for _, w in good]
                    try:
                        results = stream_and_get_chunk_words_batched(
                            asr_model, waveforms, dynchunktrain_config
                        )
                    except Exception as exc:
                        print(f"\n  Batched stream error: {exc}")
                        traceback.print_exc()
                        total_errors += len(good)
                        pbar.update(len(good))
                        continue
                    for (idx, _), (chunk_commits, n_chunks) in zip(good, results):
                        try:
                            write_textgrid_chunks(
                                str(tg_paths[idx]), chunk_commits, n_chunks, args.chunk_size
                            )
                            total_written += 1
                        except Exception as exc:
                            print(f"\n  Write error [{wav_paths[idx]}]: {exc}")
                            traceback.print_exc()
                            total_errors += 1
                        pbar.update(1)
        else:
            # ---- Single-utterance path (preserved) ----
            for wav_path, tg_path in tqdm(pending_rows, desc=tqdm_desc, position=args.rank):
                try:
                    chunk_commits, n_chunks = stream_and_get_chunk_words(
                        asr_model, str(wav_path), dynchunktrain_config
                    )
                    write_textgrid_chunks(
                        str(tg_path), chunk_commits, n_chunks, args.chunk_size
                    )
                    total_written += 1
                except Exception as exc:
                    print(f"\n  Error [{wav_path.name}]: {exc}")
                    traceback.print_exc()
                    total_errors += 1

        print("[emilia] Done.\n")

    else:
        input_root = Path(args.input_dir)
        output_dir = Path(args.output_dir) if args.output_dir else None

        for split in args.splits:
            split_dir = input_root / split
            wav_files = sorted(split_dir.glob("**/*.wav")) or sorted(
                split_dir.glob("**/*.flac")
            )
            if args.max_files:
                wav_files = wav_files[: args.max_files]
            if args.world_size > 1:
                total_before = len(wav_files)
                wav_files = wav_files[args.rank::args.world_size]
                print(f"[{split}] Rank {args.rank}: {len(wav_files)}/{total_before} .wav files.")
            else:
                print(f"[{split}] Found {len(wav_files)} .wav files.")

            tqdm_desc = f"GPU {args.rank} [{split}]" if args.world_size > 1 else f"Generating TextGrids [{split}]"
            for wav_path in tqdm(wav_files, desc=tqdm_desc, position=args.rank):
                tg_path = get_output_tg_path(wav_path, input_root, output_dir)

                if tg_path.exists() and not args.overwrite:
                    total_skipped += 1
                    continue

                try:
                    chunk_commits, n_chunks = stream_and_get_chunk_words(
                        asr_model, str(wav_path), dynchunktrain_config
                    )
                    write_textgrid_chunks(
                        str(tg_path), chunk_commits, n_chunks, args.chunk_size
                    )
                    total_written += 1
                except Exception as exc:
                    print(f"\n  Error [{wav_path.name}]: {exc}")
                    traceback.print_exc()
                    total_errors += 1

            print(f"[{split}] Done.\n")

    print(
        f"Summary — written: {total_written}, "
        f"skipped (exists): {total_skipped}, "
        f"errors: {total_errors}"
    )
    if output_dir:
        print(f"TextGrids saved under: {output_dir}")


if __name__ == "__main__":
    main()
