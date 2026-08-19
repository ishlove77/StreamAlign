#!/usr/bin/env python3
"""SALMon evaluation for StreamSLM.

Mirrors TASTE-SpokenLM/salmon.py at the protocol level: each task folder
holds ``sample_<idx>_<0|1>.wav`` pairs (0 = positive / acoustically
consistent, 1 = negative). For every pair we compute a streamSLM score
on (audio, streaming-ASR TextGrid) and count the pair correct iff
``score_pos > score_neg``.

We use the streaming Conformer-Transducer's chunk TextGrid (one
``<wav_stem>.TextGrid`` per wav under ``--tg_root``), not the original
Whisper transcript. The Whisper text leaks future context into the
constrained RNN-T alignment that drives unit extraction; the chunk
TextGrid is what the SLM training data was built on, so eval must
match.

The score's exact definition is selected by ``--scoring_mode``:

* ``likelihood``: joint log-likelihood, sum reduction across text +
  acoustic positions. Standard SLM convention.
* ``loss`` (default): TASTE-SpokenLM convention — text is mean CE per
  token, each acoustic codebook is mean CE per position computed
  independently, the R channel-means are summed, and the score is the
  negation. Length-normalised.

Usage (1× 48 GB GPU, low quota):

    sr 1 48 --qos=q-low python -m streamSLM.eval.salmon \\
        --slm_checkpoint checkpoints/streamSLM/.../step_00050000.pt \\
        --output_dir     results/salmon

TextGrids default to alongside the wavs under ``--data_root``; pass
``--tg_root`` only if they live in a separate mirror.

Datasets default to the five in-scope SALMon variants; pass
``--datasets X Y`` to restrict. Results are written under
``<output_dir>/<ckpt_tag>/<mode>/`` so the two modes never overwrite
each other.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

# Allow ``python -m streamSLM.eval.salmon`` from the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from streamSLM.eval.extractor import (  # noqa: E402
    RVQUnitExtractor,
    DEFAULT_HPARAMS,
    DEFAULT_TEACHER,
    DEFAULT_TRUTHMODEL,
)
from streamSLM.eval.scorer import SCORING_MODES, StreamSLMScorer  # noqa: E402


DEFAULT_TASKS = [
    "energy_consistency",
    "gender_consistency",
    "pitch_consistency",
    "sentiment_consistency",
    "speaker_consistency",
]


# --------------------------------------------------------------------------- #
def _load_pairs(folder: Path, tg_folder: Path) -> List[dict]:
    """Group ``sample_<idx>_<ans>.wav`` + ``<stem>.TextGrid`` into (pos, neg) pairs.

    Wavs whose mirrored TextGrid is missing are dropped (and counted by the
    caller via ``missing_tg`` book-keeping). The Whisper transcript is no
    longer consulted.
    """
    samples: dict = {}
    missing_tg = 0
    for wav_path in sorted(folder.glob("sample_*_*.wav")):
        stem = wav_path.stem
        parts = stem.split("_")
        if len(parts) != 3:
            continue
        try:
            idx = int(parts[1]); ans = int(parts[2])
        except ValueError:
            continue
        tg_path = tg_folder / f"{stem}.TextGrid"
        if not tg_path.exists():
            missing_tg += 1
            continue
        entry = {"wav_path": str(wav_path), "tg_path": str(tg_path)}
        samples.setdefault(idx, {})["positive" if ans == 0 else "negative"] = entry

    pairs = []
    for idx, pair in samples.items():
        if "positive" in pair and "negative" in pair:
            pairs.append({"ind": idx, "positive": pair["positive"], "negative": pair["negative"]})
    pairs.sort(key=lambda p: p["ind"])
    if missing_tg:
        print(f"[load_pairs] {folder.name}: {missing_tg} wavs dropped — TextGrid missing under {tg_folder}",
              flush=True)
    return pairs


def _read_wav_16k(path: str) -> torch.Tensor:
    """Load mono 16 kHz waveform as a 1-D float tensor."""
    audio, sr = sf.read(path)
    wav = torch.from_numpy(audio).float()
    if wav.dim() > 1:
        wav = wav.mean(dim=-1)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav


def _score_one(extractor: RVQUnitExtractor, scorer: StreamSLMScorer,
               wav_path: str, tg_path: str) -> dict:
    wav = _read_wav_16k(wav_path)
    units = extractor.extract_one(wav, tg_path=tg_path)
    if units.num_subwords < 2:
        return {"score":         float("nan"),
                "text_nll":      float("nan"),
                "acoustic_nll":  float("nan"),
                "text_loss":     float("nan"),
                "acoustic_loss": float("nan"),
                "n_text_tokens": 0, "n_acoustic_positions": 0,
                "n_subwords":    int(units.num_subwords)}
    s = scorer.score_units(units)
    s["n_subwords"] = int(units.num_subwords)
    return s


# --------------------------------------------------------------------------- #
def evaluate_task(
    extractor: RVQUnitExtractor,
    scorer: StreamSLMScorer,
    task_name: str,
    pairs: List[dict],
) -> dict:
    correct = 0
    total = 0
    invalid = 0
    scores = []
    pbar = tqdm(pairs, desc=task_name, ncols=100)
    for pair in pbar:
        try:
            pos = _score_one(extractor, scorer, pair["positive"]["wav_path"], pair["positive"]["tg_path"])
            neg = _score_one(extractor, scorer, pair["negative"]["wav_path"], pair["negative"]["tg_path"])
        except Exception as e:
            print(f"[{task_name}] pair {pair['ind']} failed: {e}", flush=True)
            invalid += 1
            continue
        if pos["n_subwords"] < 2 or neg["n_subwords"] < 2:
            invalid += 1
            continue

        is_correct = pos["score"] > neg["score"]
        correct += int(is_correct)
        total += 1
        scores.append({
            "ind":           pair["ind"],
            "pos_score":     pos["score"],     "neg_score":     neg["score"],
            "pos_text_nll":  pos["text_nll"],  "neg_text_nll":  neg["text_nll"],
            "pos_aco_nll":   pos["acoustic_nll"],  "neg_aco_nll":   neg["acoustic_nll"],
            "pos_text_loss": pos["text_loss"], "neg_text_loss": neg["text_loss"],
            "pos_aco_loss":  pos["acoustic_loss"], "neg_aco_loss":  neg["acoustic_loss"],
            "pos_n":         pos["n_subwords"], "neg_n":         neg["n_subwords"],
            "correct":       bool(is_correct),
        })
        pbar.set_postfix(acc=f"{correct/max(total,1):.3f}", inv=invalid)

    accuracy = correct / total if total > 0 else 0.0
    return {
        "task":         task_name,
        "scoring_mode": scorer.scoring_mode,
        "accuracy":     accuracy,
        "correct":      correct,
        "total":        total,
        "invalid":      invalid,
        "scores":       scores,
    }


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slm_checkpoint", required=True,
                    help="StreamSLM .pt produced by streamSLM.train.train.")
    ap.add_argument("--data_root", default="/home/datasets/SALMon")
    ap.add_argument("--tg_root", default=None,
                    help="Directory holding <task>/<stem>.TextGrid files "
                         "generated by streamSLM/scripts/run_eval_tg_gen.sh. "
                         "Defaults to --data_root (TGs are written alongside "
                         "the wavs). Pair-wise eval skips any wav whose "
                         "TextGrid is missing.")
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_TASKS,
                    help=f"SALMon task folder names (default: {DEFAULT_TASKS}).")
    ap.add_argument("--output_dir", default="results/salmon")
    # StreamAlign teacher config (defaults match the released C=512 R=16 stack).
    ap.add_argument("--teacher_checkpoint", default=DEFAULT_TEACHER)
    ap.add_argument("--variant", choices=["rvq"], default="rvq",
                    help="Teacher quantizer family (default: rvq, matches the "
                         "current canonical hier-durfirst-durreg sweep).")
    ap.add_argument("--hparams", default=DEFAULT_HPARAMS)
    ap.add_argument("--truthmodel_checkpoint", default=DEFAULT_TRUTHMODEL)
    ap.add_argument("--chunk_size", type=int, default=16)
    ap.add_argument("--left_context", type=int, default=8)
    ap.add_argument("--rvq_num_quantizers", type=int, default=16,
                    help="Used when --variant=rvq (R, number of residual quantizers).")
    ap.add_argument("--rvq_codebook_size", type=int, default=512,
                    help="Used when --variant=rvq (per-quantizer codebook size K).")
    ap.add_argument("--text_tokenizer", choices=["llama", "qwen3"], default="llama")
    ap.add_argument("--no_bf16", action="store_true",
                    help="Disable bf16 autocast (default: bf16 on CUDA).")
    ap.add_argument("--scoring_mode", choices=list(SCORING_MODES),
                    default="loss",
                    help="Pair-classification score: 'loss' (default) = "
                         "TASTE-style sum of per-stream mean cross-entropy "
                         "(length-normalised); 'likelihood' = joint "
                         "log-likelihood (sum reduction, classical SLM "
                         "convention).")
    ap.add_argument("--text_weight", type=float, default=1.0,
                    help="Weight on the text-stream term in the pair score.")
    ap.add_argument("--acoustic_weight", type=float, default=1.0,
                    help="Weight on the acoustic-stream term in the pair score.")
    ap.add_argument("--acoustic_ch_weights", default="",
                    help="Comma-separated per-codebook weights (length R) for "
                         "the acoustic CEs in 'loss' mode. Empty = uniform sum "
                         "over the R codebooks.")
    ap.add_argument("--rank", type=int, default=0,
                    help="Shard index (0..world-1) for parallel evaluation.")
    ap.add_argument("--world", type=int, default=1,
                    help="Total number of shards (>=1). Pairs are sliced "
                         "pairs[rank::world] before evaluation.")
    args = ap.parse_args()
    if args.world < 1 or not (0 <= args.rank < args.world):
        raise SystemExit(f"bad sharding: rank={args.rank} world={args.world}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bf16 = (not args.no_bf16)
    print(f"[init] device={device} bf16={bf16}", flush=True)

    # Output namespacing — one folder per checkpoint × scoring mode so the
    # two modes are kept side-by-side without overwriting.
    ckpt_path = Path(args.slm_checkpoint)
    parts = ckpt_path.parts
    # Use the checkpoint's parent dir (experiment name) so two configs that
    # happen to share a step number don't collide. Falls back to grandparent
    # if the parent name doesn't exist (very short paths).
    part1 = parts[-2] if len(parts) >= 2 else parts[0]
    part2 = ckpt_path.stem
    out_base = Path(args.output_dir) / f"{part1}_{part2}" / args.scoring_mode
    out_base.mkdir(parents=True, exist_ok=True)
    print(f"[init] scoring_mode={args.scoring_mode} "
          f"text_weight={args.text_weight} "
          f"acoustic_weight={args.acoustic_weight} "
          f"acoustic_ch_weights={args.acoustic_ch_weights or '<uniform>'}",
          flush=True)
    print(f"[init] writing results to {out_base}", flush=True)

    print(f"[load] teacher ({args.variant}) {args.teacher_checkpoint}", flush=True)
    extractor = RVQUnitExtractor(
        teacher_checkpoint=args.teacher_checkpoint,
        variant=args.variant,
        hparams=args.hparams,
        truthmodel_checkpoint=args.truthmodel_checkpoint,
        chunk_size=args.chunk_size,
        left_context=args.left_context,
        rvq_num_quantizers=args.rvq_num_quantizers,
        rvq_codebook_size=args.rvq_codebook_size,
        text_tokenizer=args.text_tokenizer,
        device=device,
        bf16=bf16,
    )
    print(f"[load] StreamSLM {args.slm_checkpoint}", flush=True)
    aco_ch_w = None
    if args.acoustic_ch_weights.strip():
        aco_ch_w = [float(x) for x in args.acoustic_ch_weights.split(",")]
    scorer = StreamSLMScorer(
        args.slm_checkpoint,
        device=device,
        bf16=bf16,
        scoring_mode=args.scoring_mode,
        text_weight=args.text_weight,
        acoustic_weight=args.acoustic_weight,
        acoustic_ch_weights=aco_ch_w,
    )

    tg_root = args.tg_root if args.tg_root else args.data_root
    overall_summary: dict = {}
    sharded = args.world > 1
    suffix = f".shard{args.rank}_of{args.world}" if sharded else ""
    for task in args.datasets:
        folder = Path(args.data_root) / task
        tg_folder = Path(tg_root) / task
        if not folder.exists():
            print(f"[skip] {task}: {folder} missing", flush=True)
            continue
        if not tg_folder.exists():
            print(f"[skip] {task}: TextGrid folder {tg_folder} missing — "
                  f"run streamSLM/scripts/run_eval_tg_gen.sh first", flush=True)
            continue
        pairs = _load_pairs(folder, tg_folder)
        if sharded:
            pairs = pairs[args.rank::args.world]
            print(f"[task] {task}: shard {args.rank}/{args.world} got "
                  f"{len(pairs)} pairs from {folder}", flush=True)
        else:
            print(f"[task] {task}: {len(pairs)} pairs from {folder}", flush=True)
        if not pairs:
            continue

        t0 = time.time()
        result = evaluate_task(extractor, scorer, task, pairs)
        elapsed = time.time() - t0
        print(f"[task] {task}: acc={result['accuracy']:.4f} "
              f"({result['correct']}/{result['total']}) "
              f"invalid={result['invalid']} in {elapsed:.0f}s", flush=True)

        # Per-task JSON with full per-pair scores.
        out_path = out_base / f"salmon_{task}{suffix}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        overall_summary[task] = {
            "accuracy": result["accuracy"],
            "correct":  result["correct"],
            "total":    result["total"],
            "invalid":  result["invalid"],
        }

    summary_path = out_base / (f"salmon_summary{suffix}.json" if sharded
                               else "salmon_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {"scoring_mode":    args.scoring_mode,
             "text_weight":     args.text_weight,
             "acoustic_weight": args.acoustic_weight,
             "tasks":           overall_summary},
            f, indent=2,
        )
    print(f"[done] summary -> {summary_path}", flush=True)
    if overall_summary:
        mean_acc = sum(v["accuracy"] for v in overall_summary.values()) / len(overall_summary)
        print(f"[done] scoring_mode={args.scoring_mode} mean accuracy across "
              f"{len(overall_summary)} tasks: {mean_acc:.4f}", flush=True)


if __name__ == "__main__":
    main()
