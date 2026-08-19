"""End-to-end driver: run UTMOS / SECS / GPT-4o on continuation outputs.

Designed for the two delay-2 hier-durfirst-durreg emilia4000h checkpoints
(R16, R32_d16). Operates per model directory; reads ``*_with_prompt.wav``
plus the sidecar ``<stem>.json`` for transcripts.

UTMOS and SECS share a single GPU launch; the GPT judge is CPU-only (network
I/O bound) and can be skipped via ``--metrics``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as S
import sys
import time
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_INPUT_WAV_DIR = Path("/home/datasets/continuation/input_wavs")
DEFAULT_MODEL_DIRS = [
    Path("/home/datasets/continuation/"
         "streamalign_slm_r16"),
]


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "median": S.median(values),
        "std": S.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def _load_meta(model_dir: Path, stem: str) -> dict:
    p = model_dir / f"{stem}.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return {}


def _continuation_transcript(meta: dict) -> str:
    """Pick the best available transcript of the *continuation* portion."""
    return (
        meta.get("whisper_continuation", "")
        or meta.get("gen_text_subwords", "")
        or ""
    )


def _prompt_transcript(meta: dict, input_wav_dir: Path, stem: str) -> str:
    """Prefer the GT prompt transcript if present, else the model's own decode."""
    gt = input_wav_dir.parent / "gt_wavs" / f"{stem}.json"
    if gt.is_file():
        try:
            return json.loads(gt.read_text()).get("prompt", {}).get("text", "")
        except Exception:  # noqa: BLE001
            pass
    return meta.get("prompt_text_subwords", "")


def eval_model_dir(
    model_dir: Path,
    input_wav_dir: Path,
    metrics: set[str],
    out_dir: Path,
    gpt_model: str,
    max_utts: int = 0,
) -> dict:
    from . import _audio, utmos, secs, gpt_judge

    out_dir.mkdir(parents=True, exist_ok=True)

    wavs = _audio.list_with_prompt_wavs(model_dir)
    if max_utts > 0:
        wavs = wavs[:max_utts]
    if not wavs:
        raise SystemExit(f"no *_with_prompt.wav under {model_dir}")

    stems = [_audio.stem_from_with_prompt(w) for w in wavs]
    print(f"[run] {model_dir.name}: {len(wavs)} utts", flush=True)

    per_utt: dict[str, dict] = {s: {"stem": s} for s in stems}

    # --- UTMOS ---------------------------------------------------------- #
    if "utmos" in metrics:
        t0 = time.time()
        scores = utmos.score_wavs(wavs)
        dt = time.time() - t0
        with (out_dir / "utmos.jsonl").open("w") as fh:
            for w, s in zip(wavs, stems):
                v = scores[str(w)]
                per_utt[s]["utmos"] = v
                fh.write(json.dumps({"stem": s, "utmos": v}) + "\n")
        vals = [scores[str(w)] for w in wavs]
        print(f"[utmos] mean={sum(vals) / len(vals):.3f} ({dt:.1f}s)",
              flush=True)

    # --- SECS ----------------------------------------------------------- #
    if "secs" in metrics:
        pairs = []
        for w, s in zip(wavs, stems):
            ref = input_wav_dir / f"{s}.wav"
            if not ref.is_file():
                print(f"[secs] WARN: missing reference {ref}", flush=True)
                continue
            pairs.append((ref, w))
        t0 = time.time()
        scores = secs.score_pairs(pairs)
        dt = time.time() - t0
        with (out_dir / "secs.jsonl").open("w") as fh:
            for ref, gen in pairs:
                stem = _audio.stem_from_with_prompt(gen)
                v = scores[str(gen)]
                per_utt[stem]["secs"] = v
                fh.write(json.dumps({"stem": stem, "secs": v}) + "\n")
        vals = list(scores.values())
        print(f"[secs] mean={sum(vals) / len(vals):.4f} ({dt:.1f}s)",
              flush=True)

    # --- GPT-4o judge --------------------------------------------------- #
    if "gpt" in metrics:
        gpt_pairs = []
        for s in stems:
            meta = _load_meta(model_dir, s)
            prompt_text = _prompt_transcript(meta, input_wav_dir, s)
            cont_text = _continuation_transcript(meta)
            if not prompt_text or not cont_text:
                print(f"[gpt] WARN: missing text for {s} "
                      f"(prompt={bool(prompt_text)}, cont={bool(cont_text)})",
                      flush=True)
                continue
            gpt_pairs.append((s, prompt_text, cont_text))
        t0 = time.time()
        results = gpt_judge.score_pairs(gpt_pairs, model=gpt_model)
        dt = time.time() - t0
        with (out_dir / "gpt.jsonl").open("w") as fh:
            for s, _p, _c in gpt_pairs:
                r = results.get(s, {"score": None, "rationale": ""})
                per_utt[s]["gpt_score"] = r["score"]
                per_utt[s]["gpt_rationale"] = r["rationale"]
                fh.write(json.dumps({
                    "stem": s,
                    "gpt_score": r["score"],
                    "gpt_rationale": r["rationale"],
                }) + "\n")
        vals = [r["score"] for r in results.values() if r["score"] is not None]
        if vals:
            print(f"[gpt] mean={sum(vals) / len(vals):.3f} n={len(vals)} "
                  f"({dt:.1f}s)", flush=True)

    # --- combined CSV + summary ---------------------------------------- #
    fields = ["stem", "utmos", "secs", "gpt_score", "gpt_rationale"]
    csv_path = out_dir / "per_utt.csv"
    with csv_path.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        for s in stems:
            wr.writerow(per_utt[s])

    summary = {
        "model_dir": str(model_dir),
        "n_utts": len(wavs),
        "utmos": _summary([per_utt[s].get("utmos") for s in stems
                           if per_utt[s].get("utmos") is not None]),
        "secs": _summary([per_utt[s].get("secs") for s in stems
                          if per_utt[s].get("secs") is not None]),
        "gpt": _summary([per_utt[s].get("gpt_score") for s in stems
                         if per_utt[s].get("gpt_score") is not None]),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_dir",
        action="append",
        default=None,
        help="continuation output dir(s) (repeatable); defaults to the two "
             "delay-2 R16/R32_d16 emilia4000h dirs.",
    )
    ap.add_argument(
        "--input_wav_dir",
        type=Path,
        default=DEFAULT_INPUT_WAV_DIR,
    )
    ap.add_argument(
        "--metrics",
        default="utmos,secs,gpt",
        help="comma list from {utmos,secs,gpt}",
    )
    ap.add_argument(
        "--out_root",
        type=Path,
        default=REPO_ROOT / "logs" / "continuation_eval",
    )
    ap.add_argument("--gpt_model", default="gpt-4o-2024-08-06")
    ap.add_argument("--max_utts", type=int, default=0,
                    help="0 = all; otherwise smoke-test slice")
    return ap.parse_args()


def main():
    args = parse_args()
    metrics = {m.strip() for m in args.metrics.split(",") if m.strip()}
    unknown = metrics - {"utmos", "secs", "gpt"}
    if unknown:
        raise SystemExit(f"unknown metric(s): {sorted(unknown)}")

    model_dirs = (
        [Path(d) for d in args.model_dir]
        if args.model_dir else
        DEFAULT_MODEL_DIRS
    )

    args.out_root.mkdir(parents=True, exist_ok=True)
    all_summaries = {}
    for md in model_dirs:
        out_dir = args.out_root / md.name
        s = eval_model_dir(
            model_dir=md,
            input_wav_dir=args.input_wav_dir,
            metrics=metrics,
            out_dir=out_dir,
            gpt_model=args.gpt_model,
            max_utts=args.max_utts,
        )
        all_summaries[md.name] = s

    combined = args.out_root / "all_summaries.json"
    combined.write_text(json.dumps(all_summaries, indent=2))
    print(f"\n[done] combined summary: {combined}", flush=True)


if __name__ == "__main__":
    main()
