#!/usr/bin/env python3
"""WER/CER evaluation using OpenAI Whisper (HF transformers).

Sister of evaluate_word.py but with a Whisper backbone. Same CSV input
format (ID, wav, wrd) so it pairs with build_recon_csv.py.

Usage:
    python evaluate_word_whisper.py \
        --test_csv results/wer/test-clean_<EXP>_recon.csv \
        --model openai/whisper-large-v3 \
        --batch_size 8 \
        --num_samples 5
"""
import argparse
import csv
import os
import re
import time
from tqdm import tqdm

import logging
import warnings
warnings.filterwarnings("ignore")

import torch
import torchaudio
import jiwer

from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_audio_for_pipeline(path: str) -> dict:
    wav, sr = torchaudio.load(path)
    if wav.ndim == 2:
        wav = wav.mean(dim=0)
    else:
        wav = wav.reshape(-1)
    return {"array": wav.cpu().numpy(), "sampling_rate": sr}


def main():
    parser = argparse.ArgumentParser(description="Whisper ASR Evaluation")
    parser.add_argument("--test_csv", type=str, required=True,
                        help="CSV with columns: ID,wav,wrd")
    parser.add_argument("--model", type=str, default="openai/whisper-large-v3",
                        help="HF Whisper model id, e.g. openai/whisper-large-v3, "
                             "openai/whisper-large-v3-turbo, openai/whisper-medium.en")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=5,
                        help="How many sample (ref/hyp) pairs to print at the end")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--language", type=str, default="english",
                        help="Whisper transcription language (default: english)")
    parser.add_argument("--chunk_length_s", type=int, default=30,
                        help="Whisper long-form chunk length in seconds")
    parser.add_argument("--hyps_out", type=str, default=None,
                        help="Optional TSV path to dump per-utterance hyps "
                             "as 'ID<tab>hyp_raw'. Parent dir is created.")
    args = parser.parse_args()

    torch_dtype = torch.float16 if "cuda" in args.device else torch.float32

    print(f"Loading {args.model} on {args.device} (dtype={torch_dtype})...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model, torch_dtype=torch_dtype, low_cpu_mem_usage=True
    ).to(args.device)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model)

    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=args.device,
        chunk_length_s=args.chunk_length_s,
    )

    gen_kwargs = {"language": args.language, "task": "transcribe"}

    with open(args.test_csv) as f:
        samples = list(csv.DictReader(f))
    print(f"Evaluating {len(samples)} samples...")

    # Pre-filter samples that have an existing wav and non-empty reference,
    # so the pipeline batch path doesn't blow up on missing files.
    items = []
    for s in samples:
        wav_path = s["wav"]
        if not os.path.exists(wav_path):
            continue
        ref = normalize(s["wrd"])
        if not ref:
            continue
        items.append((s["ID"], wav_path, ref))
    print(f"  kept {len(items)} after filtering missing/empty refs")

    all_refs, all_hyps, results = [], [], []
    raw_hyps_by_id: dict[str, str] = {}
    total_audio_sec = 0.0
    total_time_sec = 0.0

    paths = [it[1] for it in items]
    refs = [it[2] for it in items]
    ids = [it[0] for it in items]
    durations = []
    for p in paths:
        info = torchaudio.info(p)
        durations.append(info.num_frames / info.sample_rate)
    total_audio_sec = sum(durations)

    t0 = time.perf_counter()
    outputs = []
    bs = args.batch_size
    with torch.no_grad():
        for i in tqdm(range(0, len(paths), bs), desc="Whisper"):
            batch_paths = paths[i:i + bs]
            batch_audio = []
            for p in batch_paths:
                try:
                    batch_audio.append(load_audio_for_pipeline(p))
                except Exception as e:
                    batch_audio.append(None)
                    print(f"  [warn] failed to load {p}: {e}")
            try:
                valid_audio = [x for x in batch_audio if x is not None]
                valid_outs = []
                if valid_audio:
                    valid_outs = asr(valid_audio, batch_size=len(valid_audio),
                                generate_kwargs=gen_kwargs)
                    if isinstance(valid_outs, dict):
                        valid_outs = [valid_outs]
                valid_iter = iter(valid_outs)
                outs = [
                    next(valid_iter) if audio is not None else {"text": ""}
                    for audio in batch_audio
                ]
            except Exception as e:
                # Fall back to single-item on batch failure
                outs = []
                for p, audio in zip(batch_paths, batch_audio):
                    if audio is None:
                        outs.append({"text": ""})
                        continue
                    try:
                        outs.append(asr(audio, generate_kwargs=gen_kwargs))
                    except Exception as ee:
                        outs.append({"text": ""})
                        print(f"  [warn] failed on {p}: {ee}")
            outputs.extend(outs)
    total_time_sec = time.perf_counter() - t0

    for (uid, _, ref), dur, out in zip(items, durations, outputs):
        text = out.get("text", "") if isinstance(out, dict) else str(out)
        raw = text.strip()
        hyp = normalize(text)
        all_refs.append(ref)
        all_hyps.append(hyp if hyp else "empty")
        raw_hyps_by_id[uid] = raw
        if len(results) < args.num_samples:
            results.append({"id": uid, "ref": ref, "hyp": hyp, "duration": dur})

    wer = jiwer.wer(all_refs, all_hyps) * 100
    cer = jiwer.cer(all_refs, all_hyps) * 100
    rtf = total_time_sec / total_audio_sec if total_audio_sec > 0 else float("inf")

    print(f"\n{'='*70}")
    print(f"Model : {args.model}")
    print(f"{'='*70}")
    print(f"WER: {wer:.2f}%")
    print(f"CER: {cer:.2f}%")
    print(f"Total audio : {total_audio_sec:.1f}s")
    print(f"Processing  : {total_time_sec:.1f}s")
    print(f"RTF         : {rtf:.4f}")
    print(f"{'='*70}")

    if args.hyps_out:
        out_path = args.hyps_out
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write("ID\thyp_raw\n")
            for uid, raw in raw_hyps_by_id.items():
                fh.write(f"{uid}\t{raw}\n")
        print(f"Wrote {len(raw_hyps_by_id)} per-utterance hyps to {out_path}")

    print(f"\nSample outputs ({len(results)} examples):")
    for i, r in enumerate(results):
        match = "✓" if r["ref"].lower() == r["hyp"].lower() else "✗"
        print(f"\n[{i+1}] {r['id']} ({r['duration']:.1f}s) {match}")
        print(f"  REF: {r['ref'].lower()}")
        print(f"  HYP: {r['hyp']}")


if __name__ == "__main__":
    main()
