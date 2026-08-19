#!/usr/bin/env python3
"""
Evaluation Word Version using NVIDIA NeMo ASR.

Usage:
    python evaluate_word.py \
        --test_csv /home/datasets/test-clean-quality2.csv\
        --num_samples 10
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
import nemo.collections.asr as nemo_asr
from nemo.utils import logging as nemo_logging
import jiwer

nemo_logging.setLevel(logging.ERROR)


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    parser = argparse.ArgumentParser(description="NeMo ASR Evaluation")
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load model
    asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name="nvidia/stt_en_fastconformer_transducer_large")
    asr_model = asr_model.to(args.device)
    asr_model.eval()
    print(f"Model loaded on {args.device}")

    # Load test data
    with open(args.test_csv) as f:
        samples = list(csv.DictReader(f))

    print(f"Evaluating {len(samples)} samples...")

    all_refs = []
    all_hyps = []
    results = []
    total_audio_sec = 0.0
    total_time_sec = 0.0  
    for sample in tqdm(samples, desc="Evaluating"):
        wav_path = sample["wav"]
        if not os.path.exists(wav_path):
            continue
        ref = normalize(sample["wrd"])
        if not ref:
            print(f"[Warning] Skipping {sample['ID']}: empty reference after normalization.")
            continue

        # Audio duration
        info = torchaudio.info(wav_path)

        duration = info.num_frames / info.sample_rate
        total_audio_sec += duration

        # Transcribe
        t0 = time.perf_counter()
        logging.disable(logging.WARNING)
        with torch.no_grad():
            transcriptions = asr_model.transcribe([wav_path], batch_size=args.batch_size)
        logging.disable(logging.NOTSET)
        total_time_sec += time.perf_counter() - t0

        raw = transcriptions[0] if transcriptions else ""
        hyp = normalize(raw.text if hasattr(raw, "text") else str(raw))

        all_refs.append(ref)
        all_hyps.append(hyp if hyp else "empty")

        if len(results) < args.num_samples:
            results.append({"id": sample["ID"], "ref": ref, "hyp": hyp, "duration": duration})

    # Compute metrics — refs/hyps already normalized, no transform needed
    wer = jiwer.wer(all_refs, all_hyps) * 100
    cer = jiwer.cer(all_refs, all_hyps) * 100
    rtf = total_time_sec / total_audio_sec if total_audio_sec > 0 else float("inf")

    print(f"\n{'='*70}")
    print(f"Model : nvidia/stt_en_fastconformer_transducer_large")
    print(f"{'='*70}")
    print(f"WER: {wer:.2f}%")
    print(f"CER: {cer:.2f}%")
    print(f"Total audio : {total_audio_sec:.1f}s")
    print(f"Processing  : {total_time_sec:.1f}s")
    print(f"RTF         : {rtf:.4f}")
    print(f"{'='*70}")

    print(f"\nSample outputs ({len(results)} examples):")
    for i, r in enumerate(results):
        match = "✓" if r["ref"].lower() == r["hyp"].lower() else "✗"
        print(f"\n[{i+1}] {r['id']} ({r['duration']:.1f}s) {match}")
        print(f"  REF: {r['ref'].lower()}")
        print(f"  HYP: {r['hyp']}")


if __name__ == "__main__":
    main()
