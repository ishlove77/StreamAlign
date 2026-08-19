#!/usr/bin/env python3
"""
Streaming Evaluation Char Version.

Usage:
    python evaluate_char.py /home/streamalign/streamASR/hparams/chunk_streaming_char.yaml \
        --checkpoint_path /home/streamalign/streamASR/results/char_asr_ckpt \
        --test_csv /home/datasets/LibriSpeech/csv/test-clean.csv \
        --num_samples 10
"""

import os
import sys
import argparse
import torch
import torchaudio
import json
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm
from typing import List, Optional

# Allow imports from the streamASR root when run from evaluate/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import re
import jiwer
from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig
from models.model import StreamingCharModel
import csv


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def evaluate_dataset(hparams, checkpoint_path, test_csv, chunk_size, left_context, num_samples=10):
    print(f"\nEvaluating dataset: {test_csv}")
    all_refs = []
    all_hyps = []

    with open(test_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        samples = list(reader)

    print(f"Total samples: {len(samples)}")
    display_samples = []

    model = StreamingCharModel(hparams, checkpoint_path, output_hidden_states=False, device="cuda" if torch.cuda.is_available() else "cpu").to(device="cuda" if torch.cuda.is_available() else "cpu")
    config = DynChunkTrainConfig(
        chunk_size=chunk_size,
        left_context_size=left_context,
    )
    samples = samples[:num_samples]
    for i, sample in enumerate(tqdm(samples)):
        wav_path = sample["wav"]
        reference = normalize(sample["wrd"])

        # Streaming
        pred = ""
        for decoded_chunk in model.transcribe_file_streaming(wav_path, config):
            pred += decoded_chunk
        pred = normalize(pred)

        if not reference:
            print(f"[Warning] Skipping {sample['ID']}: empty reference after normalization.")
            continue

        all_refs.append(reference)
        all_hyps.append(pred if pred else "empty")

        if len(display_samples) < num_samples:
            display_samples.append({
                "id": sample["ID"],
                "ref": reference,
                "streaming": pred,
            })

    # Results — refs/hyps already normalized, no transform needed
    cer = jiwer.cer(all_refs, all_hyps) * 100
    wer = jiwer.wer(all_refs, all_hyps) * 100
    
    print(f"\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Streaming: chunk={chunk_size}, left_ctx={left_context}")
    print(f"Samples: {len(samples)}")
    print("-"*40)
    print(f"Streaming CER:     {cer:.2f}%")
    print(f"Streaming WER:     {wer:.2f}%")
    print("="*70)
    
    
    # Sample predictions
    print(f"\nSAMPLE PREDICTIONS ({num_samples})")
    print("="*70)
    for i, s in enumerate(display_samples):
        print(f"\n[{i+1}] {s['id']}")
        print(f"  REF:  {s['ref']}")
        print(f"  STR:  {s['streaming']}")

def main():
    parser = argparse.ArgumentParser(description="SpeechBrain Streaming Evaluator")
    parser.add_argument("hparams", type=str, help="Path to hparams YAML")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--test_csv", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--left_context", type=int, default=16)
    
    args = parser.parse_args()
    evaluate_dataset(
        hparams=args.hparams,
        checkpoint_path=args.checkpoint_path,
        test_csv=args.test_csv,
        chunk_size=args.chunk_size,
        left_context=args.left_context,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()