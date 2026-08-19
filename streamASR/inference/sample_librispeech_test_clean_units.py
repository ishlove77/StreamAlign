#!/usr/bin/env python3
"""Sample LibriSpeech test-clean items, extract CosyVoice units, and resynthesize them."""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", message=".*StreamingMediaDecoder.*")
warnings.filterwarnings("ignore", message=".*TorchCodec.*")

_STREAMASR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STREAMASR_ROOT)

import torch
from hyperpyyaml import load_hyperpyyaml
from transformers import AutoTokenizer

from utils.cosyvoice_unit_utils import (
    DEFAULT_COSYVOICE_MODEL_DIR,
    DEFAULT_LIBRISPEECH_ROOT,
    DEFAULT_LIBRISPEECH_TEST_CSV,
    save_librispeech_test_clean_unit_examples,
)

sys.path.insert(0, "/home/CosyVoice")
from cosyvoice.cli.frontend import CosyVoiceFrontEnd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(_STREAMASR_ROOT, "checkpoints", "test_clean_unit_samples"),
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=DEFAULT_LIBRISPEECH_ROOT,
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default=DEFAULT_LIBRISPEECH_TEST_CSV,
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=DEFAULT_COSYVOICE_MODEL_DIR,
    )
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="llama",
        choices=["llama", "qwen3"],
    )
    return parser.parse_args()


def build_frontend(model_dir: str):
    with open(f"{model_dir}/cosyvoice3.yaml", "r") as f:
        configs = load_hyperpyyaml(f)
    return CosyVoiceFrontEnd(
        get_tokenizer=configs["get_tokenizer"],
        feat_extractor=configs["feat_extractor"],
        campplus_model=f"{model_dir}/campplus.onnx",
        speech_tokenizer_model=f"{model_dir}/speech_tokenizer_v3.onnx",
        allowed_special=configs["allowed_special"],
    )


def build_tokenizer(tokenizer_name: str):
    if tokenizer_name == "qwen3":
        glob_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    else:
        glob_tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    glob_tok.pad_token = glob_tok.eos_token
    return glob_tok


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frontend = build_frontend(args.model_dir)
    glob_tok = build_tokenizer(args.tokenizer)

    examples = save_librispeech_test_clean_unit_examples(
        output_dir=args.output_dir,
        frontend=frontend,
        glob_tok=glob_tok,
        device=device,
        num_samples=args.num_samples,
        seed=args.seed,
        test_csv=args.test_csv,
        input_dir=args.input_dir,
        model_dir=args.model_dir,
    )

    print(f"Saved {len(examples)} examples to {args.output_dir}")


if __name__ == "__main__":
    main()
