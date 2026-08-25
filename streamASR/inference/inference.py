#!/usr/bin/env python3
"""
Streaming inference: Data2VecSemanticAcousticModel → Qwen3-TTS codes → audio.

Replaces the CosyVoice-based pipeline. The model's forward_streaming() already
handles MTP code prediction and audio decoding via the frozen Qwen3-TTS
tokenizer model; this script handles I/O and batching.
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", message=".*StreamingMediaDecoder.*")
warnings.filterwarnings("ignore", message=".*TorchCodec.*")

import argparse
import os
import sys
import random
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

_QWEN3_REPO = "/home/Qwen3-TTS"
if _QWEN3_REPO not in sys.path:
    sys.path.insert(0, _QWEN3_REPO)

import torch
import torchaudio
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer

from models.model import Data2VecSemanticAcousticModel
from utils.data_utils_origin import LibriSpeechCSVDataset, unified_collate_fn
from utils.inference_utils import load_boundary_classifier


OUTPUT_SR = 24_000   # Qwen3-TTS codec decoder output sample rate


def _extract_spk_emb(wav_16k: torch.Tensor, spk_encoder, device) -> torch.Tensor:
    """Extract 1024-dim speaker embedding from a 16kHz mono waveform."""
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

    wav_24k = torchaudio.functional.resample(wav_16k, 16000, 24000)
    enc_dtype = next(spk_encoder.parameters()).dtype

    # Chunk to ≤30 s to avoid OOM
    max_samples = 30 * 24000
    chunk_embs = []
    with torch.no_grad():
        for start in range(0, wav_24k.shape[-1], max_samples):
            chunk = wav_24k[..., start:start + max_samples]
            mel = mel_spectrogram(
                chunk.unsqueeze(0).to(device),
                n_fft=1024, num_mels=128, sampling_rate=24000,
                hop_size=256, win_size=1024, fmin=0, fmax=12000,
            ).transpose(1, 2).to(enc_dtype)
            chunk_embs.append(spk_encoder(mel).squeeze(0).float())
    return torch.stack(chunk_embs).mean(dim=0)   # (1024,)


@torch.no_grad()
def infer_sample(model, wav_path: str, spk_emb: torch.Tensor,
                 tokenizer, word_asr_model, boundary_classifier) -> torch.Tensor:
    """
    Run streaming inference on one file.
    Returns concatenated audio (samples,) at OUTPUT_SR, or empty tensor.
    """
    chunks = []
    for audio_chunk in model.forward_streaming(
        wav_path=wav_path,
        spk_emb=spk_emb.unsqueeze(0),
        tokenizer=tokenizer,
        word_asr_model=word_asr_model,
        boundary_classifier=boundary_classifier,
        verbose=False,
    ):
        # audio_chunk: (samples,) float32 @24kHz
        chunks.append(audio_chunk)

    if not chunks:
        return torch.tensor([])
    return torch.cat(chunks, dim=0)


def run(rank: int, args):
    world_size = args.world_size
    if world_size > 1:
        dist.init_process_group(
            backend="nccl", init_method="env://",
            rank=rank, world_size=world_size,
        )
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print(f"Using {world_size} GPU(s).")

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset = LibriSpeechCSVDataset(args.test_csv, args.input_dir)
    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if world_size > 1 else None
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=4,
        collate_fn=unified_collate_fn,
        pin_memory=True,
    )

    # ── Text tokenizer ────────────────────────────────────────────────────────
    if args.tokenizer == "qwen3":
        glob_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    else:
        glob_tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    glob_tok.pad_token = glob_tok.eos_token

    # ── Student model ─────────────────────────────────────────────────────────
    student = Data2VecSemanticAcousticModel(
        vocab_size=3072,
        spk_emb_dim=2048,
        chunk_size=args.chunk_size,
        left_context=args.left_context,
        hparams_file=args.hparams,
        checkpoint_path=args.truthmodel_checkpoint_path,
    )

    checkpoint = torch.load(args.studentmodel_checkpoint_path, map_location="cpu")
    if "hubert_state_dict" in checkpoint:
        student_state = checkpoint["hubert_state_dict"]
    elif "model" in checkpoint:
        student_state = checkpoint["model"]
    else:
        student_state = checkpoint
    if any(k.startswith("module.") for k in student_state):
        student_state = {k.replace("module.", ""): v for k, v in student_state.items()}
    missing, unexpected = student.load_state_dict(student_state, strict=False)
    if rank == 0:
        if missing:
            print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")
        print("Student model loaded.")

    # ── Qwen3-TTS components (speaker encoder + code predictor + tokenizer) ──
    if rank == 0:
        print(f"Loading Qwen3-TTS from {args.qwen3_tts_path} …")
    student.load_qwen3_tts_components(args.qwen3_tts_path, args.qwen3_tok_path)
    student.to(device).eval()

    spk_encoder = student.qwen3_speaker_encoder   # frozen ECAPA-TDNN

    # ── Word ASR model ────────────────────────────────────────────────────────
    from utils.inference_utils import load_boundary_classifier
    from data.generate_chunk_textgrids import load_word_asr_model
    word_asr_model = load_word_asr_model(
        hparams_file=args.word_hparams,
        checkpoint=args.word_checkpoint,
        tokenizer_ckpt=getattr(args, "word_tokenizer_ckpt", None),
        device=device,
    )

    boundary_classifier = None
    if args.boundary_classifier_ckpt:
        boundary_classifier = load_boundary_classifier(args.boundary_classifier_ckpt, device)
        if rank == 0:
            print("BoundaryClassifier loaded.")

    # ── Inference loop ────────────────────────────────────────────────────────
    for batch in tqdm(loader, desc=f"Rank {rank}", disable=rank != 0):
        try:
            file_paths = batch["file_paths"]
            waveforms  = batch["waveforms"].to(device)
            wav_lens   = batch["wav_lens"].to(device)

            for i, wav_path in enumerate(file_paths):
                actual_len = int(wav_lens[i].item())
                wav_16k    = waveforms[i, :actual_len]

                spk_emb = _extract_spk_emb(wav_16k, spk_encoder, device)

                audio = infer_sample(
                    student, wav_path, spk_emb,
                    glob_tok, word_asr_model, boundary_classifier,
                )

                if audio.numel() == 0:
                    if rank == 0:
                        print(f"No audio generated for {wav_path}, skipping.")
                    continue

                rel  = os.path.relpath(wav_path, args.input_dir)
                outp = Path(args.output_dir) / rel
                outp.parent.mkdir(parents=True, exist_ok=True)
                torchaudio.save(str(outp), audio.unsqueeze(0), OUTPUT_SR)

        except Exception as e:
            import traceback
            print(f"Skip: {e}")
            traceback.print_exc()

    if dist.is_initialized():
        dist.barrier()
        if rank == 0:
            print("Done. Output dir:", args.output_dir)
        dist.destroy_process_group()
    else:
        print("Done. Output dir:", args.output_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    STREAMASR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--input_dir",   default=os.environ.get("LIBRISPEECH_ROOT", "/data/LibriSpeech"))
    parser.add_argument("--output_dir",  default=os.path.join(STREAMASR, "results", "inference_out"))
    parser.add_argument("--split",       default="test-clean")
    parser.add_argument("--chunk_size",  type=int, default=8)
    parser.add_argument("--left_context",type=int, default=32)
    parser.add_argument("--hparams",     default=f"{STREAMASR}/hparams/alignment.yaml")
    parser.add_argument("--truthmodel_checkpoint_path",
                        default=f"{STREAMASR}/results/char_asr_ckpt")
    parser.add_argument("--studentmodel_checkpoint_path", required=True,
                        help="Path to trained Data2VecSemanticAcousticModel checkpoint (.pt)")
    parser.add_argument("--qwen3_tts_path", required=True,
                        help="Path to Qwen3-TTS-12Hz-1.7B-Base model dir")
    parser.add_argument("--qwen3_tok_path", required=True,
                        help="Path to Qwen3-TTS-Tokenizer-12Hz dir")
    parser.add_argument("--test_csv",    required=True,
                        help="CSV file listing test set wav paths")
    parser.add_argument("--word_hparams", required=True,
                        help="HParams YAML for word-level StreamingASR model")
    parser.add_argument("--word_checkpoint", required=True,
                        help="Checkpoint for word-level StreamingASR model")
    parser.add_argument("--word_tokenizer_ckpt", default=None)
    parser.add_argument("--tokenizer", default="llama", choices=["llama", "qwen3"])
    parser.add_argument("--boundary_classifier_ckpt", default=None)
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count())
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.world_size > 1:
        local_rank_env = os.environ.get("LOCAL_RANK")
        if local_rank_env is not None:
            run(int(local_rank_env), args)
        else:
            torch.multiprocessing.spawn(run, nprocs=args.world_size, args=(args,))
    else:
        run(0, args)
