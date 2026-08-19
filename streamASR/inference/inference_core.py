#!/usr/bin/env python3
"""
sr 1 24 --qos=q-low python inference_libritts_resynthesis.py --checkpoint checkpoints/mas_char_flow_noteacher_noiseaug_agg_glowdur/epoch_52.pt --split test-clean
"""
import warnings
# Suppress torchaudio deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", message=".*StreamingMediaDecoder.*")
warnings.filterwarnings("ignore", message=".*TorchCodec.*")
import argparse
import os
import sys
import random
from pathlib import Path
from contextlib import nullcontext

# Allow imports from streamASR root when run from inference/ subdirectory
_STREAMASR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STREAMASR_ROOT)

import torch
import torchaudio
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

# Import updated training components
from utils.train_utils import preprocess_batch
from utils.data_utils import LibriSpeechCSVDataset, unified_collate_fn
from utils.inference_utils import load_boundary_classifier


def _load_model_class(variant: str = "rvq"):
    # RVQ is the only supported quantizer. The continuous-representation stage
    # is realized inside the RVQ model via stop-grad passthrough (alpha anneal),
    # so no separate model class is needed.
    from models.model_tokenizer import Data2VecSemanticAcousticModel as _M
    return _M

# CosyVoice frontend & model
sys.path.insert(0, os.environ.get(
    "COSYVOICE_ROOT", os.path.join(_STREAMASR_ROOT, "third_party", "CosyVoice")
))
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.cli.model import CosyVoiceModel
from cosyvoice.utils.common import fade_in_out  # noqa: F401 (imported for completeness)


from transformers import AutoTokenizer
from nemo_text_processing.text_normalization.normalize import Normalizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.generate_chunk_textgrids import load_word_asr_model


def _split_audio(wav: torch.Tensor, sr: int, max_sec: int = 30):
    """Return a list of chunks, each ≤ max_sec seconds long."""
    max_samples = max_sec * sr
    return [
        wav[..., i : i + max_samples]
        for i in range(0, wav.shape[-1], max_samples)
    ]


def _get_output_relative_path(src_path: str, input_dir: str, input_split: str, output_split: str) -> Path:
    """Map an input wav path to its output relative path with an optional split rename."""
    rel = Path(os.path.relpath(src_path, input_dir))
    parts = list(rel.parts)
    if parts and parts[0] == input_split:
        parts[0] = output_split
        return Path(*parts)
    return rel


################################################################################
# 2) Alignment utilities -------------------------------------------------------
################################################################################


@torch.no_grad()
def infer_batch(model, batch, tokenizer, word_asr_model, boundary_classifier, device):
    """Streaming inference: process each sample via forward_streaming."""
    file_paths = batch["file_paths"]
    spk_emb = batch["spk_emb"]
    waveforms = batch["waveforms"]
    B = len(file_paths)

    units_list = []
    for i in range(B):
        sample_spk_emb = spk_emb[i:i+1]  # (1, spk_dim)
        verbose = False
        u_logits = model.forward_streaming(
            wav_path=file_paths[i],
            spk_emb=sample_spk_emb,
            tokenizer=tokenizer,
            word_asr_model=word_asr_model,
            verbose=verbose,
            boundary_classifier=boundary_classifier,
        )  # (1, T', vocab_size)
        units = u_logits.argmax(dim=1)

        units_list.append(units[i, :].cpu())


    return units_list


################################################################################
# 3) CosyVoice helpers ---------------------------------------------------------
################################################################################

def token2wav(
    model,
    token,
    prompt_token,
    prompt_feat,
    embedding,
    token_hop_len: int = 25,
    token_max_hop_len: int = 100,
    stream_scale_factor: int = 2,
    streamability: bool = True,
):
    """Unit-to-speech decoding for CosyVoice3.

    When ``streamability`` is True, tokens are consumed in causal chunks of
    ``token_hop_len`` (with ``flow.pre_lookahead_len`` lookahead). Each chunk
    runs ``flow.inference`` with ``streaming=True`` and only the newly produced
    mel slice is fed to ``hift.inference``; the running mel cache and decoded-
    speech offset are kept across iterations, mirroring
    ``CosyVoice3Model.token2wav``.

    When ``streamability`` is False, the original non-streaming path is used:
    ``flow.inference`` is called once with ``streaming=False, finalize=True``
    and the full mel is decoded by ``hift.inference`` in a single shot.
    """
    if not streamability:
        mel, _ = model.flow.inference(
            token=token.to(model.device),
            token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(model.device),
            prompt_token=prompt_token.to(model.device),
            prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(
                model.device
            ),
            prompt_feat=prompt_feat.to(model.device),
            prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(
                model.device
            ),
            embedding=embedding.to(model.device),
            streaming=False,
            finalize=True,
        )
        wav, _ = model.hift.inference(speech_feat=mel, finalize=True)
        return wav.squeeze(0).cpu()

    device = model.device
    flow = model.flow
    hift = model.hift
    token_mel_ratio = flow.token_mel_ratio
    pre_lookahead_len = flow.pre_lookahead_len

    total_len = token.shape[1]
    p_len = prompt_token.shape[1]
    prompt_token_pad = (p_len + token_hop_len - 1) // token_hop_len * token_hop_len - p_len

    prompt_token = prompt_token.to(device)
    prompt_feat = prompt_feat.to(device)
    embedding = embedding.to(device)
    prompt_token_len = torch.tensor(
        [prompt_token.shape[1]], dtype=torch.int32, device=device
    )
    prompt_feat_len = torch.tensor(
        [prompt_feat.shape[1]], dtype=torch.int32, device=device
    )

    speech_chunks = []
    hift_cache = None  # {'mel': accumulated mel, 'speech_offset': samples already emitted}
    token_offset = 0

    while True:
        # mirror CosyVoice3Model.tts(): first hop absorbs prompt alignment pad
        this_token_hop_len = token_hop_len + prompt_token_pad if token_offset == 0 else token_hop_len
        finalize = total_len - token_offset < this_token_hop_len + pre_lookahead_len

        this_token = (
            token if finalize
            else token[:, :token_offset + this_token_hop_len + pre_lookahead_len]
        ).to(device, dtype=torch.int32)

        tts_mel, _ = flow.inference(
            token=this_token,
            token_len=torch.tensor(
                [this_token.shape[1]], dtype=torch.int32, device=device
            ),
            prompt_token=prompt_token,
            prompt_token_len=prompt_token_len,
            prompt_feat=prompt_feat,
            prompt_feat_len=prompt_feat_len,
            embedding=embedding,
            streaming=True,
            finalize=finalize,
        )
        tts_mel = tts_mel[:, :, token_offset * token_mel_ratio:]

        if hift_cache is not None:
            tts_mel = torch.concat([hift_cache["mel"], tts_mel], dim=2)
            hift_cache["mel"] = tts_mel
        else:
            hift_cache = {"mel": tts_mel, "speech_offset": 0}

        tts_speech, _ = hift.inference(speech_feat=tts_mel, finalize=finalize)
        tts_speech = tts_speech[:, hift_cache["speech_offset"]:]
        hift_cache["speech_offset"] += tts_speech.shape[1]
        speech_chunks.append(tts_speech.cpu())

        if finalize:
            break

        token_offset += this_token_hop_len
        token_hop_len = min(token_max_hop_len, token_hop_len * stream_scale_factor)

    wav = torch.cat(speech_chunks, dim=1)
    return wav.squeeze(0)


def get_speaker_reference(
    frontend,
    base_dir: str,
    split: str,
    speaker_id: str,
    exclude_path: str,
    *,
    max_sec: int = 30,
    max_tok: int = 1500,
):
    """
    Pick a random utterance of the speaker (excluding *exclude_path*).
    If it is longer than `max_sec` seconds **or** would produce more
    than `max_tok` speech tokens, split it into ≤ `max_sec`-second
    chunks.  Tokens & feats are concatenated across chunks; the speaker
    embedding is extracted from the **first** chunk only.
    """
    # e.g. exclude_path = .../121/127105/121-127105-0000.wav
    #      → number_dir = "127105"
    number_dir = Path(exclude_path).parent.name
    files = list(Path(base_dir).glob(f"{split}/{speaker_id}/{number_dir}/*.wav"))
    files = [f for f in files if str(f) != str(exclude_path)]

    # if not files:  # no reference available, use itself.
    #     ref = exclude_path
    # else:
    #     ref = random.choice(files)
    ref = exclude_path

    # --------------------------------------------------------------
    # 1. pick a reference file, load mono @16 kHz
    # --------------------------------------------------------------
    
    wav, sr = torchaudio.load(str(ref))
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000

    # --------------------------------------------------------------
    # 2. decide whether to split
    # --------------------------------------------------------------
    need_split = wav.shape[-1] > max_sec * sr  # length-based guard

    if not need_split:
        # quick token-count guard (cheap because we only extract once)
        tok_tmp, _ = frontend._extract_speech_token(wav)
        need_split = tok_tmp.numel() > max_tok

    # --------------------------------------------------------------
    # 3a. short enough → original behaviour
    # --------------------------------------------------------------
    if not need_split:
        speech_tok, _ = frontend._extract_speech_token(wav)
        speech_feat, _ = frontend._extract_speech_feat(str(ref))
        emb = frontend._extract_spk_embedding(wav)
        return speech_tok, speech_feat, emb

    # --------------------------------------------------------------
    # 3b. too long → chunk, then concat along time dim
    # --------------------------------------------------------------
    import tempfile
    chunks = _split_audio(wav, sr, max_sec=max_sec)

    tok_list, feat_list = [], []
    for chunk in chunks:
        t, _ = frontend._extract_speech_token(chunk)
        tok_list.append(t)

        # CosyVoice3 _extract_speech_feat expects a file path
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            torchaudio.save(tmp.name, chunk, sr)
            f, _ = frontend._extract_speech_feat(tmp.name)
        feat_list.append(f)

    # concatenate along time dimension (dim=1)
    speech_tok = torch.cat(tok_list, dim=1)
    speech_feat = torch.cat(feat_list, dim=1)

    # speaker embedding only on the first chunk
    emb = frontend._extract_spk_embedding(chunks[0])

    return speech_tok, speech_feat, emb


################################################################################
# 4) Main (per-rank) -----------------------------------------------------------
################################################################################

def run(rank: int, args):
    """Worker function: executed once per GPU."""

    # ---------------------------------------------------------------------
    # 4A. Distributed setup
    # ---------------------------------------------------------------------
    world_size = args.world_size
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print(f"Using {world_size} GPU(s). Each rank processes ~1/{world_size} of the data.")

    # ---------------------------------------------------------------------
    # 4B. Gather dataset & loader (sharded)
    # ---------------------------------------------------------------------

    dataset = LibriSpeechCSVDataset(args.test_csv, args.input_dir)
    #dataset = torch.utils.data.Subset(dataset, list(range(10)))  # simple sharding

    # Resume: drop samples whose output flac already exists, so DataLoader workers
    # don't waste time reading audio for utterances we'd skip anyway.
    if args.resume:
        before = len(dataset.samples)
        out_root = Path(args.output_dir)
        kept = []
        for sample in dataset.samples:
            src = sample["wav"].replace("{data_root}", args.input_dir)
            outp = out_root / _get_output_relative_path(
                src, args.input_dir, args.split, args.output_split
            )
            if not outp.exists():
                kept.append(sample)
        dataset.samples = kept
        if rank == 0:
            print(f"[resume] kept {len(kept)} / {before} utterances "
                  f"(skipped {before - len(kept)} already-done)")

    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if world_size > 1
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False if sampler else True,
        sampler=sampler,
        num_workers=4,
        collate_fn=unified_collate_fn,
        pin_memory=True,
    )

    # ---------------------------------------------------------------------
    # 4C. Load CosyVoice configs & frontend
    # ---------------------------------------------------------------------


    model_dir = os.environ.get(
        "COSYVOICE_MODEL_DIR",
        os.path.join(
            os.environ.get("COSYVOICE_ROOT", os.path.join(_STREAMASR_ROOT, "third_party", "CosyVoice")),
            "pretrained_models", "Fun-CosyVoice3-0.5B",
        ),
    )
    
    from hyperpyyaml import load_hyperpyyaml
    with open('{}/cosyvoice3.yaml'.format(model_dir), 'r') as f:
        configs = load_hyperpyyaml(
            f,
            overrides={
                "qwen_pretrain_path": os.path.join(model_dir, "CosyVoice-BlankEN")
            },
        )
    frontend = CosyVoiceFrontEnd(
        get_tokenizer=configs["get_tokenizer"],
        feat_extractor=configs["feat_extractor"],
        campplus_model=f"{model_dir}/campplus.onnx",
        speech_tokenizer_model=f"{model_dir}/speech_tokenizer_v3.onnx",
        allowed_special=configs["allowed_special"],
    )

    if args.tokenizer == "qwen3":
        glob_tok = AutoTokenizer.from_pretrained(
            os.environ.get("QWEN_TOKENIZER_PATH", "Qwen/Qwen3-8B")
        )
    else:  # default: llama
        glob_tok = AutoTokenizer.from_pretrained(os.environ.get("LLAMA_TOKENIZER_PATH", "meta-llama/Llama-3.1-8B"))
    glob_tok.pad_token = glob_tok.eos_token

    # ---------------------------------------------------------------------
    # 4D. Load student model
    # ---------------------------------------------------------------------
    _Model = _load_model_class(args.variant)
    student = _Model(
        chunk_size=args.chunk_size,
        left_context=args.left_context,
        hparams_file=args.hparams,
        checkpoint_path=args.truthmodel_checkpoint_path,
    )

    
    # Load checkpoint - handle both old and new checkpoint formats
    checkpoint = torch.load(args.studentmodel_checkpoint_path, map_location="cpu")
    
    if "hubert_state_dict" in checkpoint:
        # New format from training script
        student_state = checkpoint["hubert_state_dict"]
    elif "model" in checkpoint:
        # Alternative format
        student_state = checkpoint["model"]
    else:
        # Direct state dict
        student_state = checkpoint
    
    # Remove 'module.' prefix if present (from DistributedDataParallel)
    if any(k.startswith('module.') for k in student_state.keys()):
        student_state = {k.replace('module.', ''): v for k, v in student_state.items()}
    
    # Load state dict with strict=False to handle potential mismatches
    missing_keys, unexpected_keys = student.load_state_dict(student_state, strict=False)
    
    if rank == 0:
        if missing_keys:
            print(f"Warning: Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
        print("Student model loaded successfully")
    
    student.to(device).eval()

    # ---------------------------------------------------------------------
    # 4E. Load word ASR model
    # ---------------------------------------------------------------------
    word_asr_model = load_word_asr_model(
        hparams_file=args.word_hparams,
        checkpoint=args.word_checkpoint,
        tokenizer_ckpt=getattr(args, "word_tokenizer_ckpt", None),
        device=device,
    )

    # ---------------------------------------------------------------------
    # 4F. Load CosyVoice flow + hift (per GPU)
    # ---------------------------------------------------------------------
    cosy = CosyVoiceModel(None, configs["flow"], configs["hift"])
    cosy.flow.load_state_dict(torch.load(f"{model_dir}/flow.pt", map_location=device))
    cosy.hift.load_state_dict(torch.load(f"{model_dir}/hift.pt", map_location=device))
    cosy.flow.to(device).eval()
    cosy.hift.to(device).eval()
    cosy.flow_hift_context = (
        torch.cuda.stream(torch.cuda.Stream(device)) if torch.cuda.is_available() else nullcontext()
    )

    # ---------------------------------------------------------------------
    # 4G. Load boundary classifier (optional)
    # ---------------------------------------------------------------------
    boundary_classifier = None
    if args.boundary_classifier_ckpt:
        print(f"Loading BoundaryClassifier from: {args.boundary_classifier_ckpt}")
        boundary_classifier = load_boundary_classifier(args.boundary_classifier_ckpt, device)
        print("  BoundaryClassifier loaded — boundary model will decide word commits.\n")

    # ---------------------------------------------------------------------
    # 4H. Reconstruction loop
    # ---------------------------------------------------------------------
    loader_desc = f"Rank {rank}"
    for batch in tqdm(loader, desc=loader_desc, disable=rank != 0):
        try:
            if sampler is not None:
                sampler.set_epoch(0)  # ensures deterministic order (single epoch)

            # Skip batch if all output files already exist
            file_paths = batch["file_paths"]
            all_done = all(
                (
                    Path(args.output_dir)
                    / _get_output_relative_path(
                        src, args.input_dir, args.split, args.output_split
                    )
                ).exists()
                for src in file_paths
            )
            if all_done:
                continue

            # Use the updated preprocess_batch function
            pb = preprocess_batch(batch, frontend, glob_tok, device)
            units_list = infer_batch(student, pb, glob_tok, word_asr_model, boundary_classifier, device)

            for units, src in zip(units_list, pb["file_paths"]):
                # Skip if output already exists
                rel = _get_output_relative_path(
                    src, args.input_dir, args.split, args.output_split
                )
                outp = Path(args.output_dir) / rel
                if outp.exists():
                    continue

                # Skip if no units were generated
                if units.numel() == 0:
                    if rank == 0:
                        print(f"⚠️ No units generated for {src}, skipping...")
                    continue
                    
                speaker_id = Path(src).parent.parent.name
                _, _, emb_ref = get_speaker_reference(
                    frontend, args.input_dir, args.split, speaker_id, src
                )

                with cosy.flow_hift_context:
                    wav = token2wav(
                        cosy,
                        units.unsqueeze(0),
                        torch.zeros(1, 0, dtype=torch.int32),
                        torch.zeros(1, 0, 80),
                        emb_ref,
                        streamability=args.streamability,
                    )

                outp.parent.mkdir(parents=True, exist_ok=True)
                torchaudio.save(str(outp), wav.unsqueeze(0), 24000)
        except Exception as e:
            import traceback
            print("Skip: ", e)
            traceback.print_exc()

    if dist.is_initialized():
        dist.barrier()
        if rank == 0:
            print("[All ranks] Resynthesis complete. Output dir:", args.output_dir)
        dist.destroy_process_group()
    else:
        print("Resynthesis complete. Output dir:", args.output_dir)


################################################################################
# 5) Entry-point ---------------------------------------------------------------
################################################################################


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="/home/Streaming")
    parser.add_argument("--output_dir", default="/home/Streaming/dataset")
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--left_context", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--hparams", type=str,
                        default="hparams/alignment.yaml")
    parser.add_argument("--truthmodel_checkpoint_path", type=str,
                        default="results/char_asr_ckpt",
                        help="SpeechBrain ASR checkpoint for truth_model (StreamingCharModel)")
    parser.add_argument("--studentmodel_checkpoint_path", type=str,
                        default="checkpoints/streamalign_r16/epoch_22.pt",
                        help="Hubert model checkpoint")
    parser.add_argument("--test_csv", type=str,
                    default="results/conformer_transducer_char/alignment/test-clean.csv",
                    help="Validation CSV file(s)")
    parser.add_argument("--split", type=str, default="test-clean",
                    help="LibriSpeech split name (e.g. test-clean, dev-clean)")
    parser.add_argument("--output_split", type=str, default=None,
                    help="Top-level output subdirectory name. Defaults to --split.")
    parser.add_argument("--word_hparams", type=str,
                        required=True,
                        help="HParams YAML for the word-level StreamingASR model")
    parser.add_argument("--word_checkpoint", type=str,
                        required=True,
                        help="Checkpoint dir/file for the word-level StreamingASR model")
    parser.add_argument("--word_tokenizer_ckpt", type=str, default=None,
                        help="Optional SentencePiece model path for the word ASR tokenizer")
    parser.add_argument(
        "--world_size",
        type=int,
        default=torch.cuda.device_count(),
        help="Number of GPUs to use (defaults to all visible).",
    )
    parser.add_argument("--boundary_classifier_ckpt", type=str, default=None,
                   help="Path to a trained BoundaryClassifier checkpoint ")
    parser.add_argument("--tokenizer", type=str, default="llama",
                        choices=["llama", "qwen3"],
                        help="Text tokenizer to use: 'llama' (meta-llama/Llama-3.1-8B) or 'qwen3' (Qwen/Qwen3-8B-Instruct)")
    parser.add_argument("--variant", type=str, default="rvq",
                        choices=["rvq"],
                        help="Student model variant. Only 'rvq' (ResidualVQ) is supported; "
                             "the continuous stage is realized inside the RVQ model via "
                             "stop-grad passthrough.")
    parser.add_argument("--streamability", type=lambda v: str(v).lower() in ("1", "true", "yes", "y", "t"),
                        default=True,
                        help="If true (default), use streaming token2wav (chunked flow/hift). "
                             "If false, use the original non-streaming path "
                             "(single-shot flow.inference + hift.inference).")
    parser.add_argument("--resume", type=lambda v: str(v).lower() in ("1", "true", "yes", "y", "t"),
                        default=True,
                        help="If true (default), skip utterances whose output already exists. "
                             "Filters dataset upfront so no audio is loaded for done utterances.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.output_split is None:
        args.output_split = args.split

    if args.world_size > 1:
        # Launch one process per GPU via torch.multiprocessing.spawn or torchrun.
        # If user invoked the script via torchrun, LOCAL_RANK is already set.
        local_rank_env = os.environ.get("LOCAL_RANK")
        if local_rank_env is not None:
            # Running under torchrun: execute worker directly.
            run(int(local_rank_env), args)
        else:
            # Spawn processes (stand-alone python execution)
            torch.multiprocessing.spawn(run, nprocs=args.world_size, args=(args,))
    else:
        # Single-GPU fall-back
        run(0, args)
