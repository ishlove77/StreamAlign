"""Helpers for decoding CosyVoice unit sequences and sampling test-clean audio."""

import importlib.util
import json
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torchaudio
from torch.utils.data import DataLoader, Subset

# Support direct execution via `python .../utils/cosyvoice_unit_utils.py`.
STREAM_ASR_ROOT = Path(__file__).resolve().parents[1]
if str(STREAM_ASR_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM_ASR_ROOT))

from utils.data_utils_cosyvoice import LibriSpeechCSVDataset, unified_collate_fn
from utils.text_utils import PAD_ID
from utils.train_utils_cosyvoice import preprocess_batch

DEFAULT_COSYVOICE_MODEL_DIR = (
    "/home/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"
)
DEFAULT_LIBRISPEECH_ROOT = "/home/datasets/LibriSpeech"
DEFAULT_LIBRISPEECH_TEST_CSV = (
    "/home/datasets/LibriSpeech/csv/test-clean.csv"
)
DEFAULT_UNIT_SAMPLE_RATE = 24000

_COSYVOICE_DECODER_CACHE: Dict[tuple[str, str], Any] = {}
_INFERENCE_COSYVOICE_MODULE = None


def trim_padded_units(units: torch.Tensor) -> torch.Tensor:
    """Remove PAD_ID values from one speech-unit sequence."""
    units = units.detach()
    if units.dim() != 1:
        units = units.reshape(-1)
    units = units[units != PAD_ID]
    return units.long().cpu()


def infer_librispeech_style_location(
    source_path: str,
    input_dir: Optional[str] = None,
    split: Optional[str] = None,
) -> tuple[str, str, str]:
    """Infer dataset root, split, and speaker id from a LibriSpeech-style path."""
    src = Path(source_path).resolve()

    if input_dir is not None:
        base_dir = str(Path(input_dir).resolve())
    else:
        try:
            base_dir = str(src.parents[3])
        except IndexError as exc:
            raise ValueError(f"Could not infer dataset root from path: {source_path}") from exc

    if split is None:
        try:
            split = src.parents[2].name
        except IndexError as exc:
            raise ValueError(f"Could not infer split from path: {source_path}") from exc

    try:
        speaker_id = src.parents[1].name
    except IndexError as exc:
        raise ValueError(f"Could not infer speaker id from path: {source_path}") from exc

    return base_dir, split, speaker_id


def load_cosyvoice_waveform_decoder(
    device: torch.device,
    model_dir: str = DEFAULT_COSYVOICE_MODEL_DIR,
):
    """Lazy-load the CosyVoice flow/HiFT decoder used by token2wav()."""
    cache_key = (str(Path(model_dir).resolve()), str(device))
    if cache_key in _COSYVOICE_DECODER_CACHE:
        return _COSYVOICE_DECODER_CACHE[cache_key]

    from hyperpyyaml import load_hyperpyyaml

    cosyvoice_root = "/home/CosyVoice"
    if cosyvoice_root not in sys.path:
        sys.path.insert(0, cosyvoice_root)

    from cosyvoice.cli.model import CosyVoiceModel

    with open(f"{model_dir}/cosyvoice3.yaml", "r") as f:
        configs = load_hyperpyyaml(f)

    cosy = CosyVoiceModel(None, configs["flow"], configs["hift"])
    cosy.flow.load_state_dict(torch.load(f"{model_dir}/flow.pt", map_location=device))
    cosy.hift.load_state_dict(torch.load(f"{model_dir}/hift.pt", map_location=device))
    cosy.flow.to(device).eval()
    cosy.hift.to(device).eval()
    cosy.flow_hift_context = (
        torch.cuda.stream(torch.cuda.Stream(device))
        if torch.cuda.is_available() and device.type == "cuda"
        else nullcontext()
    )

    _COSYVOICE_DECODER_CACHE[cache_key] = cosy
    return cosy


def _load_inference_core_module():
    """Load inference_core.py by file path to avoid package-name collisions."""
    global _INFERENCE_COSYVOICE_MODULE
    if _INFERENCE_COSYVOICE_MODULE is not None:
        return _INFERENCE_COSYVOICE_MODULE

    module_path = (
        Path(__file__).resolve().parents[1] / "inference" / "inference_core.py"
    )
    spec = importlib.util.spec_from_file_location(
        "streamasr_inference_core",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _INFERENCE_COSYVOICE_MODULE = module
    return module


@torch.no_grad()
def decode_cosyvoice_units_to_waveform(
    units: torch.Tensor,
    frontend,
    device: torch.device,
    source_path: str,
    cosy=None,
    *,
    input_dir: Optional[str] = None,
    split: Optional[str] = None,
    model_dir: str = DEFAULT_COSYVOICE_MODEL_DIR,
) -> torch.Tensor:
    """Decode one padded unit sequence back to waveform using token2wav()."""
    inference_core = _load_inference_core_module()

    trimmed_units = trim_padded_units(units)
    if trimmed_units.numel() == 0:
        return torch.empty(0)

    cosy = cosy or load_cosyvoice_waveform_decoder(device, model_dir=model_dir)
    base_dir, split_name, speaker_id = infer_librispeech_style_location(
        source_path,
        input_dir=input_dir,
        split=split,
    )
    tok_ref, feat_ref, emb_ref = inference_core.get_speaker_reference(
        frontend,
        base_dir,
        split_name,
        speaker_id,
        source_path,
    )

    with cosy.flow_hift_context:
        return inference_core.token2wav(
            cosy,
            trimmed_units.unsqueeze(0),
            tok_ref,
            feat_ref,
            emb_ref,
        )


@torch.no_grad()
def decode_cosyvoice_units_batch(
    units_batch: torch.Tensor,
    file_paths: List[str],
    frontend,
    device: torch.device,
    cosy=None,
    *,
    input_dir: Optional[str] = None,
    split: Optional[str] = None,
    model_dir: str = DEFAULT_COSYVOICE_MODEL_DIR,
) -> List[torch.Tensor]:
    """Decode a batch of padded speech-unit sequences to waveforms."""
    cosy = cosy or load_cosyvoice_waveform_decoder(device, model_dir=model_dir)
    waveforms = []
    for units, source_path in zip(units_batch, file_paths):
        wav = decode_cosyvoice_units_to_waveform(
            units,
            frontend,
            device,
            source_path,
            cosy=cosy,
            input_dir=input_dir,
            split=split,
            model_dir=model_dir,
        )
        waveforms.append(wav.cpu())
    return waveforms


@torch.no_grad()
def sample_librispeech_test_clean_unit_examples(
    frontend,
    glob_tok,
    device: torch.device,
    num_samples: int = 10,
    seed: int = 13,
    *,
    test_csv: str = DEFAULT_LIBRISPEECH_TEST_CSV,
    input_dir: str = DEFAULT_LIBRISPEECH_ROOT,
    model_dir: str = DEFAULT_COSYVOICE_MODEL_DIR,
    cosy=None,
) -> List[Dict[str, Any]]:
    """Sample about 10 LibriSpeech test-clean items and decode their unit sequences."""
    dataset = LibriSpeechCSVDataset(test_csv, input_dir)
    if len(dataset) == 0:
        return []

    n_samples = min(num_samples, len(dataset))
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), n_samples)
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=n_samples,
        shuffle=False,
        num_workers=0,
        collate_fn=unified_collate_fn,
        pin_memory=device.type == "cuda",
    )

    batch_data = next(iter(loader))
    preprocessed = preprocess_batch(batch_data, frontend, glob_tok, device)
    decoded_waveforms = decode_cosyvoice_units_batch(
        preprocessed["speech_units"],
        preprocessed["file_paths"],
        frontend,
        device,
        cosy=cosy,
        input_dir=input_dir,
        split="test-clean",
        model_dir=model_dir,
    )

    examples: List[Dict[str, Any]] = []
    for local_idx, dataset_idx in enumerate(indices):
        wav_len = int(preprocessed["wav_lens"][local_idx].item())
        sample_row = dataset.samples[dataset_idx]
        examples.append(
            {
                "dataset_index": dataset_idx,
                "file_path": preprocessed["file_paths"][local_idx],
                "text": sample_row.get("wrd", ""),
                "ground_truth_waveform": preprocessed["waveforms"][local_idx, :wav_len].detach().cpu(),
                "ground_truth_sample_rate": 16000,
                "speech_units": trim_padded_units(preprocessed["speech_units"][local_idx]),
                "unit_waveform": decoded_waveforms[local_idx],
                "unit_sample_rate": DEFAULT_UNIT_SAMPLE_RATE,
            }
        )

    return examples


@torch.no_grad()
def save_librispeech_test_clean_unit_examples(
    output_dir: str,
    frontend,
    glob_tok,
    device: torch.device,
    num_samples: int = 10,
    seed: int = 13,
    *,
    test_csv: str = DEFAULT_LIBRISPEECH_TEST_CSV,
    input_dir: str = DEFAULT_LIBRISPEECH_ROOT,
    model_dir: str = DEFAULT_COSYVOICE_MODEL_DIR,
    cosy=None,
) -> List[Dict[str, Any]]:
    """Save sampled test-clean originals and unit reconstructions to disk."""
    examples = sample_librispeech_test_clean_unit_examples(
        frontend=frontend,
        glob_tok=glob_tok,
        device=device,
        num_samples=num_samples,
        seed=seed,
        test_csv=test_csv,
        input_dir=input_dir,
        model_dir=model_dir,
        cosy=cosy,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for i, example in enumerate(examples):
        stem = f"{i:02d}_{Path(example['file_path']).stem}"
        source_path = out_dir / f"{stem}_source.wav"
        units_path = out_dir / f"{stem}_units.wav"
        unit_wav_out = ""

        torchaudio.save(
            str(source_path),
            example["ground_truth_waveform"].unsqueeze(0),
            example["ground_truth_sample_rate"],
        )

        if example["unit_waveform"].numel() > 0:
            torchaudio.save(
                str(units_path),
                example["unit_waveform"].unsqueeze(0),
                example["unit_sample_rate"],
            )
            unit_wav_out = str(units_path)

        manifest.append(
            {
                "index": i,
                "dataset_index": example["dataset_index"],
                "file_path": example["file_path"],
                "text": example["text"],
                "num_units": int(example["speech_units"].numel()),
                "source_wav": str(source_path),
                "unit_wav": unit_wav_out,
            }
        )

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return examples
