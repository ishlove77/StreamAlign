import os
import random, math, torchaudio
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchaudio.functional as F
from pathlib import Path
import json

import os
import glob

from datasets import Dataset as HFDataset, concatenate_datasets
PAD_ID = -100


def _resolve_data_path(path_value: str, data_folder: str) -> str:
    """Resolve CSV wav paths that may use data_root placeholders."""
    resolved = path_value.strip()
    for token in ("$data_root", "${data_root}", "{data_root}"):
        resolved = resolved.replace(token, data_folder)

    if os.path.isabs(resolved):
        resolved = os.path.normpath(resolved)
        if os.path.exists(resolved):
            return resolved

        legacy_roots = os.environ.get(
            "LEGACY_SPEECH_DATA_ROOTS",
            "/data/LibriSpeech:/data/LibriTTS",
        ).split(":")
        for legacy_root in legacy_roots:
            legacy_root = os.path.normpath(legacy_root)
            if resolved == legacy_root or resolved.startswith(legacy_root + os.sep):
                relative = os.path.relpath(resolved, legacy_root)
                return os.path.normpath(os.path.join(data_folder, relative))
        return resolved

    return os.path.normpath(os.path.join(data_folder, resolved))


class TASTEArrowDataset(Dataset):
    """
    PyTorch Dataset for loading audio-text data from HuggingFace Arrow shards.
    We flatten away the nested `json` struct (which had mismatched field orders
    across shards) and only keep `json["text"]` as a top-level `text` column.
    """
    def __init__(self, arrow_paths: str):
        super().__init__()

        # 2) load + flatten each shard
        datasets = []
        for path in arrow_paths:
            ds = HFDataset.from_file(path)
            # extract just the transcript and drop the struct
            ds = ds.map(
                lambda ex: {"text": ex["json"]["text"]},
                remove_columns=["json"],
                load_from_cache_file=False
            )
            datasets.append(ds)

        # 3) now safe to concatenate
        self.dataset = (
            concatenate_datasets(datasets)
            if len(datasets) > 1
            else datasets[0]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        mp3 = row["mp3"]
        array = mp3["array"]
        sr    = mp3["sampling_rate"]
        tokens = row["s3_token"]
        emb    = row["spk_emb"]

        waveform = torch.from_numpy(array)
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0)

        return {
            "waveform":      waveform,
            "sampling_rate": sr,
            "text":          row["text"],
            "s3_token":      tokens,
            "spk_emb":       emb,
        }
    
def taste_collate_fn(batch):
    """
    Collate function for TASTEArrowDataset.
    Pads and batches:
      - waveforms (resampled to 16 kHz)
      - s3_token sequences
      - speaker embeddings (spk_emb)
    Keeps raw texts in a list.
    """
    # Lists to collect per-sample data
    wavs, wav_lens = [], []
    texts = []
    token_seqs, token_lens = [], []
    spk_embs = []

    # First pass: extract and resample / convert
    for item in batch:
        # --- audio ---
        wav = item["waveform"]
        sr  = item["sampling_rate"]
        if sr != 16_000:
            wav = F.resample(wav, sr, 16_000)
        wavs.append(wav)
        wav_lens.append(wav.size(0))

        # --- transcript ---
        texts.append(item["text"])

        # --- token sequence ---
        tokens = torch.tensor(item["s3_token"], dtype=torch.long)
        token_seqs.append(tokens)
        token_lens.append(tokens.size(0))

        # --- speaker embedding ---
        emb = torch.tensor(item["spk_emb"], dtype=torch.float)
        spk_embs.append(emb)

    batch_size = len(wavs)

    # Pad waveforms
    max_wav_len = max(wav_lens)
    waveforms = torch.zeros(batch_size, max_wav_len)
    wav_mask   = torch.zeros(batch_size, max_wav_len, dtype=torch.bool)
    for i, wav in enumerate(wavs):
        L = wav.size(0)
        waveforms[i, :L] = wav
        wav_mask[i, :L] = True

    # Pad token sequences
    max_tok_len = max(token_lens)
    token_tensor = torch.full((batch_size, max_tok_len), PAD_ID, dtype=torch.long)
    token_mask   = torch.zeros(batch_size, max_tok_len, dtype=torch.bool)
    for i, toks in enumerate(token_seqs):
        L = toks.size(0)
        token_tensor[i, :L] = toks
        token_mask[i, :L] = True

    # Stack speaker embeddings into (B, D)
    spk_emb = torch.stack(spk_embs, dim=0)

    return {
        "waveforms":       waveforms,                       # (B, Tmax)
        "wav_lens":        torch.tensor(wav_lens),          # (B,)
        "wav_mask":        wav_mask,                        # (B, Tmax)
        "raw_text":           texts,                           # list[str]
        "s3_token":        token_tensor,                    # (B, Tmax_tokens)
        "s3_token_lens":   torch.tensor(token_lens),        # (B,)
        "s3_token_mask":   token_mask,                      # (B, Tmax_tokens)
        "spk_emb":         spk_emb,                         # (B, D)
    }

###############################################################################
# LibriSpeech CSV Dataset
###############################################################################
import csv
class LibriSpeechCSVDataset(Dataset):
    """
    Dataset that reads LibriSpeech data from CSV file.
    CSV format: ID,duration,wav,spk_id,wrd
    """
    def __init__(self, csv_path, data_folder):
        super().__init__()
        self.data_folder = data_folder
        self.samples = []

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append(row)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        while True:
            try:
                sample = self.samples[idx]
                wav_path = _resolve_data_path(sample["wav"], self.data_folder)

                waveform, sr = torchaudio.load(wav_path)
                duration_sec = waveform.shape[-1] / sr

                if waveform.size(0) > 1:
                    waveform = waveform.mean(0)
                else:
                    waveform = waveform.squeeze(0)

                text = sample['wrd']

                return {
                    "source": "librispeech",
                    "waveform": waveform,
                    "sampling_rate": sr,
                    "wav_path": wav_path,
                    "text": text,
                }
            except Exception as e:
                print(f"Skipping file at index {idx} due to error: {e}")

            idx = random.randint(0, len(self.samples) - 1)

def _librispeech_root() -> str:
    root = os.environ.get("LIBRISPEECH_ROOT")
    if root:
        return root
    legacy = os.environ.get("LIBRI_ROOT")
    if legacy:
        print("[data_utils_cosyvoice] LIBRI_ROOT is deprecated; "
              "use LIBRISPEECH_ROOT instead.")
        return legacy
    return "/data/LibriSpeech"


_LIBRI_ROOT = _librispeech_root()
# Explicit alias for consumers that mean "the LibriSpeech flac root".
_LIBRISPEECH_ROOT = _LIBRI_ROOT
# LibriTTS root (wav layout), for consumers building LibriTTS datasets.
_LIBRITTS_ROOT = os.environ.get("LIBRITTS_ROOT", "/data/LibriTTS")
# TextGrid root defaults to a sibling of the audio root so the two cannot
# silently diverge; override with TEXTGRID_ROOT.
_TEXTGRID_ROOT = os.environ.get(
    "TEXTGRID_ROOT",
    os.path.join(_LIBRI_ROOT, "chunk_textgrids_word_model_final2"),
)

# Cache root for precomputed CosyVoice3 features (mirrors LibriTTS layout).
_FEATURE_CACHE_ROOT = os.environ.get(
    "COSYVOICE_FEATURE_CACHE_ROOT",
    os.path.join(
        os.environ.get(
            "STREAMASR_ROOT",
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ),
        "cache",
        "cosyvoice_features",
    ),
)


def _libri_feature_paths(wav_path: str):
    """(speech_tokens.pt, spk_emb.pt) under _FEATURE_CACHE_ROOT for a LibriTTS wav.

    The cache mirrors the layout written by
    scripts/precompute/precompute_speech_tokens.py, which relpaths against
    its --data_root (the LibriTTS root) — so resolve against _LIBRITTS_ROOT.
    """
    rel = os.path.relpath(wav_path, _LIBRITTS_ROOT)
    if rel.endswith(".flac"):
        rel = rel[:-5] + ".wav"
    base = os.path.join(_FEATURE_CACHE_ROOT, rel)
    return base + ".speech_tokens.pt", base + ".spk_emb.pt"


# TextGrid root for LibriTTS training wavs (generate_textgrids.sh default
# output). A run reading LibriSpeech instead sets TEXTGRID_ROOT explicitly.
_LIBRITTS_TEXTGRID_ROOT = os.environ.get(
    "TEXTGRID_ROOT",
    os.path.join(_LIBRITTS_ROOT, "chunk_textgrids_word_model_final2"),
)


def _get_textgrid_path(wav_path: str) -> str:
    """Return the corresponding chunk TextGrid path for a LibriTTS wav."""
    rel = os.path.relpath(wav_path, _LIBRITTS_ROOT)      # e.g. train-clean-100/103/1241/....wav
    stem = rel.rsplit(".", 1)[0]                          # strip .wav
    return os.path.join(_LIBRITTS_TEXTGRID_ROOT, stem + ".TextGrid")


class LibriTTSDataset(Dataset):
    """Dataset that reads .wav files and their chunk-level TextGrid alignments.

    TextGrids are loaded from ``chunk_textgrids_word_model`` (encoder-frame units).
    No .normalized.txt is required; the text label is derived from the TextGrid.

    When ``use_precomputed_features=True``, also loads cached CosyVoice3
    speech tokens (``<wav>.speech_tokens.pt``) and speaker embeddings
    (``<wav>.spk_emb.pt``) and returns them in the item dict; samples missing
    either cache file are skipped (the index is reshuffled).
    """

    def __init__(self, wavpaths, use_precomputed_features=False):
        super().__init__()
        self.wavpaths = wavpaths
        self.use_precomputed_features = use_precomputed_features

    def __len__(self):
        return len(self.wavpaths)

    def __getitem__(self, idx):
        cache_misses = 0
        while True:
            try:
                wav_path = self.wavpaths[idx]

                speech_tokens, spk_emb = None, None
                if self.use_precomputed_features:
                    tok_path, spk_path = _libri_feature_paths(wav_path)
                    if not (os.path.exists(tok_path) and os.path.exists(spk_path)):
                        cache_misses += 1
                        if cache_misses >= 1000:
                            raise RuntimeError(
                                f"{cache_misses} consecutive precomputed-feature cache "
                                f"misses (last: {tok_path}). The CosyVoice feature cache "
                                "is missing — run scripts/precompute/ first, or check "
                                "COSYVOICE_FEATURE_CACHE_ROOT / LIBRITTS_ROOT."
                            )
                        idx = random.randint(0, len(self.wavpaths) - 1)
                        continue
                    speech_tokens = torch.load(tok_path, map_location="cpu").long()
                    spk_emb = torch.load(spk_path, map_location="cpu").float()
                    if spk_emb.dim() == 2 and spk_emb.size(0) == 1:
                        spk_emb = spk_emb.squeeze(0)

                waveform, sr = torchaudio.load(wav_path)
                duration_sec = waveform.shape[-1] / sr

                if waveform.size(0) > 1:
                    waveform = waveform.mean(0)
                else:
                    waveform = waveform.squeeze(0)

                if duration_sec > 30:
                    idx = random.randint(0, len(self.wavpaths) - 1)
                    continue

                # Load TextGrid from chunk_textgrids_word_model (may not exist)
                tg_path = _get_textgrid_path(wav_path)
                textgrid_intervals = None
                if os.path.exists(tg_path):
                    from utils.rnnt_align_loss import parse_textgrid_words
                    textgrid_intervals = parse_textgrid_words(tg_path)

                item = {
                    "source": "libri",
                    "waveform": waveform,
                    "sampling_rate": sr,
                    "wav_path": wav_path,
                    "textgrid_intervals": textgrid_intervals,  # list[(t_s, t_e, text)] or None
                }
                if speech_tokens is not None:
                    item["speech_tokens"] = speech_tokens
                    item["spk_emb"] = spk_emb
                return item
            except RuntimeError:
                raise  # fail loudly on an all-miss feature cache
            except Exception as e:
                print(f"Skipping file at index {idx} due to error: {e}")

            idx = random.randint(0, len(self.wavpaths) - 1)

class LibriSpeechFlacDataset(Dataset):
    """LibriSpeech .flac dataset with chunk-level TextGrid alignments.

    Mirrors LibriTTSDataset, but reads the LibriSpeech flac layout and
    resolves TextGrids against ``_TEXTGRID_ROOT`` relative to the
    LibriSpeech root (``LIBRISPEECH_ROOT``).
    """

    def __init__(self, wavpaths):
        super().__init__()
        self.wavpaths = wavpaths

    def __len__(self):
        return len(self.wavpaths)

    def __getitem__(self, idx):
        while True:
            try:
                wav_path = self.wavpaths[idx]
                waveform, sr = torchaudio.load(wav_path)
                duration_sec = waveform.shape[-1] / sr
                if waveform.size(0) > 1:
                    waveform = waveform.mean(0)
                else:
                    waveform = waveform.squeeze(0)
                if duration_sec > 30:
                    idx = random.randint(0, len(self.wavpaths) - 1)
                    continue

                rel = os.path.relpath(wav_path, _LIBRISPEECH_ROOT)
                tg_path = os.path.join(
                    _TEXTGRID_ROOT, rel.rsplit(".", 1)[0] + ".TextGrid"
                )
                textgrid_intervals = None
                if os.path.exists(tg_path):
                    from utils.rnnt_align_loss import parse_textgrid_words
                    textgrid_intervals = parse_textgrid_words(tg_path)

                return {
                    "source": "libri",
                    "waveform": waveform,
                    "sampling_rate": sr,
                    "wav_path": wav_path,
                    "textgrid_intervals": textgrid_intervals,
                }
            except Exception as e:
                print(f"Skipping file at index {idx} due to error: {e}")

            idx = random.randint(0, len(self.wavpaths) - 1)


class EmiliaDataset(Dataset):
    def __init__(self, wavpaths):
        super().__init__()
        self.wavpaths = wavpaths
        self.base_path = os.environ.get("EMILIA_ROOT", "/data/Emilia")

    def __len__(self):
        return len(self.wavpaths)

    def __getitem__(self, idx):
        while True:
            try:
                # Load audio
                wav_path = self.wavpaths[idx]
                waveform, sr = torchaudio.load(wav_path)
                duration_sec = waveform.shape[-1] / sr

                if waveform.size(0) > 1:            # stereo → mix down to mono
                    waveform = waveform.mean(0)     # or pick one channel
                else:
                    waveform = waveform.squeeze(0)

                # Skip if too long
                if duration_sec > 30:
                    idx = random.randint(0, len(self.wavpaths) - 1)
                    continue

                # Load corresponding JSON
                json_path = wav_path.replace('.mp3', '.json')
                with open(json_path, 'r') as f:
                    json_data = json.load(f)
                text = json_data["text"]

                return {
                    "source": "emilia",
                    "waveform": waveform,
                    "sampling_rate": sr,
                    "wav_path": wav_path,
                    "text": text,
                }

            except Exception as e:
                print(f"⚠️ Skipping due to error: {e}")
                idx = random.randint(0, len(self.wavpaths) - 1)


# ── Emilia (TextGrid-based) ──────────────────────────────────────────────────

_EMILIA_ROOT = os.environ.get("EMILIA_ROOT", "/data/Emilia")
_EMILIA_TEXTGRID_ROOT = os.environ.get(
    "EMILIA_TEXTGRID_ROOT",
    os.path.join(_EMILIA_ROOT, "chunk_textgrids_word_model_final2"),
)
_EMILIA_FEATURE_CACHE_ROOT = os.environ.get(
    "COSYVOICE_FEATURE_CACHE_ROOT_EMILIA",
    os.path.join(_FEATURE_CACHE_ROOT, "_emilia"),
)


def _emilia_feature_paths(wav_path: str):
    rel = os.path.relpath(wav_path, _EMILIA_ROOT)
    base = os.path.join(_EMILIA_FEATURE_CACHE_ROOT, rel)
    return base + ".speech_tokens.pt", base + ".spk_emb.pt"


def _get_emilia_textgrid_path(wav_path: str) -> str:
    """Return the corresponding chunk_textgrids_word_model3 TextGrid path for an Emilia file."""
    rel = os.path.relpath(wav_path, _EMILIA_ROOT)        # e.g. EN-B000062/EN_B00006_S08732_W000109.mp3
    stem = rel.rsplit(".", 1)[0]                        # strip .mp3
    return os.path.join(_EMILIA_TEXTGRID_ROOT, stem + ".TextGrid")


class EmiliaTextGridDataset(Dataset):
    """Emilia dataset that loads audio + chunk-level TextGrid alignments.

    Same interface / return format as LibriTTSDataset so that
    ``unified_collate_fn`` works without modification.
    """

    def __init__(self, wavpaths, use_precomputed_features=False):
        super().__init__()
        self.wavpaths = wavpaths
        self.use_precomputed_features = use_precomputed_features

    def __len__(self):
        return len(self.wavpaths)

    def __getitem__(self, idx):
        cache_misses = 0
        while True:
            try:
                wav_path = self.wavpaths[idx]

                speech_tokens, spk_emb = None, None
                if self.use_precomputed_features:
                    tok_path, spk_path = _emilia_feature_paths(wav_path)
                    if not (os.path.exists(tok_path) and os.path.exists(spk_path)):
                        cache_misses += 1
                        if cache_misses >= 1000:
                            raise RuntimeError(
                                f"{cache_misses} consecutive precomputed-feature cache "
                                f"misses (last: {tok_path}). The Emilia feature cache is "
                                "missing — run scripts/precompute/ first, or check "
                                "EMILIA_ROOT / the emilia cache root."
                            )
                        idx = random.randint(0, len(self.wavpaths) - 1)
                        continue
                    speech_tokens = torch.load(tok_path, map_location="cpu").long()
                    spk_emb = torch.load(spk_path, map_location="cpu").float()
                    if spk_emb.dim() == 2 and spk_emb.size(0) == 1:
                        spk_emb = spk_emb.squeeze(0)

                waveform, sr = torchaudio.load(wav_path)
                duration_sec = waveform.shape[-1] / sr

                if waveform.size(0) > 1:
                    waveform = waveform.mean(0)
                else:
                    waveform = waveform.squeeze(0)

                if duration_sec > 30:
                    idx = random.randint(0, len(self.wavpaths) - 1)
                    continue

                tg_path = _get_emilia_textgrid_path(wav_path)
                textgrid_intervals = None
                if os.path.exists(tg_path):
                    from utils.rnnt_align_loss import parse_textgrid_words
                    textgrid_intervals = parse_textgrid_words(tg_path)

                item = {
                    "source": "emilia",
                    "waveform": waveform,
                    "sampling_rate": sr,
                    "wav_path": wav_path,
                    "textgrid_intervals": textgrid_intervals,
                }
                if speech_tokens is not None:
                    item["speech_tokens"] = speech_tokens
                    item["spk_emb"] = spk_emb
                return item
            except RuntimeError:
                raise  # fail loudly on an all-miss feature cache
            except Exception as e:
                print(f"Skipping Emilia file at index {idx} due to error: {e}")
                idx = random.randint(0, len(self.wavpaths) - 1)


def load_emilia_wavpaths(csv_path: str, data_root: str) -> list:
    """Read an Emilia CSV manifest and return resolved audio paths."""
    wavpaths = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wavpaths.append(_resolve_data_path(row["wav"], data_root))
    return wavpaths


class NoiseAugmentor:
    """
    Implements the paper’s two noise strategies:
      • 75 % env. noise from a directory with *.wav / *.flac
      • 25 % other-speech from the same mini-batch
    It applies *per-batch* and touches only a random 20 % of samples.
    """
    def __init__(self, noise_root: str, sr: int = 16_000):
        self.sr = sr
        self.noise_files = list(Path(noise_root).rglob("*.wav")) + \
                           list(Path(noise_root).rglob("*.flac"))
        assert self.noise_files, f"No noise wavs found in {noise_root}"

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _zscore(self, wav: torch.Tensor):
        w = wav - wav.mean()
        std = w.std()
        return w / std.clamp_min(1e-6)

    def _rand_env_noise(self, L: int, device):
        """5-s clip from an env-noise wav. Handles resample + crop."""
        wav_path = random.choice(self.noise_files)
        wav, sr = torchaudio.load(wav_path)
        if sr != self.sr:
            wav = F.resample(wav, sr, self.sr)
        wav = wav[0]
        if wav.numel() < 5 * self.sr:
            # loop-pad if file is shorter than 5 s
            wav = wav.repeat(math.ceil((5*self.sr)/wav.numel()) + 1)
        start = random.randint(0, wav.numel() - 5*self.sr)
        clip  = wav[start : start + 5*self.sr]
        # repeat / pad to target length L
        if clip.numel() < L:
            clip = clip.repeat(math.ceil(L / clip.numel()) + 1)
        return clip[:L].to(device)

    def _rand_other_speech(self, batch_wav, src_idx):
        """Pick *another* sample in the batch and shift it 40-70 %."""
        B, L = batch_wav.size()
        choices = [i for i in range(B) if i != src_idx]
        j = random.choice(choices)
        wav = batch_wav[j]
        shift = int(L * random.uniform(0.4, 0.7))
        # positive = right shift, negative = left shift
        if random.random() < 0.5:
            pad = torch.zeros(shift, device=wav.device)
            wav = torch.cat([pad, wav[:-shift]])
        else:
            pad = torch.zeros(shift, device=wav.device)
            wav = torch.cat([wav[shift:], pad])
        return wav

    # ------------------------------------------------------------------
    # main entry
    # ------------------------------------------------------------------
    def __call__(self, batch_wav: torch.Tensor):
        """
        batch_wav : (B, L) float32 in range [-1,1]  (already loaded by you)
        Returns a *copy* of batch_wav with noise mixed into 20 % of rows.
        """
        B, L = batch_wav.size()
        device = batch_wav.device
        out = batch_wav.clone()

        # indices to corrupt 20 %
        keep_prob = 0.20
        mask = torch.rand(B, device=batch_wav.device) < keep_prob      # (B,) bool
        if not mask.any():                                             # rare when B < 5
            mask[random.randrange(B)] = True                           # guarantee ≥1

        noisy_idx = mask.nonzero(as_tuple=False).squeeze(1).tolist()

        for i in noisy_idx:
            # decide noise source
            if random.random() < 0.75:     # 75 % env noise
                noise = self._rand_env_noise(L, device)
                scale = random.uniform(0.05, 0.7)
            else:                          # 25 % other speech
                noise = self._rand_other_speech(batch_wav, i)
                scale = random.uniform(0.0, 0.2)

            clean = out[i]
            clean_z  = self._zscore(clean)
            noise_z  = self._zscore(noise)
            out[i]   = clean_z + scale * noise_z

        return out

def unified_collate_fn(batch):
    """Collate LibriTTS samples.

    Pads waveforms and passes TextGrid interval lists through as-is
    (they are variable-length and cannot be padded into a tensor).

    When precomputed CosyVoice3 features are present in items
    (``"speech_tokens"`` and ``"spk_emb"``), they are padded/stacked into
    ``"speech_units"`` (B, T_tok_max) int64 with PAD_ID=-100 and
    ``"spk_emb"`` (B, 192) float32 so ``preprocess_batch`` can pass them
    through unchanged.
    """
    wavs, wav_lens = [], []
    for it in batch:
        wav = it["waveform"]
        if it["sampling_rate"] != 16_000:
            wav = F.resample(wav, it["sampling_rate"], 16_000)
        wavs.append(wav)
        wav_lens.append(wav.size(0))

    max_L = max(wav_lens)
    padded_wav = torch.zeros(len(wavs), max_L)
    wav_mask   = torch.zeros(len(wavs), max_L, dtype=torch.bool)
    for i, w in enumerate(wavs):
        padded_wav[i, : w.size(0)] = w
        wav_mask[i, : w.size(0)] = True

    out = {
        "waveforms":          padded_wav,
        "wav_lens":           torch.tensor(wav_lens),
        "wav_mask":           wav_mask,
        "file_paths":         [b["wav_path"] for b in batch],
        # list of (list[(t_s, t_e, text)] or None), one entry per sample
        "textgrid_intervals": [b.get("textgrid_intervals") for b in batch],
    }

    if all("speech_tokens" in b for b in batch):
        max_T = max(b["speech_tokens"].size(0) for b in batch)
        units_pad = torch.full((len(batch), max_T), -100, dtype=torch.long)
        for i, b in enumerate(batch):
            t = b["speech_tokens"]
            units_pad[i, : t.size(0)] = t
        out["speech_units"] = units_pad
        out["spk_emb"] = torch.stack([b["spk_emb"] for b in batch], dim=0)

    return out
