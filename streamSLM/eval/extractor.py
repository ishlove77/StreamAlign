"""On-the-fly streaming-aligned unit extraction for evaluation.

Wraps the StreamAlign teacher to extract per-subword
(subword_ids, q_codes, duration_frames) from a (waveform, TextGrid) pair.

Why TextGrids and not Whisper transcripts? Training-time extraction
(``streamSLM.extract.extract_tokens``) consumes chunk TextGrids produced
by the streaming Conformer-Transducer (``streamASR/data/generate_chunk_
textgrids.py``). Each interval's text is what that streaming ASR
committed in a 4-frame chunk given strictly-causal left context — i.e.
the same distribution of words/letters per chunk that the SLM saw during
training. Substituting a Whisper transcript instead (oracle whole-utterance
text) leaks future context into the constrained RNN-T alignment, which
breaks the streaming-distribution match. The eval pipeline must use the
same streaming ASR + chunk-wise alignment that built the training data.
"""

from __future__ import annotations

import os
import sys
from contextlib import nullcontext as _nullcontext
from typing import List, Optional

# Silence wandb before any streamASR import touches alignment.yaml's logger.
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")

import torch

# Resolve sibling streamASR / CosyVoice imports the same way
# streamSLM.extract.extract_tokens does.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_STREAMASR_ROOT = os.path.join(_REPO_ROOT, "streamASR")
_COSYVOICE_ROOT = os.environ.get(
    "COSYVOICE_ROOT", os.path.join(_STREAMASR_ROOT, "third_party", "CosyVoice")
)
for _p in [
    os.path.join(_STREAMASR_ROOT, "third_party", "Matcha-TTS"),
    os.path.join(_COSYVOICE_ROOT, "third_party", "Matcha-TTS"),
    _COSYVOICE_ROOT,
    _STREAMASR_ROOT,
    _REPO_ROOT,
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from transformers import AutoTokenizer  # noqa: E402

from utils.train_utils import preprocess_batch  # noqa: E402
from utils.rnnt_align_loss import parse_textgrid_words  # noqa: E402

from streamSLM.units import SubwordUnits  # noqa: E402


# --------------------------------------------------------------------------- #
# Defaults — match streamSLM.inference.synthesize / extract_tokens.py
# --------------------------------------------------------------------------- #
DEFAULT_HPARAMS = os.environ.get(
    "ASR_HPARAMS", os.path.join(_STREAMASR_ROOT, "hparams", "alignment.yaml")
)
DEFAULT_TRUTHMODEL = os.environ.get(
    "TRUTH_MODEL_CKPT", os.path.join(_STREAMASR_ROOT, "results", "char_asr_ckpt")
)
# Released R=16 tokenizer (Streamalign-R16, rvq_teacher/epoch_22.pt).
DEFAULT_TEACHER = os.environ.get(
    "TEACHER_CKPT",
    os.path.join(_REPO_ROOT, "weights", "Streamalign-R16", "rvq_teacher", "epoch_22.pt"),
)
DEFAULT_RVQ_NUM_QUANTIZERS = 8
DEFAULT_RVQ_CODEBOOK_SIZE = 512


def _load_model_class(variant: str = "rvq"):
    """RVQ is the only supported quantizer."""
    if variant != "rvq":
        raise ValueError(f"unsupported variant: {variant} (only 'rvq')")
    from models.model_tokenizer import Data2VecSemanticAcousticModel as M
    return M


class RVQUnitExtractor:
    """Audio + transcript -> SubwordUnits via the StreamAlign RVQ teacher."""

    def __init__(
        self,
        teacher_checkpoint: str = DEFAULT_TEACHER,
        variant: str = "rvq",
        hparams: str = DEFAULT_HPARAMS,
        truthmodel_checkpoint: str = DEFAULT_TRUTHMODEL,
        chunk_size: int = 16,
        left_context: int = 8,
                        rvq_num_quantizers: int = DEFAULT_RVQ_NUM_QUANTIZERS,
        rvq_codebook_size: int = DEFAULT_RVQ_CODEBOOK_SIZE,
        text_tokenizer: str = "llama",
        device: Optional[torch.device] = None,
        bf16: bool = True,
    ):
        # Quantizer knobs are read by the teacher's RVQ module on
        # construction — set both families regardless of variant so the
        # CosyVoice teacher constructor picks up whichever it inspects.
        os.environ["RVQ_R"] = str(rvq_num_quantizers)
        os.environ["RVQ_CODEBOOK_SIZE"] = str(rvq_codebook_size)

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.bf16 = bf16

        ModelCls = _load_model_class(variant)
        self.model = ModelCls(
            chunk_size=chunk_size,
            left_context=left_context,
            hparams_file=hparams,
            checkpoint_path=truthmodel_checkpoint,
        )
        ckpt = torch.load(teacher_checkpoint, map_location="cpu", weights_only=False)
        if "hubert_state_dict" in ckpt:
            sd = ckpt["hubert_state_dict"]
        elif "model" in ckpt:
            sd = ckpt["model"]
        else:
            sd = ckpt
        if any(k.startswith("module.") for k in sd.keys()):
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(
                f"[extractor] missing={len(missing)} unexpected={len(unexpected)} "
                f"(proceeding)",
                flush=True,
            )
        self.model.to(self.device).eval()

        # The "global" tokenizer used inside extract_subword_units to build
        # word_token_ids. Must match the streamSLM training-time choice
        # (Llama-3.1-8B is what extract_tokens.py uses for the llama branch
        # and gives the same vocab as Llama-3.2-1B).
        if text_tokenizer == "qwen3":
            self.glob_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
        else:
            self.glob_tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
        self.glob_tok.pad_token = self.glob_tok.eos_token

    @staticmethod
    def _make_batch(
        wavs: List[torch.Tensor],
        intervals_per_sample: List[Optional[list]],
        sample_rate: int = 16000,
        device: Optional[torch.device] = None,
    ) -> dict:
        """Build the dict consumed by ``preprocess_batch``.

        ``intervals_per_sample[i]`` is a list of ``(frame_start, frame_end,
        text)`` tuples (encoder-frame indices, 25 Hz). Pass ``None`` to skip
        the sample at the alignment stage. This is the same format
        ``parse_textgrid_words(tg_path)`` returns and what
        ``streamSLM.extract.extract_tokens`` feeds the teacher.
        """
        B = len(wavs)
        wav_lens = [w.numel() for w in wavs]
        max_len = max(wav_lens) if wav_lens else 0
        wav_pad = torch.zeros(B, max_len, dtype=torch.float32)
        wav_mask = torch.zeros(B, max_len, dtype=torch.bool)
        for i, w in enumerate(wavs):
            wav_pad[i, : wav_lens[i]] = w
            wav_mask[i, : wav_lens[i]] = True

        return {
            "waveforms":          wav_pad,
            "wav_lens":           torch.tensor(wav_lens, dtype=torch.long),
            "wav_mask":           wav_mask,
            "textgrid_intervals": list(intervals_per_sample),
            "file_paths":         [f"_eval_{i}" for i in range(B)],
        }

    @torch.inference_mode()
    def extract_batch(
        self,
        wavs: List[torch.Tensor],
        intervals_per_sample: List[Optional[list]],
        sample_rate: int = 16000,
    ) -> List[SubwordUnits]:
        """Run the teacher on a batch of (wav, TG-intervals) pairs.

        ``wavs[i]`` is a 1-D float tensor at ``sample_rate`` Hz.
        ``intervals_per_sample[i]`` is the TG word tier produced by the
        streaming Conformer-Transducer (one interval per encoder chunk).
        Returns one ``SubwordUnits`` per input; samples whose alignment
        fails (e.g. all-empty TG) yield zero-length units — the caller
        should skip / drop those.
        """
        batch = self._make_batch(
            wavs, intervals_per_sample, sample_rate=sample_rate, device=self.device,
        )
        preproc = preprocess_batch(
            batch, frontend=None, glob_tok=self.glob_tok, device=self.device,
            txt_normalizer=None, extract_only=True,
        )

        ctx = (
            torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.bf16 and self.device.type == "cuda"
            else _nullcontext()
        )
        with ctx:
            units_per_sample = self.model.extract_subword_units(
                waveforms=preproc["waveforms"],
                wav_mask=preproc["wav_mask"],
                tokenizer=self.glob_tok,
                txt_normalizer=None,
                char_ids_list=preproc["char_ids_list"],
                constraints_list=preproc["constraints_list"],
                spk_emb=preproc["spk_emb"],  # None -> auto zero-fill in extract_subword_units
            )

        out = []
        for u in units_per_sample:
            pqf = u.get("pre_quant_feat", None)
            out.append(SubwordUnits(
                subword_ids=u["subword_ids"].to(torch.long).cpu(),
                q_codes=u["q_codes"].to(torch.long).cpu(),
                duration_frames=u["duration_frames"].to(torch.long).cpu(),
                pre_quant_feat=(pqf.to(torch.float32).cpu() if pqf is not None else None),
            ))
        return out

    def extract_one(
        self,
        wav: torch.Tensor,
        tg_path: Optional[str] = None,
        intervals: Optional[list] = None,
        sample_rate: int = 16000,
    ) -> SubwordUnits:
        """Extract units for one (wav, TG) pair.

        Exactly one of ``tg_path`` or ``intervals`` must be provided. When
        ``tg_path`` is given, the TextGrid's words tier is parsed via
        ``parse_textgrid_words`` (same loader as training-time extraction).
        """
        if (tg_path is None) == (intervals is None):
            raise ValueError("extract_one expects exactly one of tg_path / intervals")
        if tg_path is not None:
            intervals = parse_textgrid_words(tg_path)
        return self.extract_batch([wav], [intervals], sample_rate=sample_rate)[0]
