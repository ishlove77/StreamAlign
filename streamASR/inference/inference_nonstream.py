#!/usr/bin/env python3
"""Prod-style RVQ inference, NON-STREAMING.

Same as inference_stream.py (loads the RVQ model with the multi-GPU
.to() fix and reuses the prod run()/token2wav pipeline) EXCEPT it replaces the
chunk-by-chunk ``forward_streaming`` with a single whole-utterance
``forward()`` pass.

Alignment without TextGrids: the word RNN-T model transcribes the utterance
(via its streaming chunk API, text only), then the full transcript is used as
a whole-utterance constraint for ``model.forward`` -> constrained_rnnt_align
runs over the entire utterance at once (non-streaming).
"""
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_STREAMASR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _STREAMASR_ROOT not in sys.path:
    sys.path.insert(0, _STREAMASR_ROOT)

import torch
import torch.nn as nn

from models.model_tokenizer import Data2VecSemanticAcousticModel as _RVQModelBase


class _RVQModel(_RVQModelBase):
    """Multi-GPU device fix (walk encoder.hparams onto target device)."""
    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        device = None
        for a in args:
            if isinstance(a, (str, torch.device)):
                try:
                    device = torch.device(a); break
                except Exception:
                    pass
        if device is None:
            d = kwargs.get("device")
            if d is not None:
                device = torch.device(d)
        if device is None:
            return result
        enc = getattr(self, "encoder", None)
        if enc is not None:
            try:
                enc.device = device
            except Exception:
                pass
            hp = getattr(enc, "hparams", None)
            if isinstance(hp, dict):
                for v in hp.values():
                    if isinstance(v, nn.Module):
                        v.to(device)
                    elif torch.is_tensor(v):
                        try:
                            v.data = v.data.to(device)
                        except Exception:
                            pass
        return result


# Load the prod inference module (gives us run(), token2wav, get_speaker_reference,
# preprocess_batch usage, parse_args, etc.) and monkey-patch the model class.
_PROD_INFER = os.path.join(_STREAMASR_ROOT, "inference", "inference_core.py")
_spec = importlib.util.spec_from_file_location("inference_core_prod", _PROD_INFER)
_orig = importlib.util.module_from_spec(_spec)
sys.modules["inference_core_prod"] = _orig
_spec.loader.exec_module(_orig)

_orig.Data2VecSemanticAcousticModel = _RVQModel


# ----------------------------------------------------------------------------
# Non-streaming infer_batch (replaces prod's streaming infer_batch)
# ----------------------------------------------------------------------------
@torch.no_grad()
def _word_model_transcript(model, word_asr_model, wav_1d, device):
    """Stream the waveform through the word model (text only) to get the
    committed transcript, mirroring forward_streaming's text path."""
    chunk_size = model.encoder.get_chunk_size_frames(model.chunk_config)
    word_context = word_asr_model.make_streaming_context(model.chunk_config)
    final_chunk_count = (
        model.encoder.hparams["fea_streaming_extractor"]
        .get_recommended_final_chunk_count(chunk_size)
    )
    # split into chunk_size pieces
    chunks = list(torch.split(wav_1d, chunk_size))
    final_chunks = [torch.zeros(chunk_size, device=device)] * final_chunk_count
    text = ""
    for chunk in chunks + final_chunks:
        if chunk.numel() < chunk_size:
            pad = torch.zeros(chunk_size, device=device)
            pad[: chunk.numel()] = chunk
            chunk = pad
        out = word_asr_model.transcribe_chunk(word_context, chunk.unsqueeze(0).to(device))
        new_text = out[0] if out else ""
        text += new_text.replace("▁", " ")
    return text


@torch.no_grad()
def _infer_batch_nonstream(model, batch, tokenizer, word_asr_model,
                           boundary_classifier, device):
    file_paths = batch["file_paths"]
    spk_emb = batch["spk_emb"]
    waveforms = batch["waveforms"]
    wav_lens = batch["wav_lens"]
    wav_mask = batch["wav_mask"]
    B = len(file_paths)
    print(f"[nonstream] infer_batch ENTER B={B}", flush=True)

    units_list = []
    for i in range(B):
        L = int(wav_lens[i].item())
        wav_1d = waveforms[i, :L].to(device)
        # 1) transcript via word model (text only)
        text = _word_model_transcript(model, word_asr_model, wav_1d, device)
        char_ids = [model._CHAR_TO_IDX[ch] for ch in text.lower()
                    if ch in model._CHAR_TO_IDX]
        T_speech = int(model._compute_model_feat_len(L))
        _dbg = os.environ.get("NONSTREAM_DEBUG", "0") == "1"
        if _dbg:
            print(f"[nonstream] {os.path.basename(file_paths[i])} "
                  f"text='{text[:80]}' n_char={len(char_ids)} T_speech={T_speech}",
                  flush=True)
        if not char_ids:
            if _dbg: print("[nonstream]   -> empty char_ids, skip", flush=True)
            units_list.append(torch.empty(0, dtype=torch.long))
            continue
        # 2) whole-utterance constraint
        constraints = [(0, T_speech, 0, len(char_ids))]
        # 3) single non-streaming forward over the whole utterance.
        #    Optionally drop the chunked attention mask (full bidirectional
        #    attention) via NONSTREAM_FULL_ATTN=1 -- swap only around forward
        #    so the word-model transcript above still uses the chunk size.
        wav_in = waveforms[i:i + 1, :L].to(device)
        mask_in = wav_mask[i:i + 1, :L].to(device)
        _full_attn = os.environ.get("NONSTREAM_FULL_ATTN", "0") == "1"
        _saved_cc = model.chunk_config
        if _full_attn:
            model.chunk_config = None
        try:
            out = model.forward(
                wav_in, mask_in,
                spk_emb=spk_emb[i:i + 1].to(device),
                tokenizer=tokenizer,
                txt_normalizer=None,
                char_ids_list=[char_ids],
                constraints_list=[constraints],
            )
        finally:
            model.chunk_config = _saved_cc
        u_logits = out.get("u_logits", None)
        if _dbg:
            ca = out.get("char_alignment", None); wa = out.get("word_alignment", None)
            print(f"[nonstream]   keys={list(out.keys())} "
                  f"char_any={bool(ca.any()) if ca is not None else None} "
                  f"word_any={bool(wa.any()) if wa is not None else None} "
                  f"u_logits={tuple(u_logits.shape) if u_logits is not None else None}",
                  flush=True)
        if u_logits is None:
            units_list.append(torch.empty(0, dtype=torch.long))
            continue
        units = u_logits.argmax(dim=1)  # (1, T')
        units_list.append(units[0].cpu())
    return units_list


_orig.infer_batch = _infer_batch_nonstream
print("[nonstream] PATCH APPLIED: infer_batch ->", _orig.infer_batch.__name__, flush=True)


if __name__ == "__main__":
    args = _orig.parse_args()
    if args.output_split is None:
        args.output_split = args.split
    if args.world_size > 1:
        local_rank_env = os.environ.get("LOCAL_RANK")
        if local_rank_env is not None:
            _orig.run(int(local_rank_env), args)
        else:
            torch.multiprocessing.spawn(_orig.run, nprocs=args.world_size, args=(args,))
    else:
        _orig.run(0, args)
