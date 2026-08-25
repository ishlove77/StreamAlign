#!/usr/bin/env python3
"""End-to-end StreamSLM inference: text or speech prompt -> waveform.

Combines the two existing pieces into a single CLI:
    streamSLM.inference.generate.generate()      AR sampling of (w, q, d)
    streamSLM.inference.reconstruct.*            units -> wav via StreamAlign + CosyVoice

Two prompt modes (mutually exclusive; both optional):

    --text_prompt "Once upon a time"
        Pure text-prompted generation. The text is tokenized with the LLM
        tokenizer, BOS-prefixed by the tokenizer's add_special_tokens path,
        and used as the AR seed. Acoustic / duration prompt slots are
        zero-filled (as in generate.py) — fine for short prompts because
        the model's `delay` shift makes the first `delay` slots use the
        learned pad embedding regardless.

    --prompt_units path.units.pt   [+ optional --prompt_max_subwords N]
        Speech continuation. Loads (subword_ids, q_codes, duration_frames)
        produced by extract_tokens.py and uses them verbatim as the AR
        prefix. Generation continues from the end of the prefix.

If neither flag is given, generation starts from a single BOS token.

Usage:
    sr 1 24 python -m streamSLM.inference.synthesize \
        --slm_checkpoint checkpoints/streamSLM/.../step_00050000.pt \
        --streamalign_ckpt weights/Streamalign-R16/rvq_teacher/epoch_22.pt \
        --speaker_wav prompts/jane.wav \
        --text_prompt "Once upon a time" \
        --max_new_tokens 96 \
        --out_wav out/sample.wav

    # Speech continuation:
    sr 1 24 python -m streamSLM.inference.synthesize \
        --slm_checkpoint .../step_00050000.pt \
        --streamalign_ckpt .../rvq_teacher/epoch_22.pt \
        --speaker_wav prompts/jane.wav \
        --prompt_units cache/streamSLM_units_C2048/librispeech/.../foo.flac.units.pt \
        --prompt_max_subwords 8 \
        --max_new_tokens 64 \
        --out_wav out/continuation.wav
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Match reconstruct.py: silence wandb before any streamASR / cosyvoice import.
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")

import torch
import torchaudio

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_STREAMASR_ROOT = os.path.join(_REPO_ROOT, "streamASR")
_COSYVOICE_ROOT = os.environ.get(
    "COSYVOICE_ROOT", os.path.join(_STREAMASR_ROOT, "third_party", "CosyVoice")
)
for p in [
    os.path.join(_STREAMASR_ROOT, "third_party", "Matcha-TTS"),
    os.path.join(_COSYVOICE_ROOT, "third_party", "Matcha-TTS"),
    _COSYVOICE_ROOT,
    _STREAMASR_ROOT,
    _REPO_ROOT,
]:
    if p not in sys.path:
        sys.path.insert(0, p)

from streamSLM.inference.generate import generate, _load_streamslm  # noqa: E402
from streamSLM.inference.reconstruct import (  # noqa: E402
    _load_streamalign, _load_cosy,
    units_to_speech_tokens, prepare_speaker, speech_tokens_to_wav,
)
from streamSLM.model.slm import load_lm_tokenizer  # noqa: E402
from streamSLM.units import SubwordUnits  # noqa: E402


def _build_prompt(args, lm_tok):
    """Return (prompt_ids, prompt_q, prompt_d). Any of q/d may be None."""
    if args.prompt_units:
        prefix = SubwordUnits.load(args.prompt_units)
        if args.prompt_max_subwords > 0:
            keep = min(args.prompt_max_subwords, prefix.num_subwords)
            prefix = SubwordUnits(
                subword_ids=prefix.subword_ids[:keep],
                q_codes=prefix.q_codes[:keep],
                duration_frames=prefix.duration_frames[:keep],
            )
        print(f"[prompt] units prefix: N={prefix.num_subwords} "
              f"R={prefix.num_quantizers} frames={int(prefix.duration_frames.sum())}")
        return prefix.subword_ids.long(), prefix.q_codes.long(), prefix.duration_frames.long()
    if args.text_prompt:
        ids = lm_tok(args.text_prompt, return_tensors="pt",
                     add_special_tokens=True).input_ids[0]
        print(f"[prompt] text: {args.text_prompt!r} -> N={ids.numel()} subwords")
        return ids, None, None
    bos = lm_tok.bos_token_id if lm_tok.bos_token_id is not None else lm_tok.eos_token_id
    print(f"[prompt] BOS-only (id={bos})")
    return torch.tensor([bos], dtype=torch.long), None, None


def main():
    ap = argparse.ArgumentParser(
        description="StreamSLM end-to-end inference (text/units prompt -> wav)."
    )
    # ---- StreamSLM (the language model that produces units) ----
    ap.add_argument("--slm_checkpoint", required=True,
                    help="StreamSLM .pt (saved by streamSLM.train.train).")
    # ---- StreamAlign student (units -> CosyVoice speech tokens) ----
    ap.add_argument("--streamalign_ckpt", required=True,
                    help="StreamAlign student .pt (the encoder used during extraction).")
    ap.add_argument("--variant", choices=["rvq"], default="rvq")
    ap.add_argument("--hparams", default=
                    os.environ.get("ASR_HPARAMS", os.path.join(_STREAMASR_ROOT, "hparams", "alignment.yaml")))
    ap.add_argument("--truthmodel_checkpoint_path", default=
                    os.environ.get("TRUTH_MODEL_CKPT", os.path.join(_STREAMASR_ROOT, "results", "char_asr_ckpt")))
    ap.add_argument("--chunk_size", type=int, default=16)
    ap.add_argument("--left_context", type=int, default=8)
    # ---- CosyVoice (flow + hift -> wav) ----
    ap.add_argument("--cosyvoice_model_dir", default=
                    os.path.join(_COSYVOICE_ROOT, "pretrained_models", "Fun-CosyVoice3-0.5B"))
    ap.add_argument("--speaker_wav", required=True,
                    help="Reference wav for speaker embedding + flow prompt.")
    # ---- Prompting ----
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--text_prompt", default="",
                   help="Text-only generation prompt. Empty -> BOS only.")
    g.add_argument("--prompt_units", default="",
                   help="Speech-continuation prefix .units.pt (overrides --text_prompt).")
    ap.add_argument("--prompt_max_subwords", type=int, default=0,
                    help="If --prompt_units is set, keep only the first N subwords (0 = all).")
    # ---- Sampling ----
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature_text", type=float, default=0.9)
    ap.add_argument("--top_p_text", type=float, default=0.95)
    ap.add_argument("--top_k_text", type=int, default=0)
    ap.add_argument("--temperature_aco", type=float, default=1.0)
    ap.add_argument("--top_p_aco", type=float, default=0.9)
    ap.add_argument("--top_k_aco", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    # ---- Output ----
    ap.add_argument("--out_wav", required=True, help="Output waveform path (.wav).")
    ap.add_argument("--save_units", default="",
                    help="Optional path to also dump the generated SubwordUnits .pt.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device} variant={args.variant}", flush=True)

    # ---- Load StreamSLM and AR-generate units ----
    print(f"[slm ] loading {args.slm_checkpoint}", flush=True)
    slm, tok_cfg, _ = _load_streamslm(args.slm_checkpoint, device)
    lm_tok = load_lm_tokenizer(tok_cfg)
    prompt_ids, prompt_q, prompt_d = _build_prompt(args, lm_tok)

    print(f"[slm ] generating up to {args.max_new_tokens} new subwords"
          f" (T={args.temperature_text}/{args.temperature_aco}, "
          f"top_p={args.top_p_text}/{args.top_p_aco})", flush=True)
    units = generate(
        slm,
        prompt_ids=prompt_ids,
        prompt_q=prompt_q,
        prompt_d=prompt_d,
        max_new_tokens=args.max_new_tokens,
        eos_id=lm_tok.eos_token_id,
        temperature_text=args.temperature_text,
        top_k_text=args.top_k_text,
        top_p_text=args.top_p_text,
        temperature_aco=args.temperature_aco,
        top_k_aco=args.top_k_aco,
        top_p_aco=args.top_p_aco,
        device=device,
    )
    text = lm_tok.decode(units.subword_ids.tolist())
    print(f"[slm ] N={units.num_subwords} frames={int(units.duration_frames.sum())} "
          f"R={units.num_quantizers}", flush=True)
    print(f"[slm ] decoded text: {text!r}", flush=True)

    if args.save_units:
        Path(args.save_units).parent.mkdir(parents=True, exist_ok=True)
        units.save(args.save_units)
        print(f"[slm ] saved units -> {args.save_units}")

    # SLM is no longer needed; free its VRAM before loading the StreamAlign +
    # CosyVoice stack (StreamAlign + flow + hift can be ~2-3 GB combined).
    del slm
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- Reconstruct units -> wav ----
    print(f"[recon] loading StreamAlign {args.streamalign_ckpt}", flush=True)
    sa = _load_streamalign(
        args.variant, args.streamalign_ckpt,
        args.hparams, args.truthmodel_checkpoint_path,
        args.chunk_size, args.left_context, device,
    )
    print(f"[recon] loading CosyVoice from {args.cosyvoice_model_dir}", flush=True)
    frontend, cosy = _load_cosy(args.cosyvoice_model_dir, device)

    spk = prepare_speaker(frontend, args.speaker_wav)
    tokens = units_to_speech_tokens(
        sa,
        subword_ids=units.subword_ids,
        q_codes=units.q_codes,
        durations=units.duration_frames,
        spk_emb=spk["flow_embedding"],
        lm_tokenizer=lm_tok,
    )
    print(f"[recon] speech tokens: {tuple(tokens.shape)}", flush=True)

    wav = speech_tokens_to_wav(cosy, tokens, spk, device)
    out_wav = Path(args.out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_wav), wav.unsqueeze(0), 24000)
    print(f"[save ] {out_wav}  shape={tuple(wav.shape)} sr=24000", flush=True)

    # Persist the SLM-generated text + per-subword IDs next to the wav so the
    # caller can audit what was decoded without rerunning the model.
    out_txt = out_wav.with_suffix(".txt")
    out_txt.write_text(
        f"text: {text}\n"
        f"subword_ids: {units.subword_ids.tolist()}\n"
        f"duration_frames: {units.duration_frames.tolist()}\n"
        f"n_subwords: {units.num_subwords}\n"
        f"total_frames: {int(units.duration_frames.sum())}\n"
    )
    print(f"[save ] {out_txt}", flush=True)


if __name__ == "__main__":
    main()
