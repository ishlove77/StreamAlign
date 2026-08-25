"""End-to-end speech-continuation pipeline in a single process.

Replaces the per-utt fan-out in `_infer_speech_cont_continuous.sh` (3 separate
`sr` calls per utt: generate -> reconstruct -> whisper). Each invocation of the
old script paid:
    * slurm scheduling per step (~10-20 s)
    * fresh python import + checkpoint load per step (SLM .pt is 14.9 GB)
    * three model-load waves per utt

This module loads every model exactly once and loops over utts, doing all three
steps inline. Same artifacts on disk: per-utt `sample.units.pt`, `sample.units.txt`,
`sample.wav`, plus a `summary.jsonl` with per-utt timings + ASR transcript.

Usage:
    python -m streamSLM.inference.speech_cont_pipeline \
        --checkpoint <slm.pt> \
        --teacher_checkpoint <streamalign.pt> \
        --utts_csv <utt_list.csv|"-"> \
        --units_cache <prequant_cache_root> \
        --speaker_root <librispeech_test_clean> \
        --out_dir <out_root> \
        --prompt_max_subwords 20 --max_new_tokens 20 --bf16

The `utts_csv` is a one-id-per-line file; each id is "<spkr>-<book>-<utt>".
Use "-" to read ids from stdin (one per line). The launcher passes the same
id list it currently builds inline from the manifest CSVs.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_STREAMASR_ROOT = os.path.join(_REPO_ROOT, "streamASR")
_COSYVOICE_ROOT = os.environ.get(
    "COSYVOICE_ROOT", os.path.join(_STREAMASR_ROOT, "third_party", "CosyVoice")
)

# Match extract_tokens.py / reconstruct.py: silence wandb before imports.
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")

import torch
import torchaudio

from streamSLM.units import SubwordUnits
from streamSLM.config import TokenizerConfig
from streamSLM.model.slm import load_lm_tokenizer
from streamSLM.inference.generate import (
    _load_streamslm,
    _load_teacher_residual_vq,
    _load_teacher_residual_vq,
    generate,
)
from streamSLM.inference.reconstruct import (
    _load_streamalign,
    _load_cosy,
    units_to_speech_tokens,
    prepare_speaker,
    speech_tokens_to_wav,
)


def _read_utts(arg: str) -> list[str]:
    if arg == "-":
        return [ln.strip() for ln in sys.stdin if ln.strip()]
    p = Path(arg)
    if p.is_file():
        return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    # Treat as whitespace-separated literal list.
    return [t for t in arg.split() if t.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--teacher_checkpoint", required=True)
    ap.add_argument("--utts_csv", required=True,
                    help="Path to a file with one utt-id per line, '-' for stdin, "
                         "or a whitespace-separated literal list.")
    ap.add_argument("--units_cache", required=True,
                    help="Root of the precomputed .units.pt cache "
                         "(layout: <root>/<spkr>/<book>/<utt>.flac.units.pt).")
    ap.add_argument("--speaker_root", required=True,
                    help="Root of LibriSpeech .flac files "
                         "(layout: <root>/<spkr>/<book>/<utt>.flac).")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prompt_max_subwords", type=int, default=20)
    ap.add_argument("--max_new_tokens", type=int, default=20)
    ap.add_argument("--temperature_text", type=float, default=0.9)
    ap.add_argument("--top_p_text", type=float, default=0.95)
    # Acoustic head is deterministic by default (temp=0 -> argmax in _sample).
    # The text head stays stochastic; pass --temperature_aco>0 to re-enable.
    ap.add_argument("--temperature_aco", type=float, default=0.0)
    ap.add_argument("--top_p_aco", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--encoder_hz", type=float, default=25.0)
    ap.add_argument("--allowed_text_token_ids", default="",
                    help="Optional path to a .pt tensor of allowed LM-tokenizer "
                         "ids (1-D long). When set, non-allowed ids are masked "
                         "to -inf before text sampling, restricting decoding "
                         "to vocab seen in the training cache.")
    ap.add_argument("--redo_final_aco", type=int, default=0,
                    help="Drop the last K acoustic positions of the speech "
                         "prompt and let the model re-predict them. Mitigates "
                         "trailing-silence/EOS bias from the prompt's final "
                         "acoustic frames. Typically set to model.delay.")

    # Reconstruct / streamalign knobs (mirrors reconstruct.py defaults).
    ap.add_argument("--variant", choices=["rvq"], default="rvq")
    ap.add_argument("--hparams", default=os.environ.get(
                    "ASR_HPARAMS", os.path.join(_STREAMASR_ROOT, "hparams", "alignment.yaml")))
    ap.add_argument("--truthmodel_checkpoint_path", default=os.environ.get(
                    "TRUTH_MODEL_CKPT", os.path.join(_STREAMASR_ROOT, "results", "char_asr_ckpt")))
    ap.add_argument("--chunk_size", type=int, default=16)
    ap.add_argument("--left_context", type=int, default=8)
    ap.add_argument("--cosyvoice_model_dir", default=os.path.join(
                    _COSYVOICE_ROOT, "pretrained_models", "Fun-CosyVoice3-0.5B"))

    # Whisper.
    ap.add_argument("--whisper_model", default="base.en")
    ap.add_argument("--skip_whisper", action="store_true")

    # Zero out CosyVoice flow prompt_token / prompt_feat (TASTE-SpokenLM style).
    # Keeps the CAMP++ flow_embedding (speaker identity) but drops the
    # autoregressive prompt-conditioning to the flow head, so the generation
    # is conditioned only on the predicted speech tokens + speaker embedding.
    ap.add_argument("--zero_prompt_audio", action="store_true",
                    help="Zero out flow prompt_token + prompt_feat (length-0 "
                         "tensors); keep flow_embedding. Mirrors "
                         "TASTE-SpokenLM/inference_audio.py.")

    args = ap.parse_args()


    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device} bf16={args.bf16}", flush=True)

    # --- model loads (once per process) ---
    t0 = time.time()
    print("[load] streamSLM checkpoint", flush=True)
    slm, tok_cfg, model_cfg = _load_streamslm(args.checkpoint, device)
    lm_tok = load_lm_tokenizer(tok_cfg)
    print(f"[load] streamSLM ready ({time.time()-t0:.1f}s) "
          f"acoustic_target={model_cfg.acoustic_target}", flush=True)

    allowed_text_token_ids = None
    if args.allowed_text_token_ids:
        allowed_text_token_ids = torch.load(
            args.allowed_text_token_ids, map_location="cpu", weights_only=True
        ).long().view(-1)
        if lm_tok.bos_token_id is not None:
            allowed_text_token_ids = torch.cat([
                allowed_text_token_ids,
                torch.tensor([int(lm_tok.bos_token_id)], dtype=torch.long),
            ]).unique()
        print(f"[vocab_mask] {allowed_text_token_ids.numel()} allowed ids "
              f"(from {args.allowed_text_token_ids})", flush=True)

    teacher_rvq = None
    if model_cfg.acoustic_target == "continuous":
        t = time.time()
        print("[load] teacher residual_vq", flush=True)
        teacher_rvq = _load_teacher_residual_vq(
            args.teacher_checkpoint,
            num_quantizers=int(os.environ.get("RVQ_R", "16")),
            codebook_size=int(os.environ.get("RVQ_CODEBOOK_SIZE", "512")),
            feat_dim=model_cfg.acoustic_feat_dim,
            device=device,
            codebook_dim=(int(os.environ["RVQ_CODEBOOK_DIM"])
                          if os.environ.get("RVQ_CODEBOOK_DIM") else None),
        )
        print(f"[load] teacher residual_vq ready ({time.time()-t:.1f}s)", flush=True)

    t = time.time()
    print("[load] StreamAlign teacher (decode head)", flush=True)
    streamalign = _load_streamalign(
        args.variant, args.teacher_checkpoint,
        args.hparams, args.truthmodel_checkpoint_path,
        args.chunk_size, args.left_context, device,
    )
    print(f"[load] StreamAlign ready ({time.time()-t:.1f}s)", flush=True)

    t = time.time()
    print("[load] CosyVoice frontend + flow + hift", flush=True)
    frontend, cosy = _load_cosy(args.cosyvoice_model_dir, device)
    print(f"[load] CosyVoice ready ({time.time()-t:.1f}s)", flush=True)

    whisper_model = None
    if not args.skip_whisper:
        t = time.time()
        import whisper
        print(f"[load] whisper {args.whisper_model}", flush=True)
        whisper_model = whisper.load_model(args.whisper_model)
        print(f"[load] whisper ready ({time.time()-t:.1f}s)", flush=True)

    utts = _read_utts(args.utts_csv)
    print(f"[plan] {len(utts)} utts to process", flush=True)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "summary.jsonl"
    summary_fp = summary_path.open("a")

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.bf16 and device.type == "cuda"
        else contextlib.nullcontext()
    )

    pipeline_t0 = time.time()
    n_done = 0
    n_skipped = 0
    n_failed = 0

    for i, utt in enumerate(utts):
        try:
            spkr, book = utt.split("-", 2)[0], utt.split("-", 2)[1]
        except (IndexError, ValueError):
            print(f"[skip] {utt}: bad utt-id format", flush=True)
            n_skipped += 1
            continue

        prompt_units_path = (
            Path(args.units_cache) / spkr / book / f"{utt}.flac.units.pt"
        )
        speaker_wav_path = (
            Path(args.speaker_root) / spkr / book / f"{utt}.flac"
        )
        utt_dir = out_root / utt
        utt_dir.mkdir(parents=True, exist_ok=True)
        units_pt = utt_dir / "sample.units.pt"
        units_txt = utt_dir / "sample.units.txt"
        out_wav = utt_dir / "sample.wav"

        if not prompt_units_path.is_file():
            print(f"[skip] {utt}: missing {prompt_units_path}", flush=True)
            n_skipped += 1
            continue
        if not speaker_wav_path.is_file():
            print(f"[skip] {utt}: missing {speaker_wav_path}", flush=True)
            n_skipped += 1
            continue

        print(f"\n===== [{i+1}/{len(utts)}] utt {utt} =====", flush=True)
        utt_t0 = time.time()

        # ---- step 1: generate (continuous -> re-quantize via teacher_rvq) ----
        prefix = SubwordUnits.load(str(prompt_units_path))
        keep = min(args.prompt_max_subwords, prefix.num_subwords) \
            if args.prompt_max_subwords > 0 else prefix.num_subwords
        prefix = SubwordUnits(
            subword_ids=prefix.subword_ids[:keep],
            q_codes=prefix.q_codes[:keep],
            duration_frames=prefix.duration_frames[:keep],
        )
        prompt_frames = int(prefix.duration_frames.sum())
        prompt_seconds = prompt_frames / args.encoder_hz
        print(f"[prompt] N={prefix.num_subwords} R={prefix.num_quantizers} "
              f"frames={prompt_frames} ({prompt_seconds:.2f}s)", flush=True)

        t_gen = time.time()
        with amp_ctx:
            units = generate(
                slm,
                prompt_ids=prefix.subword_ids.long(),
                prompt_q=prefix.q_codes.long(),
                prompt_d=prefix.duration_frames.long(),
                max_new_tokens=args.max_new_tokens,
                eos_id=lm_tok.eos_token_id,
                temperature_text=args.temperature_text,
                top_p_text=args.top_p_text,
                temperature_aco=args.temperature_aco,
                top_p_aco=args.top_p_aco,
                device=device,
                teacher_rvq=teacher_rvq,
                allowed_text_token_ids=allowed_text_token_ids,
                redo_final_aco=args.redo_final_aco,
            )
        gen_s = time.time() - t_gen
        units.save(str(units_pt))
        all_ids = units.subword_ids.tolist()
        prompt_text = lm_tok.decode(all_ids[:keep])
        gen_text = lm_tok.decode(all_ids[keep:])
        text = prompt_text + " <|PROMPT_END|> " + gen_text
        units_txt.write_text(text)
        print(f"[gen ] N={units.subword_ids.numel()} "
              f"(prompt={keep} new={units.subword_ids.numel() - keep}) "
              f"frames={int(units.duration_frames.sum())} ({gen_s:.1f}s)", flush=True)
        print(f"[gen ] prompt: {prompt_text!r}", flush=True)
        print(f"[gen ] cont  : {gen_text!r}", flush=True)

        # ---- step 2: reconstruct (units -> wav) ----
        t_rec = time.time()
        spk = prepare_speaker(
            frontend, str(speaker_wav_path), prompt_seconds=prompt_seconds,
        )
        tokens = units_to_speech_tokens(
            streamalign,
            subword_ids=units.subword_ids,
            q_codes=units.q_codes,
            durations=units.duration_frames,
            spk_emb=spk["flow_embedding"],
            lm_tokenizer=lm_tok,
        )
        if args.zero_prompt_audio:
            # TASTE-SpokenLM pattern: keep flow_embedding (speaker identity),
            # but pass empty prompt_token / prompt_feat to flow.inference.
            spk = {
                **spk,
                "flow_prompt_speech_token":
                    torch.zeros(1, 0, dtype=torch.int32, device=device),
                "flow_prompt_speech_token_len":
                    torch.zeros(1, dtype=torch.int32, device=device),
                "prompt_speech_feat":
                    torch.zeros(1, 0, 80, device=device),
                "prompt_speech_feat_len":
                    torch.zeros(1, dtype=torch.int32, device=device),
            }
        wav = speech_tokens_to_wav(cosy, tokens, spk, device)
        torchaudio.save(str(out_wav), wav.unsqueeze(0), 24000)
        rec_s = time.time() - t_rec
        wav_secs = wav.shape[-1] / 24000.0
        print(f"[recon] tokens={tuple(tokens.shape)} wav={wav_secs:.2f}s "
              f"({rec_s:.1f}s)", flush=True)

        # ---- step 3: whisper ASR (sanity) ----
        asr_text = ""
        whisper_s = 0.0
        if whisper_model is not None:
            t_w = time.time()
            w16, sr = torchaudio.load(str(out_wav))
            if sr != 16000:
                w16 = torchaudio.functional.resample(w16, sr, 16000)
            asr_text = whisper_model.transcribe(
                w16.mean(0).numpy().astype("float32"),
                language="en", fp16=False,
            )["text"].strip()
            whisper_s = time.time() - t_w
            print(f"[asr ] ({whisper_s:.1f}s) {asr_text!r}", flush=True)

        utt_s = time.time() - utt_t0
        print(f"[done] {utt} total={utt_s:.1f}s "
              f"(gen={gen_s:.1f} recon={rec_s:.1f} asr={whisper_s:.1f})",
              flush=True)
        summary_fp.write(json.dumps({
            "utt": utt,
            "n_subwords": int(units.subword_ids.numel()),
            "n_prompt_subwords": int(keep),
            "n_frames": int(units.duration_frames.sum()),
            "wav_seconds": float(wav_secs),
            "gen_seconds": gen_s,
            "recon_seconds": rec_s,
            "whisper_seconds": whisper_s,
            "total_seconds": utt_s,
            "asr": asr_text,
            "prompt_text": prompt_text,
            "gen_text": gen_text,
            "text": text,
        }) + "\n")
        summary_fp.flush()
        n_done += 1

    summary_fp.close()
    total_s = time.time() - pipeline_t0
    print(f"\n[summary] done={n_done} skipped={n_skipped} failed={n_failed} "
          f"wall={total_s:.1f}s avg={total_s/max(n_done,1):.1f}s/utt",
          flush=True)
    print(f"[summary] -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
