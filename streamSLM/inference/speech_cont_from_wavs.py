"""Speech-continuation from a directory of arbitrary 16 kHz prompt wavs.

Differs from `streamSLM.inference.speech_cont_pipeline` (which expects
precomputed `.units.pt` indexed by LibriSpeech utt-ids): here we accept any
flat directory of `.wav` prompts, run the chunk-streaming word ASR online
to recover the (text, frame_start, frame_end) intervals the teacher
needs, then extract units → generate continuation → reconstruct with
CosyVoice → optional Whisper transcript.

Output layout (matches the existing `/home/.../continuation/<exp>/`
convention used by the cascaded / TWIST / SpiritLM baselines):

    <out_dir>/<stem>_continuation.wav   only the synthesized continuation
                                        (after prompt_seconds)
    <out_dir>/<stem>_with_prompt.wav    original input wav + continuation
    <out_dir>/<stem>.json               metadata (prompt+gen text, ASR, timings)
    <out_dir>/_summary.jsonl            per-utt JSONL log
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")

import torch
import torchaudio

# Resolve sibling streamASR + CosyVoice imports the same way the rest of the
# repo does (mirrors streamSLM.eval.extractor).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_STREAMASR_ROOT = os.path.join(_REPO_ROOT, "streamASR")
_COSYVOICE_ROOT = "/home/CosyVoice"
for _p in [
    os.path.join(_STREAMASR_ROOT, "third_party", "Matcha-TTS"),
    os.path.join(_COSYVOICE_ROOT, "third_party", "Matcha-TTS"),
    _COSYVOICE_ROOT,
    _STREAMASR_ROOT,
    _REPO_ROOT,
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig  # noqa: E402

from streamSLM.units import SubwordUnits  # noqa: E402
from streamSLM.config import TokenizerConfig  # noqa: E402
from streamSLM.model.slm import load_lm_tokenizer  # noqa: E402
from streamSLM.inference.generate import (  # noqa: E402
    _load_streamslm,
    _load_teacher_residual_vq,
    _load_teacher_residual_vq,
    generate,
)
from streamSLM.inference.reconstruct import (  # noqa: E402
    _load_streamalign,
    _load_cosy,
    units_to_speech_tokens,
    prepare_speaker,
    speech_tokens_to_wav,
)
from streamSLM.eval.extractor import RVQUnitExtractor  # noqa: E402

from data.generate_chunk_textgrids import (  # noqa: E402  (streamASR.data)
    load_word_asr_model,
    stream_and_get_chunk_words,
)

SAMPLE_RATE = 16_000


def _read_wav_16k(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.dim() > 1:
        wav = wav.mean(dim=0)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav.float()


def _commits_to_intervals(
    chunk_commits: List[Tuple[str, int, int]],
    enc_chunk_size: int,
    n_audio_chunks: int,
) -> list:
    """Project chunk_commits onto the per-chunk frame grid the teacher
    expects (mirrors `write_textgrid_chunks` in `generate_chunk_textgrids.py`,
    but builds the in-memory list directly rather than via a Praat TG file).

    Returns a list of (frame_start, frame_end, text) tuples — exactly what
    `parse_textgrid_words` would yield if we had written + reparsed the TG.
    """
    chunk_data: dict = {}
    for chunk_text, frame_start, frame_end in chunk_commits:
        if frame_start not in chunk_data:
            chunk_data[frame_start] = [chunk_text, frame_end]
        else:
            chunk_data[frame_start][0] += chunk_text

    intervals = []
    for ci in range(n_audio_chunks):
        seg_start = ci * enc_chunk_size
        default_end = (ci + 1) * enc_chunk_size
        if seg_start in chunk_data:
            text, seg_end = chunk_data[seg_start]
        else:
            text, seg_end = "", default_end
        # parse_textgrid_words returns int encoder-frame indices; mirror that
        # so downstream build_char_constraints sees the same dtype.
        intervals.append((int(seg_start), int(seg_end), text))
    return intervals


def main():
    ap = argparse.ArgumentParser()

    # --- streamSLM checkpoint + RVQ-teacher knobs --------------------------- #
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--teacher_checkpoint", required=True)
    ap.add_argument("--variant", choices=["rvq"], default="rvq")

    # --- streamASR / CosyVoice decode head ---------------------------------- #
    ap.add_argument("--hparams",
                    default="/home/streamalign/streamASR/hparams/alignment.yaml")
    ap.add_argument("--truthmodel_checkpoint_path",
                    default="/home/streamalign/streamASR/results/char_asr_ckpt")
    ap.add_argument("--chunk_size", type=int, default=16)
    ap.add_argument("--left_context", type=int, default=8)
    ap.add_argument("--cosyvoice_model_dir",
                    default="/home/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B")

    # --- streaming word ASR (TG generator) ---------------------------------- #
    ap.add_argument("--word_asr_hparams",
                    default="/home/streamalign/streamASR/hparams/chunk_streaming_word_fastemit.yaml")
    ap.add_argument("--word_asr_checkpoint",
                    default="/home/streamalign/streamASR/results/conformer_transducer_char/word_fastemit/save/word_asr_ckpt")
    ap.add_argument("--word_asr_tokenizer",
                    default="/home/streamalign/streamASR/results/conformer_transducer_char/word_fastemit/pretrained/tokenizer.ckpt")
    ap.add_argument("--word_chunk_size", type=int, default=4)
    ap.add_argument("--word_left_context", type=int, default=32)

    # --- I/O --------------------------------------------------------------- #
    ap.add_argument("--input_wav_dir", required=True,
                    help="Directory of .wav prompt files; processed in sorted order.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_utts", type=int, default=0,
                    help="Optional cap (0 = all).")

    # --- generation knobs --------------------------------------------------- #
    ap.add_argument("--prompt_max_subwords", type=int, default=0,
                    help="0 = use all subwords from the prompt wav.")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature_text", type=float, default=0.9)
    ap.add_argument("--top_p_text", type=float, default=0.95)
    ap.add_argument("--temperature_aco", type=float, default=0.0)
    ap.add_argument("--top_p_aco", type=float, default=1.0)
    ap.add_argument("--redo_final_aco", type=int, default=0,
                    help="Re-predict the last K acoustic positions before reconstruction.")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--encoder_hz", type=float, default=25.0)
    ap.add_argument("--allowed_text_token_ids", default="",
                    help="Optional .pt with allowed LM-tokenizer ids (1-D long).")

    # --- teacher knobs (used by the extractor + streamalign decoder) -------- #

    # --- Whisper ASR (sanity) ---------------------------------------------- #
    ap.add_argument("--whisper_model", default="base.en")
    ap.add_argument("--skip_whisper", action="store_true")

    ap.add_argument("--zero_prompt_audio", action="store_true",
                    help="Zero out flow prompt_token + prompt_feat; keep flow_embedding.")

    args = ap.parse_args()


    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device} bf16={args.bf16}", flush=True)

    # ---------------- model loads ----------------------------------------- #
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
        print(f"[vocab_mask] {allowed_text_token_ids.numel()} allowed ids", flush=True)

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
    print("[load] unit extractor (shares teacher_checkpoint)", flush=True)
    extractor = RVQUnitExtractor(
        teacher_checkpoint=args.teacher_checkpoint,
        variant=args.variant,
        hparams=args.hparams,
        truthmodel_checkpoint=args.truthmodel_checkpoint_path,
        chunk_size=args.chunk_size,
        left_context=args.left_context,
        rvq_num_quantizers=int(os.environ.get("RVQ_R", "16")),
        rvq_codebook_size=int(os.environ.get("RVQ_CODEBOOK_SIZE", "512")),
        device=device,
        bf16=args.bf16,
    )
    print(f"[load] unit extractor ready ({time.time()-t:.1f}s)", flush=True)

    t = time.time()
    print("[load] CosyVoice frontend + flow + hift", flush=True)
    frontend, cosy = _load_cosy(args.cosyvoice_model_dir, device)
    print(f"[load] CosyVoice ready ({time.time()-t:.1f}s)", flush=True)

    t = time.time()
    print("[load] streaming word ASR (TG generator)", flush=True)
    word_asr = load_word_asr_model(
        args.word_asr_hparams, args.word_asr_checkpoint,
        args.word_asr_tokenizer, device,
    )
    word_cfg = DynChunkTrainConfig(
        chunk_size=args.word_chunk_size,
        left_context_size=args.word_left_context,
    )
    print(f"[load] streaming word ASR ready ({time.time()-t:.1f}s)", flush=True)

    whisper_model = None
    if not args.skip_whisper:
        t = time.time()
        import whisper
        print(f"[load] whisper {args.whisper_model}", flush=True)
        whisper_model = whisper.load_model(args.whisper_model)
        print(f"[load] whisper ready ({time.time()-t:.1f}s)", flush=True)

    # ---------------- discover input wavs --------------------------------- #
    in_dir = Path(args.input_wav_dir)
    wav_paths = sorted(in_dir.glob("*.wav"))
    if args.max_utts > 0:
        wav_paths = wav_paths[: args.max_utts]
    print(f"[plan] {len(wav_paths)} input wavs from {in_dir}", flush=True)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "_summary.jsonl"
    summary_fp = summary_path.open("a")

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.bf16 and device.type == "cuda"
        else contextlib.nullcontext()
    )

    pipeline_t0 = time.time()
    n_done = 0
    n_failed = 0

    for i, wav_path in enumerate(wav_paths):
        stem = wav_path.stem
        try:
            print(f"\n===== [{i+1}/{len(wav_paths)}] {wav_path.name} =====", flush=True)
            utt_t0 = time.time()
            wav16 = _read_wav_16k(str(wav_path))

            # ---- step 0: streaming-ASR -> chunk intervals --------------- #
            t_tg = time.time()
            torch_dyn_cache = None  # noqa: F841  (streamingcontext is built per-call)
            # stream_and_get_chunk_words loads + resamples the wav itself, so
            # we just pass the path.
            chunk_commits, n_chunks = stream_and_get_chunk_words(
                word_asr, str(wav_path), word_cfg,
            )
            intervals = _commits_to_intervals(
                chunk_commits, args.word_chunk_size, n_chunks,
            )
            tg_s = time.time() - t_tg
            asr_words = " ".join(t for (_, _, t) in intervals if t.strip())
            print(f"[asr ] {n_chunks} chunks, {tg_s:.1f}s, prompt_text={asr_words!r}",
                  flush=True)

            # ---- step 1: extract RVQ units from the prompt ------------- #
            t_ex = time.time()
            prompt_units = extractor.extract_one(wav16, intervals=intervals)
            ex_s = time.time() - t_ex
            if prompt_units.num_subwords == 0:
                print(f"[skip] {stem}: extractor produced 0 subwords", flush=True)
                n_failed += 1
                continue
            keep = (min(args.prompt_max_subwords, prompt_units.num_subwords)
                    if args.prompt_max_subwords > 0 else prompt_units.num_subwords)
            prefix = SubwordUnits(
                subword_ids=prompt_units.subword_ids[:keep],
                q_codes=prompt_units.q_codes[:keep],
                duration_frames=prompt_units.duration_frames[:keep],
            )
            prompt_frames = int(prefix.duration_frames.sum())
            prompt_seconds = prompt_frames / args.encoder_hz
            print(f"[extr] N={prefix.num_subwords} R={prefix.num_quantizers} "
                  f"frames={prompt_frames} ({prompt_seconds:.2f}s, {ex_s:.1f}s)",
                  flush=True)

            # ---- step 2: generate continuation ------------------------- #
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
                    redo_final_aco=args.redo_final_aco,
                    device=device,
                    teacher_rvq=teacher_rvq,
                    allowed_text_token_ids=allowed_text_token_ids,
                )
            gen_s = time.time() - t_gen
            all_ids = units.subword_ids.tolist()
            prompt_text = lm_tok.decode(all_ids[:keep])
            gen_text = lm_tok.decode(all_ids[keep:])
            print(f"[gen ] N={units.subword_ids.numel()} "
                  f"(prompt={keep} new={units.subword_ids.numel() - keep}) "
                  f"frames={int(units.duration_frames.sum())} ({gen_s:.1f}s)",
                  flush=True)
            print(f"[gen ] prompt: {prompt_text!r}", flush=True)
            print(f"[gen ] cont  : {gen_text!r}", flush=True)

            # ---- step 3: reconstruct ----------------------------------- #
            t_rec = time.time()
            spk = prepare_speaker(
                frontend, str(wav_path), prompt_seconds=prompt_seconds,
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
            full_wav = speech_tokens_to_wav(cosy, tokens, spk, device)  # (T,) @ 24 kHz
            rec_s = time.time() - t_rec
            full_secs = full_wav.shape[-1] / 24000.0
            print(f"[recon] tokens={tuple(tokens.shape)} wav={full_secs:.2f}s "
                  f"({rec_s:.1f}s)", flush=True)

            # ---- step 4: slice into continuation-only + with-prompt ---- #
            # CosyVoice synth has its own time base; the prompt subwords were
            # resynthesized at the start of `full_wav`. Cut at prompt_seconds
            # to isolate the new continuation, then concatenate with the
            # original 16 kHz input wav (resampled to 24 kHz) for the
            # "with_prompt" delivery so the user hears the *real* prompt.
            n_prompt_samples_24k = int(round(prompt_seconds * 24000))
            n_prompt_samples_24k = min(n_prompt_samples_24k, full_wav.shape[-1])
            cont_only = full_wav[n_prompt_samples_24k:].contiguous()

            input_24k = torchaudio.functional.resample(
                wav16.unsqueeze(0), SAMPLE_RATE, 24000,
            ).squeeze(0)
            with_prompt = torch.cat([input_24k, cont_only], dim=0)

            out_cont = out_root / f"{stem}_continuation.wav"
            out_full = out_root / f"{stem}_with_prompt.wav"
            torchaudio.save(str(out_cont), cont_only.unsqueeze(0), 24000)
            torchaudio.save(str(out_full), with_prompt.unsqueeze(0), 24000)

            # ---- step 5: whisper ASR (sanity) -------------------------- #
            asr_text_full = ""
            asr_text_cont = ""
            whisper_s = 0.0
            if whisper_model is not None and cont_only.numel() > 0:
                t_w = time.time()
                w16_cont = torchaudio.functional.resample(
                    cont_only.unsqueeze(0), 24000, 16000,
                ).mean(0).numpy().astype("float32")
                asr_text_cont = whisper_model.transcribe(
                    w16_cont, language="en", fp16=False,
                )["text"].strip()

                w16_full = torchaudio.functional.resample(
                    with_prompt.unsqueeze(0), 24000, 16000,
                ).mean(0).numpy().astype("float32")
                asr_text_full = whisper_model.transcribe(
                    w16_full, language="en", fp16=False,
                )["text"].strip()
                whisper_s = time.time() - t_w
                print(f"[asr ] cont ({whisper_s:.1f}s): {asr_text_cont!r}", flush=True)
                print(f"[asr ] full        : {asr_text_full!r}", flush=True)

            # ---- step 6: save per-utt metadata ------------------------- #
            utt_s = time.time() - utt_t0
            meta = {
                "stem": stem,
                "input_wav": str(wav_path),
                "n_subwords": int(units.subword_ids.numel()),
                "n_prompt_subwords": int(keep),
                "n_total_frames": int(units.duration_frames.sum()),
                "duration_frames": units.duration_frames.tolist(),
                "prompt_seconds": float(prompt_seconds),
                "prompt_duration_frames": prefix.duration_frames.tolist(),
                "cont_seconds": float(cont_only.shape[-1] / 24000.0),
                "full_seconds": float(full_secs),
                "with_prompt_seconds": float(with_prompt.shape[-1] / 24000.0),
                "prompt_text_streaming_asr": asr_words,
                "prompt_text_subwords": prompt_text,
                "gen_text_subwords": gen_text,
                "whisper_continuation": asr_text_cont,
                "whisper_with_prompt": asr_text_full,
                "timings": {
                    "tg": tg_s, "extract": ex_s, "gen": gen_s,
                    "recon": rec_s, "whisper": whisper_s, "total": utt_s,
                },
            }
            (out_root / f"{stem}.json").write_text(json.dumps(meta, indent=2))
            summary_fp.write(json.dumps(meta) + "\n")
            summary_fp.flush()

            print(f"[done] {stem} total={utt_s:.1f}s "
                  f"(tg={tg_s:.1f} ex={ex_s:.1f} gen={gen_s:.1f} "
                  f"rec={rec_s:.1f} asr={whisper_s:.1f})", flush=True)
            n_done += 1
        except Exception as exc:  # noqa: BLE001
            import traceback
            print(f"[fail] {stem}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            n_failed += 1

    summary_fp.close()
    total_s = time.time() - pipeline_t0
    print(f"\n[summary] done={n_done} failed={n_failed} "
          f"wall={total_s:.1f}s avg={total_s/max(n_done,1):.1f}s/utt", flush=True)
    print(f"[summary] -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
