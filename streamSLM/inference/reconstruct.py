#!/usr/bin/env python3
"""Reconstruct waveform from a generated .units.pt produced by generate.py.

Pipeline:
    .units.pt (subword_ids, q_codes [N,R], duration_frames [N])
        |  for each subword n, sum RVQ.indices_to_codes(q_codes[n,r]) for r=1..R
        v
    z_hat per subword (N, dim=256)
        |  repeat each row by duration_frames[n]  (encoder hop = 40 ms / 25 Hz)
        v
    z_hat per frame (T, 256)
        |  StreamAlign.expanded_forward(exp_feat=z_hat_perframe, word_alignment, mask)
        v
    CosyVoice token logits (1, V, T) -> argmax -> (1, T) speech tokens
        |  CosyVoice flow.inference (with prompt_token / prompt_feat / spk_emb)
        v
    mel (1, n_mels, T_mel)
        |  CosyVoice hift.inference
        v
    waveform (1, samples)

The StreamAlign student + ResidualVQ + CosyVoice frontend/flow/hift come
straight from the same loading path used in extract_tokens.py, so we share
the WANDB_MODE shim and the alignment.yaml / truth-model defaults.

Usage:
    sr 1 24 python -m streamSLM.inference.reconstruct \
        --units_pt out/sample.units.pt \
        --streamalign_ckpt weights/Streamalign-R16/rvq_teacher/epoch_22.pt \
        --speaker_wav prompt.wav \
        --out_wav out/sample.wav
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

# Match extract_tokens.py: silence wandb before importing streamASR/cosyvoice.
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")

import torch
import torch.nn.functional as F
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

from cosyvoice.cli.frontend import CosyVoiceFrontEnd  # type: ignore  # noqa: E402
from cosyvoice.cli.model import CosyVoiceModel  # type: ignore  # noqa: E402
from hyperpyyaml import load_hyperpyyaml  # noqa: E402

from utils.text_utils import CHAR_TO_IDX  # noqa: E402

from streamSLM.units import SubwordUnits  # noqa: E402
from streamSLM.config import TokenizerConfig  # noqa: E402
from streamSLM.model.slm import load_lm_tokenizer  # noqa: E402


# --------------------------------------------------------------------------- #
# StreamAlign loader (mirrors extract_tokens.py)
# --------------------------------------------------------------------------- #
def _load_model_class(variant: str = "rvq"):
    """RVQ is the only supported quantizer."""
    if variant != "rvq":
        raise ValueError(f"unsupported variant: {variant} (only 'rvq')")
    from models.model_tokenizer import Data2VecSemanticAcousticModel as M
    return M


def _load_streamalign(variant, ckpt_path, hparams, truthmodel_ckpt,
                      chunk_size, left_context, device):
    Cls = _load_model_class(variant)
    model = Cls(
        chunk_size=chunk_size,
        left_context=left_context,
        hparams_file=hparams,
        checkpoint_path=truthmodel_ckpt,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("hubert_state_dict", ckpt.get("model", ckpt))
    if any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    return model.to(device).eval()


def _load_cosy(model_dir, device):
    cfg = os.path.join(model_dir, "cosyvoice3.yaml")
    with open(cfg) as f:
        configs = load_hyperpyyaml(f)
    frontend = CosyVoiceFrontEnd(
        get_tokenizer=configs["get_tokenizer"],
        feat_extractor=configs["feat_extractor"],
        campplus_model=f"{model_dir}/campplus.onnx",
        speech_tokenizer_model=f"{model_dir}/speech_tokenizer_v3.onnx",
        allowed_special=configs["allowed_special"],
    )
    cosy = CosyVoiceModel(None, configs["flow"], configs["hift"])
    cosy.flow.load_state_dict(torch.load(f"{model_dir}/flow.pt", map_location=device))
    cosy.hift.load_state_dict(torch.load(f"{model_dir}/hift.pt", map_location=device), strict=False)
    cosy.flow.to(device).eval()
    cosy.hift.to(device).eval()
    return frontend, cosy


# --------------------------------------------------------------------------- #
# Core reconstruction
# --------------------------------------------------------------------------- #
def _subwords_to_chars(subword_ids: torch.Tensor, lm_tokenizer):
    """Map per-subword LLM IDs to (char_indices, char_to_word_map).

    Each subword is decoded individually so we keep its position in the source
    sequence stable; chars not in StreamAlign's CHAR_VOCAB (e.g., BPE leading
    space markers, special tokens) are dropped, which mirrors what
    `text_to_char_sequence` does during training. char_to_word_map[i] gives
    the source-subword index (0..N-1) of char_indices[i].
    """
    char_indices: list[int] = []
    c2w: list[int] = []
    for tok_idx, sw_id in enumerate(subword_ids.tolist()):
        # decode([single]) emits the leading space (Llama "ĠOnce" -> " Once")
        # which keeps inter-word spaces in the char stream.
        s = lm_tokenizer.decode([sw_id])
        for ch in s:
            ch_low = ch.lower()
            if ch_low in CHAR_TO_IDX:
                char_indices.append(CHAR_TO_IDX[ch_low])
                c2w.append(tok_idx)
    return char_indices, c2w


@torch.no_grad()
def units_to_speech_tokens(
    model,
    subword_ids: torch.Tensor,
    q_codes: torch.Tensor,
    durations: torch.Tensor,
    spk_emb: torch.Tensor,
    lm_tokenizer,
) -> torch.Tensor:
    """Replicate StreamAlign's training-time `_segment_pass` -> `expanded_forward`
    on per-subword units, returning CosyVoice speech-token IDs.

    Builds the same 1472-d fused per-frame feature the trained head expects:
        seg_feat = cat([seg_word_emb, seg_z_q, spk_emb])     per word
        exp_feat = expand_word_features(seg_feat, word_alignment)  per frame
    where
        seg_word_emb : char-embedding stream pooled per word (1024-d)
        seg_z_q      : sum_r residual_vq.layers[r].indices_to_codes(q_codes[:,r])  (256-d)
        spk_emb      : CAMP++ embedding broadcast across words            (192-d)
        word_alignment is the one-hot frame->word mask built from durations.
    """
    device = next(model.parameters()).device
    subword_ids = subword_ids.to(device).long().view(-1)
    if q_codes.dim() == 1:
        q_codes = q_codes.unsqueeze(-1)
    q_codes = q_codes.to(device).long()
    durations = durations.to(device).long().clamp_min(1).view(-1)
    spk_emb = spk_emb.to(device).float().view(1, -1)

    N = subword_ids.numel()
    R = q_codes.size(1)
    if N == 0:
        return torch.zeros(1, 0, dtype=torch.long, device=device)

    char_indices, c2w = _subwords_to_chars(subword_ids, lm_tokenizer)
    if not char_indices:
        return torch.zeros(1, 0, dtype=torch.long, device=device)
    char_indices_t = torch.tensor(char_indices, dtype=torch.long, device=device)
    c2w_t = torch.tensor(c2w, dtype=torch.long, device=device)
    T_chars = char_indices_t.numel()

    # 1) Char embeddings (1, T_chars, word_emb_dim)
    seg_char_emb = model._get_char_embeddings(char_indices_t.unsqueeze(0))

    # 2) word<->char alignments (training builds these in _segment_pass).
    char_one_hot = F.one_hot(c2w_t, num_classes=N).float()        # (T_chars, N)
    word_to_char_alignment = char_one_hot.T.unsqueeze(0)          # (1, N, T_chars)
    char_to_char_alignment = torch.ones(1, T_chars, T_chars, device=device)

    # 3) Pool char embeddings per word, then dense (1, N, word_emb_dim).
    aggregated_char_embs, char_seg_info = model._pool_character_segments_transformer(
        seg_char_emb, word_to_char_alignment, char_to_char_alignment,
    )
    word_emb_dim = aggregated_char_embs.size(-1)
    seg_word_emb = model._reconstruct_word_features_batch(
        aggregated_char_embs, char_seg_info, 1, N, word_emb_dim, device,
    )                                                              # (1, N, 1024)

    # 4) Reconstruct seg_z_q from residual codebook indices.
    # get_output_from_indices works for both ResidualVQ (plain RVQ) and
    # ResidualVQ — sums codes across all R layers and applies project_out
    # (Identity when codebook_dim == feat_dim).
    rvq = model.residual_vq
    seg_z_q = rvq.get_output_from_indices(q_codes.unsqueeze(0))    # (1, N, D)

    # 5) seg_feat = [seg_word_emb || seg_z_q || spk_emb_expanded]  (1, N, in_seg)
    spk_word = spk_emb.unsqueeze(1).expand(-1, N, -1)              # (1, N, spk_emb_dim)
    seg_feat = torch.cat([seg_word_emb, seg_z_q, spk_word], dim=-1)

    # 6) word_alignment from durations: one-hot (1, N, T_speech).
    T_speech = int(durations.sum().item())
    word_id_per_frame = torch.repeat_interleave(
        torch.arange(N, device=device), durations,
    )                                                              # (T_speech,)
    word_alignment = F.one_hot(word_id_per_frame, num_classes=N) \
        .T.unsqueeze(0).float()                                    # (1, N, T_speech)

    # 7) Expand to per-frame fused feature, run expanded transformer head.
    exp_feat = model.expand_word_features(seg_feat, word_alignment)  # (1, T_speech, 1472)
    padding_mask = torch.zeros(1, T_speech, dtype=torch.bool, device=device)
    u_logits = model.expanded_forward(exp_feat, padding_mask, word_alignment)  # (1, V, T)
    return u_logits.argmax(dim=1)                                  # (1, T_speech)


@torch.no_grad()
def prepare_speaker(
    frontend,
    speaker_wav: str,
    sr: int = 16000,
    prompt_seconds: float | None = None,
):
    """Run CosyVoice's zero-shot frontend once; reuse outputs for SLM + flow.

    Pass an empty zero_shot_spk_id to force inline (per-call) speaker
    extraction from `speaker_wav` rather than a pre-registered speaker.

    If `prompt_seconds` is given, truncate `speaker_wav` to that duration so
    CosyVoice's prompt_token / prompt_feat / embedding are derived from the
    EXACT same audio segment that the SLM was prompted with (the first
    PROMPT_MAX_SUBWORDS subwords). The frontend's `_extract_speech_feat`
    requires a path (it calls `load_wav` internally), so we write the
    truncated waveform to a temp file.
    """
    if prompt_seconds is None or prompt_seconds <= 0:
        return frontend.frontend_zero_shot("dummy", "dummy", speaker_wav, sr, "")

    wav, src_sr = torchaudio.load(speaker_wav)
    n_keep = int(round(prompt_seconds * src_sr))
    n_keep = min(n_keep, wav.shape[-1])
    wav = wav[:, :n_keep]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    try:
        torchaudio.save(tmp_path, wav, src_sr)
        return frontend.frontend_zero_shot("dummy", "dummy", tmp_path, sr, "")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@torch.no_grad()
def speech_tokens_to_wav(
    cosy, tokens: torch.Tensor, spk: dict, device,
) -> torch.Tensor:
    """tokens (1, T) long -> waveform via CosyVoice flow + hift, given a
    pre-computed speaker dict from `prepare_speaker`.
    """
    mel, _ = cosy.flow.inference(
        token=tokens.to(device),
        token_len=torch.tensor([tokens.shape[1]], dtype=torch.int32, device=device),
        prompt_token=spk["flow_prompt_speech_token"].to(device),
        prompt_token_len=spk["flow_prompt_speech_token_len"].to(device),
        prompt_feat=spk["prompt_speech_feat"].to(device),
        prompt_feat_len=spk["prompt_speech_feat_len"].to(device),
        embedding=spk["flow_embedding"].to(device),
        streaming=False,
        finalize=True,
    )
    wav, _ = cosy.hift.inference(speech_feat=mel, finalize=True)
    return wav.squeeze(0).cpu()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units_pt", required=True, help="SubwordUnits .pt produced by generate.py")
    ap.add_argument("--streamalign_ckpt", required=True)
    ap.add_argument("--variant", choices=["rvq"], default="rvq")
    ap.add_argument("--hparams", default=
                    os.environ.get("ASR_HPARAMS", os.path.join(_STREAMASR_ROOT, "hparams", "alignment.yaml")))
    ap.add_argument("--truthmodel_checkpoint_path", default=
                    os.environ.get("TRUTH_MODEL_CKPT", os.path.join(_STREAMASR_ROOT, "results", "char_asr_ckpt")))
    ap.add_argument("--chunk_size", type=int, default=16)
    ap.add_argument("--left_context", type=int, default=8)
    ap.add_argument("--cosyvoice_model_dir", default=
                    os.path.join(_COSYVOICE_ROOT, "pretrained_models", "Fun-CosyVoice3-0.5B"))
    ap.add_argument("--speaker_wav", required=True, help="Reference wav for speaker embedding + prompt")
    ap.add_argument("--out_wav", required=True)
    ap.add_argument("--prompt_units", default=None,
                    help="Optional .units.pt of the SLM prompt prefix. If set, "
                         "CosyVoice's prompt_token/prompt_feat/embedding are "
                         "derived from the matching audio segment of "
                         "--speaker_wav (first prompt_max_subwords subwords) "
                         "rather than the full utterance.")
    ap.add_argument("--prompt_max_subwords", type=int, default=0,
                    help="Number of prompt subwords (matches generate.py). "
                         "Used with --prompt_units to truncate --speaker_wav "
                         "to the prompt audio segment.")
    ap.add_argument("--encoder_hz", type=float, default=25.0,
                    help="Encoder frame rate in Hz (40 ms hop = 25 Hz).")
    ap.add_argument("--text_tokenizer", choices=["llama", "qwen3"], default="llama",
                    help="LLM tokenizer used to decode subword IDs into chars.")
    args = ap.parse_args()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device} variant={args.variant}")

    model = _load_streamalign(
        args.variant, args.streamalign_ckpt,
        args.hparams, args.truthmodel_checkpoint_path,
        args.chunk_size, args.left_context, device,
    )
    frontend, cosy = _load_cosy(args.cosyvoice_model_dir, device)
    lm_tok = load_lm_tokenizer(TokenizerConfig(text_tokenizer=args.text_tokenizer))

    units = SubwordUnits.load(args.units_pt)
    print(f"[units] N={units.num_subwords} R={units.num_quantizers} "
          f"total_frames={int(units.duration_frames.sum())}")

    prompt_seconds = None
    if args.prompt_units and args.prompt_max_subwords > 0:
        prefix = SubwordUnits.load(args.prompt_units)
        keep = min(args.prompt_max_subwords, prefix.num_subwords)
        prompt_frames = int(prefix.duration_frames[:keep].sum())
        prompt_seconds = prompt_frames / args.encoder_hz
        print(f"[prompt] truncating speaker_wav to first {keep} subwords "
              f"= {prompt_frames} frames = {prompt_seconds:.2f}s")

    spk = prepare_speaker(frontend, args.speaker_wav, prompt_seconds=prompt_seconds)
    tokens = units_to_speech_tokens(
        model,
        subword_ids=units.subword_ids,
        q_codes=units.q_codes,
        durations=units.duration_frames,
        spk_emb=spk["flow_embedding"],
        lm_tokenizer=lm_tok,
    )
    print(f"[cosy ] speech tokens: {tuple(tokens.shape)}")

    wav = speech_tokens_to_wav(cosy, tokens, spk, device)
    Path(args.out_wav).parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(args.out_wav, wav.unsqueeze(0), 24000)
    print(f"[save ] {args.out_wav}  shape={tuple(wav.shape)} sr=24000")


if __name__ == "__main__":
    main()
