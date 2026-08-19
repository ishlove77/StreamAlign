"""Training utilities: loss computation, batch preprocessing, checkpoint helpers."""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torchaudio
import torch.distributed as dist

from utils.text_utils import PAD_ID, batch_word_char_alignment, CHAR_TO_IDX
from utils.rnnt_align_loss import build_char_constraints


# ── Distributed helpers ─────────────────────────────────────────────────────────

def is_main_process() -> bool:
    return dist.get_rank() == 0


# ── Checkpoint helpers ──────────────────────────────────────────────────────────

def encoder_compatible_transfer(obj, path):
    """Pretrainer custom hook: load only shape-compatible weights.

    Signature matches speechbrain.utils.parameter_transfer.Pretrainer
    custom_hooks: (obj, path) -> None.

    Skips layers whose shapes differ (e.g. vocab-dependent heads when
    adapting a BPE-pretrained model to a character vocabulary).
    """
    state_dict = torch.load(path, map_location="cpu")
    load_filtered_state(obj, state_dict, name="pretrained model")


def load_filtered_state(model, ckpt_state: dict, name: str):
    """Load checkpoint state, silently skipping shape-mismatched keys.

    Returns the set of state-dict keys that were actually loaded.
    """
    model_dict = model.state_dict()
    filtered = {
        k: v for k, v in ckpt_state.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    skipped = [k for k in ckpt_state if k not in filtered and k in model_dict]
    if is_main_process() and skipped:
        print(f"Skipping {len(skipped)} {name} keys due to size mismatch:")
        for k in skipped:
            print(f"  • {k}: ckpt {tuple(ckpt_state[k].shape)} vs model {tuple(model_dict[k].shape)}")
    model_dict.update(filtered)
    missing, unexpected = model.load_state_dict(model_dict, strict=False)
    if is_main_process():
        print(f"{name} loaded → {len(missing)} missing, {len(unexpected)} unexpected keys\n")
    return set(filtered.keys())


def load_filtered_optimizer_state(optimizer, ckpt_optim_state, loaded_keys, ckpt_model_state, model):
    """Load optimizer state only for parameters that were actually loaded.

    When the model architecture changes between checkpoint and current run,
    ``optimizer.load_state_dict`` fails because the parameter count differs.
    This function remaps the checkpoint optimizer state so that only parameters
    present in *loaded_keys* (returned by :func:`load_filtered_state`) are
    restored; new or reshaped parameters start with fresh optimizer state.
    """
    current_param_names = [name for name, _ in model.named_parameters()]
    new_name_to_idx = {name: i for i, name in enumerate(current_param_names)}

    # Infer old param name → index from checkpoint model state_dict key order.
    # Buffer keys (present in state_dict but not in named_parameters) are
    # identified via the *current* model and skipped.
    current_buffer_names = set(model.state_dict().keys()) - set(current_param_names)

    old_name_to_idx = {}
    param_idx = 0
    for k in ckpt_model_state.keys():
        if k in current_buffer_names:
            continue
        old_name_to_idx[k] = param_idx
        param_idx += 1

    # Sanity check: inferred param count must match checkpoint optimizer.
    ckpt_param_count = sum(len(g["params"]) for g in ckpt_optim_state["param_groups"])
    if param_idx != ckpt_param_count:
        if is_main_process():
            print(
                f"Optimizer: inferred {param_idx} params from checkpoint model state "
                f"but optimizer expects {ckpt_param_count} — skipping restore."
            )
        return

    # Remap: copy old optimizer state to new indices for loaded parameters.
    old_state = ckpt_optim_state.get("state", {})
    new_state = {}
    for name in loaded_keys:
        if name in old_name_to_idx and name in new_name_to_idx:
            old_idx = old_name_to_idx[name]
            new_idx = new_name_to_idx[name]
            if old_idx in old_state:
                new_state[new_idx] = old_state[old_idx]

    # Build a state_dict that matches the *current* optimizer structure.
    current_sd = optimizer.state_dict()
    new_sd = {"state": new_state, "param_groups": current_sd["param_groups"]}
    optimizer.load_state_dict(new_sd)

    if is_main_process():
        print(f"Optimizer: restored state for {len(new_state)}/{len(current_param_names)} parameters")


# ── Loss computation ────────────────────────────────────────────────────────────

def compute_losses(batch, model, model_out, char_alignment, word_alignment, use_device, use_recon_loss=False):
    """Loss computation using RNN-T alignment (no CTC, no char loss)."""
    main_out = model_out["forward_output"]

    if char_alignment is not None and "u_logits" in main_out:
        u_gt = batch["speech_units"].to(use_device)
        u_logits = main_out["u_logits"]
        min_len = min(u_logits.shape[2], u_gt.shape[1])
        loss_u = nn.CrossEntropyLoss()(u_logits[:, :, :min_len], u_gt[:, :min_len])
    else:
        loss_u = torch.tensor(0.0, device=use_device, requires_grad=True)

    loss_vq = main_out.get(
        "vq_loss", torch.tensor(0.0, device=use_device, requires_grad=True)
    )

    recon_loss = main_out.get(
        "recon_loss", torch.tensor(0.0, device=use_device, requires_grad=True)
    )

    total = loss_u + loss_vq
    if use_recon_loss:
        total = total + recon_loss
    return {"total_loss": total, "loss_u": loss_u, "loss_vq": loss_vq, "recon_loss": recon_loss}


def alignment_to_char_segments(char_alignment_path: torch.Tensor, device):
    """Convert character alignment path to per-character frame durations."""
    return char_alignment_path.sum(dim=2).long().to(device)


# ── Batch preprocessing ─────────────────────────────────────────────────────────

def _build_textgrid_constraints(
    textgrid_intervals_batch: List[Optional[list]],
):
    """Build per-sample char IDs and RNNT waypoint constraints from TextGrid intervals.

    Parameters
    ----------
    textgrid_intervals_batch : list of (list[(t_s, t_e, text)] or None)

    Returns
    -------
    char_ids_list : list of (list[int] or None)
    constraints_list : list of (list[tuple] or None)
    """
    char_ids_list = []
    constraints_list = []
    for intervals in textgrid_intervals_batch:
        if intervals is None:
            char_ids_list.append(None)
            constraints_list.append(None)
        else:
            char_ids, constraints = build_char_constraints(intervals, CHAR_TO_IDX)
            char_ids_list.append(char_ids if char_ids else None)
            constraints_list.append(constraints if constraints else None)
    return char_ids_list, constraints_list


def preprocess_batch(
    batch_data: Dict[str, torch.Tensor],
    frontend,
    glob_tok,
    device: torch.device,
    mask_prob: float = 0.08,
    max_wav_seconds: float = 30.0,
    sample_rate: int = 16000,
    txt_normalizer=None,
    extract_only: bool = False,
) -> Dict[str, Any]:
    """Preprocessing: mel spectrograms, speech units, speaker embeddings, char metadata.

    extract_only=True: skip mel-spec, CosyVoice speech-token ONNX, and CAMP++
    speaker-embedding ONNX. extract_subword_units() does not consume any of
    those — only waveforms, wav_mask, char_ids_list, constraints_list, char_data
    are needed — so the lean path is ~2-3x faster on the hot loop.
    """
    waveforms = batch_data["waveforms"].to(device)
    wav_lens = batch_data["wav_lens"].to(device)
    wav_mask = batch_data["wav_mask"].to(device)
    B = waveforms.size(0)

    # Build char constraints from TextGrid data (present when LibriTTSDataset is used)
    textgrid_intervals = batch_data.get("textgrid_intervals", [None] * B)
    char_ids_list, constraints_list = _build_textgrid_constraints(textgrid_intervals)

    if extract_only:
        mel_batch = None
        mel_lengths_tensor = None
    else:
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=80,
            f_min=0,
            f_max=8000,
        ).to(device)

        mel_spectrograms, mel_lengths = [], []
        for i, wav in enumerate(waveforms):
            actual_len = wav_lens[i].item()
            mel_spec = mel_transform(wav[:actual_len].unsqueeze(0))
            mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
            mel_spectrograms.append(mel_spec.squeeze(0))
            mel_lengths.append(mel_spec.size(-1))

        max_mel_len = max(mel_lengths)
        mel_batch = torch.zeros(B, 80, max_mel_len, device=device)
        for i, mel in enumerate(mel_spectrograms):
            mel_batch[i, :, :mel_lengths[i]] = mel
        mel_lengths_tensor = torch.tensor(mel_lengths, dtype=torch.long, device=device)

    # Build char_data from TextGrid-derived char IDs
    from utils.text_utils import IDX_TO_CHAR, CHAR_TO_IDX as _C2I
    char_data = []
    for b in range(B):
        cids = char_ids_list[b] or []
        char_seq = [IDX_TO_CHAR.get(i, " ") for i in cids]
        full_text = "".join(char_seq)
        # Word-token alignment for downstream segment_pass (best-effort via glob_tok)
        if cids and glob_tok is not None:
            try:
                enc = glob_tok(full_text, add_special_tokens=False,
                               return_offsets_mapping=True, return_attention_mask=False)
                offsets = enc["offset_mapping"]
                token_ids = enc["input_ids"]
                tokens = glob_tok.convert_ids_to_tokens(token_ids)
                c2w = [-1] * len(cids)
                for tok_idx, (s, e) in enumerate(offsets):
                    for pos in range(s, min(e, len(c2w))):
                        c2w[pos] = tok_idx
            except Exception:
                tokens, token_ids, c2w = [], [], [-1] * len(cids)
        else:
            tokens, token_ids, c2w = [], [], [-1] * len(cids)

        char_data.append({
            "char_sequence":    char_seq,
            "char_indices":     cids,
            "char_to_word_map": c2w,
            "tokens":           tokens,
            "word_token_ids":   token_ids,
        })

    max_samples = int(max_wav_seconds * sample_rate)
    if extract_only:
        units_pad = None
        spk_emb = None
    else:
        units_list = []
        for i, wav in enumerate(waveforms):
            actual_len = wav_lens[i].item()
            wav_actual = wav[:actual_len]
            if actual_len <= max_samples:
                units, _ = frontend._extract_speech_token(wav_actual.unsqueeze(0))
                units_list.append(units.squeeze(0))
            else:
                chunk_units = []
                for start in range(0, actual_len, max_samples):
                    u, _ = frontend._extract_speech_token(
                        wav_actual[start:min(start + max_samples, actual_len)].unsqueeze(0)
                    )
                    chunk_units.append(u.squeeze(0))
                units_list.append(torch.cat(chunk_units, dim=0))

        max_len_speech = max(u.size(0) for u in units_list)
        units_pad = torch.full((B, max_len_speech), PAD_ID, dtype=torch.long, device=device)
        for i, u in enumerate(units_list):
            u = u.to(device)
            units_pad[i, :u.size(0)] = u

        spk_embeddings = []
        for i, wav in enumerate(waveforms):
            actual_len = wav_lens[i].item()
            wav_actual = wav[:actual_len]
            if actual_len <= max_samples:
                spk_embeddings.append(frontend._extract_spk_embedding(wav_actual.unsqueeze(0)))
            else:
                chunk_embs = [
                    frontend._extract_spk_embedding(
                        wav_actual[start:min(start + max_samples, actual_len)].unsqueeze(0)
                    )
                    for start in range(0, actual_len, max_samples)
                ]
                spk_embeddings.append(torch.stack(chunk_embs).mean(dim=0))
        spk_emb = torch.cat(spk_embeddings).to(device)

    return {
        "waveforms":        waveforms,
        "wav_mask":         wav_mask,
        "wav_lens":         wav_lens,
        "mel_spectrograms": mel_batch,
        "mel_lengths":      mel_lengths_tensor,
        "speech_units":     units_pad,
        "spk_emb":          spk_emb,
        "char_data":        char_data,
        "file_paths":       batch_data["file_paths"],
        # TextGrid-derived alignment constraints
        "char_ids_list":    char_ids_list,    # list[list[int] or None]
        "constraints_list": constraints_list,  # list[list[tuple] or None]
    }


def process_batch(
    model,
    preprocessed: Dict[str, Any],
    device: torch.device,
    detach: bool = False,
    noise_aug=lambda x: x,
    tokenizer=None,
    txt_normalizer=None,
    vq_blend: float = 1.0,
) -> Dict[str, Any]:
    """Single forward pass with RNN-T alignment.

    When TextGrid constraints are present in ``preprocessed``, the model uses
    constrained Viterbi alignment instead of unconstrained greedy decoding.
    """
    waveforms = preprocessed["waveforms"]
    wav_mask = preprocessed["wav_mask"]
    spk_emb = preprocessed["spk_emb"]

    noisy_waveforms = noise_aug(waveforms)

    model_out = model(
        waveforms=noisy_waveforms,
        wav_mask=wav_mask,
        spk_emb=spk_emb,
        detach=detach,
        tokenizer=tokenizer,
        txt_normalizer=txt_normalizer,
        char_ids_list=preprocessed.get("char_ids_list"),
        constraints_list=preprocessed.get("constraints_list"),
        vq_blend=vq_blend,
    )

    return {
        "forward_output": model_out,
        "char_alignment": model_out.get("char_alignment"),
        "word_alignment": model_out.get("word_alignment"),
    }