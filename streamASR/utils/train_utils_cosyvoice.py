"""Training utilities: loss computation, batch preprocessing, checkpoint helpers."""

import os
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
    """Load checkpoint state, silently skipping shape-mismatched keys."""
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


def load_subalign_overrides(model, ckpt_path: str,
                            exclude_prefixes=("encoder.",),
                            name: str = "subalign"):
    """Selectively load RVQ + decoder modules from a SubAlign checkpoint.

    Reads ``ckpt["hubert_state_dict"]`` (or the file itself if it's already a
    state_dict), excludes any key whose name starts with ``exclude_prefixes``
    (default: ``encoder.*`` so the StreamAlign encoder is preserved), then
    applies the remainder non-strictly with shape filtering.
    Logs counts and per-key skip reasons on the main process.
    """
    if not ckpt_path:
        return {}
    obj = torch.load(ckpt_path, map_location="cpu")
    src = obj["hubert_state_dict"] if isinstance(obj, dict) and "hubert_state_dict" in obj else obj
    model_dict = model.state_dict()

    excluded = [k for k in src if k.startswith(exclude_prefixes)]
    candidate = {k: v for k, v in src.items() if not k.startswith(exclude_prefixes)}

    loaded, skipped_shape, dropped_missing = {}, [], []
    for k, v in candidate.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                loaded[k] = v
            else:
                skipped_shape.append((k, tuple(v.shape), tuple(model_dict[k].shape)))
        else:
            dropped_missing.append(k)

    if is_main_process():
        print(f"\n[{name}] source: {ckpt_path}")
        print(f"[{name}] excluded prefixes: {exclude_prefixes}  -> {len(excluded)} keys skipped from source")
        print(f"[{name}] loaded: {len(loaded)} / {len(candidate)} candidate keys")
        if skipped_shape:
            print(f"[{name}] shape-mismatch skipped: {len(skipped_shape)}")
            for k, s, d in skipped_shape:
                print(f"    {k}: src {s} vs model {d}")
        if dropped_missing:
            print(f"[{name}] not present in current model: {len(dropped_missing)}")
            for k in dropped_missing[:30]:
                print(f"    {k}")
            if len(dropped_missing) > 30:
                print(f"    ...and {len(dropped_missing) - 30} more")

    model_dict.update(loaded)
    missing, unexpected = model.load_state_dict(model_dict, strict=False)
    if is_main_process():
        print(f"[{name}] post-load missing in model: {len(missing)} | unexpected: {len(unexpected)}\n")
    return loaded


def load_optimizer_state_flexible(optimizer, saved_opt_state, device, model=None,
                                  saved_model_state=None):
    """Load optimizer state dict, handling parameter group size mismatches.

    The most common cause is a change in ``requires_grad`` (e.g. freezing the
    encoder).  The saved optimizer tracked *all* trainable params under one
    numbering; the current optimizer only tracks the *still-trainable* subset
    under a different numbering.

    When ``model`` and ``saved_model_state`` are provided, matching is done
    **by parameter name** which is robust to any reordering or subset change.
    Otherwise falls back to positional index matching (legacy behaviour).

    Returns the number of parameters whose state was successfully restored.
    """
    saved_states = saved_opt_state.get("state", {})
    current_params = [p for group in optimizer.param_groups for p in group["params"]]

    # Pre-validate: native loader attaches state blobs by positional index without
    # checking tensor shapes, so a d=512→d=768 architecture change is silently
    # accepted and blows up inside Adam.step(). Only take the fast path when the
    # total param count matches AND every saved exp_avg matches the current
    # param shape at the same positional index.
    shapes_ok = (
        len(saved_states) == len(current_params)
        and all(
            "exp_avg" not in saved_states.get(i, {})
            or saved_states[i]["exp_avg"].shape == current_params[i].shape
            for i in range(len(current_params))
        )
    )
    if shapes_ok:
        try:
            optimizer.load_state_dict(saved_opt_state)
            n_total = sum(len(g["params"]) for g in optimizer.param_groups)
            if is_main_process():
                print(f"Optimizer state fully restored ({n_total} parameters)")
            return n_total
        except (ValueError, RuntimeError):
            pass  # fall through to flexible loading

    # ── Name-based matching (preferred) ──────────────────────────────────
    if model is not None and saved_model_state is not None:
        # Build old_name → old_optimizer_index.
        # The saved optimizer indices correspond to all parameters that were
        # trainable at save time.  Without knowing which were frozen then,
        # assume *all* parameters were trainable (the common case).
        saved_names = list(saved_model_state.keys())
        # Filter to only nn.Parameter keys (exclude buffers) by checking
        # against the names present in the model's named_parameters.
        all_param_names = {n for n, _ in model.named_parameters()}
        old_name_to_idx = {}
        idx = 0
        for name in saved_names:
            if name in all_param_names:
                old_name_to_idx[name] = idx
                idx += 1

        # Build current_name → (new_optimizer_index, param_tensor).
        cur_name_to_info = {}
        cur_idx = 0
        for name, p in model.named_parameters():
            if p.requires_grad:
                cur_name_to_info[name] = (cur_idx, p)
                cur_idx += 1

        loaded = 0
        for name, (new_idx, param) in cur_name_to_info.items():
            old_idx = old_name_to_idx.get(name)
            if old_idx is None or old_idx not in saved_states:
                continue
            saved_s = saved_states[old_idx]
            ref_key = "exp_avg" if "exp_avg" in saved_s else None
            if ref_key is not None and saved_s[ref_key].shape != param.shape:
                continue
            optimizer.state[param] = {
                k: v.clone().to(device) if isinstance(v, torch.Tensor) else v
                for k, v in saved_s.items()
            }
            loaded += 1

        if is_main_process():
            print(f"Optimizer state restored (by name): {loaded}/{len(current_params)} parameters "
                  f"(checkpoint had {len(saved_states)} parameter states)")
        return loaded

    # ── Fallback: positional index matching ──────────────────────────────
    loaded = 0
    for new_idx, param in enumerate(current_params):
        if new_idx not in saved_states:
            continue
        saved_s = saved_states[new_idx]
        ref_key = "exp_avg" if "exp_avg" in saved_s else None
        if ref_key is not None and saved_s[ref_key].shape != param.shape:
            continue
        optimizer.state[param] = {
            k: v.clone().to(device) if isinstance(v, torch.Tensor) else v
            for k, v in saved_s.items()
        }
        loaded += 1

    if is_main_process():
        print(f"Optimizer state partially restored: {loaded}/{len(current_params)} parameters "
              f"(checkpoint had {len(saved_states)} parameter states)")
    return loaded


# ── Loss computation ────────────────────────────────────────────────────────────

def compute_losses(batch, model, model_out, char_alignment, word_alignment, use_device):
    """Loss computation using RNN-T alignment (no CTC, no char loss)."""
    main_out = model_out["forward_output"]

    if char_alignment is not None and "u_logits" in main_out:
        u_gt = batch["speech_units"].to(use_device)
        u_logits = main_out["u_logits"]
        # Optional delayed-prediction: predict u_gt[t+shift] from logits at t.
        shift = int(os.environ.get("DELAYED_PRED_SHIFT", "0"))
        if shift > 0:
            u_gt = u_gt[:, shift:]
        min_len = min(u_logits.shape[2], u_gt.shape[1])
        if min_len > 0:
            loss_u = nn.CrossEntropyLoss()(u_logits[:, :, :min_len], u_gt[:, :min_len])
        else:
            loss_u = torch.tensor(0.0, device=use_device, requires_grad=True)
    else:
        loss_u = torch.tensor(0.0, device=use_device, requires_grad=True)

    loss_vq = main_out.get(
        "vq_loss", torch.tensor(0.0, device=use_device, requires_grad=True)
    )

    # Drop a non-finite component before summing so a single bad batch
    # doesn't break the optimizer step or contaminate the epoch mean.
    if not torch.isfinite(loss_u):
        loss_u = torch.zeros((), device=use_device, dtype=loss_u.dtype)
    if not torch.isfinite(loss_vq):
        loss_vq = torch.zeros((), device=use_device, dtype=loss_vq.dtype)

    total = loss_u + loss_vq
    return {"total_loss": total, "loss_u": loss_u, "loss_vq": loss_vq}


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
) -> Dict[str, Any]:
    """Preprocessing: mel spectrograms, speech units, speaker embeddings, char metadata."""
    waveforms = batch_data["waveforms"].to(device)
    wav_lens = batch_data["wav_lens"].to(device)
    wav_mask = batch_data["wav_mask"].to(device)
    B = waveforms.size(0)

    # Build char constraints from TextGrid data (present when LibriTTSDataset is used)
    textgrid_intervals = batch_data.get("textgrid_intervals", [None] * B)
    char_ids_list, constraints_list = _build_textgrid_constraints(textgrid_intervals)

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

    if "speech_units" in batch_data and "spk_emb" in batch_data:
        # Precomputed via collate; just move to device.
        units_pad = batch_data["speech_units"].to(device)
        spk_emb_batch = batch_data["spk_emb"].to(device).float()
    else:
        max_samples = int(max_wav_seconds * sample_rate)
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
        spk_emb_batch = torch.cat(spk_embeddings).to(device)

    return {
        "waveforms":        waveforms,
        "wav_mask":         wav_mask,
        "wav_lens":         wav_lens,
        "mel_spectrograms": mel_batch,
        "mel_lengths":      mel_lengths_tensor,
        "speech_units":     units_pad,
        "spk_emb":          spk_emb_batch,
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
    )

    return {
        "forward_output": model_out,
        "char_alignment": model_out.get("char_alignment"),
        "word_alignment": model_out.get("word_alignment"),
    }
