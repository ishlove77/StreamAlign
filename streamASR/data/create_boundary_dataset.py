#!/usr/bin/env python3
"""
create_boundary_dataset.py

Extracts (enc_feat, pred_feat, label) triples from the frozen streaming ASR
model to build a dataset for training the BoundaryClassifier.

Overview
--------
For each utterance in a LibriSpeech split the script:

1. Extracts Fbank features → normalizes → CNN → conformer encoder (with
   streaming attention mask matching chunk_size / left_context).

2. Runs greedy transducer decoding with alignment tracking to obtain:
     tokens            : token IDs in emission order
     frame_of_emission : encoder frame where each token was emitted
     pred_outputs      : proj_dec output after every emitted token
                         (pred_outputs[u] = state after emitting u tokens,
                          i.e. pred_outputs[0] is the initial BOS state)

3. For every chunk boundary (last encoder frame of each chunk):
     enc_feat  = proj_enc output at the last frame of the chunk
     pred_feat = pred_outputs[u_last]   where u_last = # tokens emitted so far
     label     = 1 if tokens[u_last-1] ends a complete word, else 0

   A token ends a complete word when the NEXT token in the sequence starts
   with the SentencePiece word-start character '▁' (U+2581), or when it is
   the last token in the utterance.

   Chunks where no token has been emitted yet are skipped.

4. Saves each split as a single .pt file (dict with keys 'enc_feats',
   'pred_feats', 'labels') under <dataset_dir>.

Usage
-----
python data/create_boundary_dataset.py hparams/boundary_classifier.yaml \\
    [--max_files_per_split 500]
"""

import os
import sys
import warnings
import argparse
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torchaudio
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.checkpoints import Checkpointer
from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig

# ── SentencePiece word-start marker ──────────────────────────────────────────
SP_SPACE = "\u2581"  # ▁


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def extract_encoder_output(wav_tensor, wav_len_ratio, asr_hparams, device, dynchunk_config):
    """
    Run Fbank → normalize → CNN → Conformer encoder → proj_enc.

    Parameters
    ----------
    wav_tensor : Tensor [1, T_samples]
    wav_len_ratio : float  (typically 1.0 for full-length utterance)
    asr_hparams : dict   loaded from the ASR YAML
    device : torch.device
    dynchunk_config : DynChunkTrainConfig

    Returns
    -------
    enc_out : Tensor [T_enc, joint_dim]   (no batch dimension)
    """
    modules = asr_hparams["modules"]
    wav_lens = torch.tensor([wav_len_ratio], device=device)

    feats = asr_hparams["compute_features"](wav_tensor.to(device))  # [1, T_feat, n_mels]
    feats = modules["normalize"](feats, wav_lens, epoch=9999)
    src = modules["CNN"](feats)                                       # [1, T_cnn, feat]
    cnn_lens = wav_lens

    enc_out = modules["enc"](
        src,
        cnn_lens,
        pad_idx=asr_hparams["pad_index"],
        dynchunktrain_config=dynchunk_config,
    )                                                                  # [1, T_enc, d_model]
    enc_out = modules["proj_enc"](enc_out)                             # [1, T_enc, joint_dim]
    return enc_out[0]                                                  # [T_enc, joint_dim]


# ─────────────────────────────────────────────────────────────────────────────
# Greedy alignment
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def greedy_align(enc_out, asr_hparams, device):
    """
    Greedy transducer decoding with full alignment tracking.

    Parameters
    ----------
    enc_out : Tensor [T_enc, joint_dim]   (proj_enc output)
    asr_hparams : dict
    device : torch.device

    Returns
    -------
    tokens : list[int]
        Emitted token IDs in order (blank never included).
    frame_of_emission : list[int]
        enc_out frame index (0-based) where each token was emitted.
    pred_outputs : list[Tensor]
        proj_dec output after each predictor step.
        pred_outputs[0]   = state after processing BOS (before any real token).
        pred_outputs[u+1] = state after emitting tokens[u].
        Length = len(tokens) + 1.
    """
    modules    = asr_hparams["modules"]
    emb_mod    = modules["emb"]
    dec_mod    = modules["dec"]
    proj_dec   = modules["proj_dec"]
    Tjoint     = modules["Tjoint"]
    t_lin      = modules["transducer_lin"]
    blank_id   = asr_hparams["blank_index"]
    bos_id     = asr_hparams["bos_index"]
    T          = enc_out.shape[0]

    # ── Initialise predictor with BOS ──────────────────────────────────────
    bos_t = torch.tensor([[bos_id]], dtype=torch.long, device=device)  # [1, 1]
    emb_out = emb_mod(bos_t)                                            # [1, 1, emb_dim]
    dec_out, hidden = dec_mod(emb_out)                                  # [1,1,dec_dim], hx
    pred_proj = proj_dec(dec_out)                                       # [1, 1, joint_dim]
    pred_state = pred_proj[0, 0]                                        # [joint_dim]

    pred_outputs       = [pred_state]   # pred_outputs[0] = BOS state
    tokens             = []
    frame_of_emission  = []

    for t in range(T):
        enc_t      = enc_out[t]             # [joint_dim]
        pred_u     = pred_outputs[-1]       # [joint_dim]

        # Safety: at most 30 tokens per encoder frame
        for _ in range(30):
            # Joint (sum mode): nonlinearity(enc + pred)
            # Use the actual Tjoint module with matching 4-D shapes so we
            # respect whatever nonlinearity is configured in the ASR model.
            e4 = enc_t.view(1, 1, 1, -1)    # [1, 1, 1, joint_dim]
            p4 = pred_u.view(1, 1, 1, -1)   # [1, 1, 1, joint_dim]
            joint   = Tjoint(e4, p4)         # [1, 1, 1, joint_dim]
            logits  = t_lin(joint)           # [1, 1, 1, vocab]
            pred_id = int(logits.argmax(-1).item())

            if pred_id == blank_id:
                break

            # Emit token: update predictor
            tokens.append(pred_id)
            frame_of_emission.append(t)

            tok_t   = torch.tensor([[pred_id]], dtype=torch.long, device=device)
            emb_out = emb_mod(tok_t)                                   # [1, 1, emb_dim]
            dec_out, hidden = dec_mod(emb_out, hx=hidden)              # [1,1,dec_dim], hx
            pred_u  = proj_dec(dec_out)[0, 0]                          # [joint_dim]
            pred_outputs.append(pred_u)

    return tokens, frame_of_emission, pred_outputs


# ─────────────────────────────────────────────────────────────────────────────
# Label computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_label(token_idx, tokens, tokenizer):
    """
    Return 1 if tokens[token_idx] ends a complete word, 0 otherwise.

    A token ends a word when:
      * it is the last token in the sequence, OR
      * the next token's SentencePiece piece starts with '▁' (SP_SPACE).
    """
    if token_idx == len(tokens) - 1:
        return 1
    next_piece = tokenizer.id_to_piece(tokens[token_idx + 1])
    return 1 if next_piece.startswith(SP_SPACE) else 0


# ─────────────────────────────────────────────────────────────────────────────
# Per-file extraction
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_RATE = 16_000


@torch.no_grad()
def extract_from_file(wav_path, asr_hparams, tokenizer, device, dynchunk_config):
    """
    Process one .wav file and return lists of features and labels.

    Returns
    -------
    enc_feats  : list[Tensor([joint_dim])]
    pred_feats : list[Tensor([joint_dim])]
    labels     : list[int]   (0 or 1)

    Returns three empty lists if the utterance produces no tokens.
    """
    # Load audio
    waveform, sr = torchaudio.load(str(wav_path))
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    waveform = waveform.mean(dim=0, keepdim=True)  # [1, T]
    wav_tensor = waveform.unsqueeze(0)              # [1, 1, T] → will be squeezed in extract

    # Encoder forward pass
    # extract_encoder_output expects [1, T_samples]
    enc_out = extract_encoder_output(
        waveform,       # [1, T_samples]
        1.0,
        asr_hparams,
        device,
        dynchunk_config,
    )  # [T_enc, joint_dim]

    T_enc = enc_out.shape[0]
    if T_enc == 0:
        return [], [], []

    # Greedy alignment
    tokens, frame_of_emission, pred_outputs = greedy_align(enc_out, asr_hparams, device)

    if not tokens:
        return [], [], []

    chunk_size = dynchunk_config.chunk_size  # encoder output frames

    enc_feats  = []
    pred_feats = []
    labels     = []

    u_last = 0  # number of tokens emitted so far

    for chunk_idx in range((T_enc + chunk_size - 1) // chunk_size):
        t_end = min((chunk_idx + 1) * chunk_size - 1, T_enc - 1)

        # Advance u_last to include all tokens emitted at or before t_end
        while u_last < len(tokens) and frame_of_emission[u_last] <= t_end:
            u_last += 1

        if u_last == 0:
            continue  # No tokens emitted yet; nothing to classify

        # enc feature: last encoder frame of this chunk
        enc_f  = enc_out[t_end].cpu()              # [joint_dim]
        # pred feature: predictor state after u_last emitted tokens
        pred_f = pred_outputs[u_last].cpu()        # [joint_dim]
        # label: does the last emitted token end a word?
        label  = compute_label(u_last - 1, tokens, tokenizer)

        enc_feats.append(enc_f)
        pred_feats.append(pred_f)
        labels.append(label)

    return enc_feats, pred_feats, labels


# ─────────────────────────────────────────────────────────────────────────────
# Split processing
# ─────────────────────────────────────────────────────────────────────────────

def process_split(split_name, wav_files, asr_hparams, tokenizer, device,
                  dynchunk_config, out_path, max_files=None):
    """
    Process all .wav files in a split and save a .pt dataset file.

    The saved file is a dict:
        {
            'enc_feats':  Tensor [N, joint_dim],
            'pred_feats': Tensor [N, joint_dim],
            'labels':     Tensor [N],           (int64, 0 or 1)
        }
    """
    if max_files is not None:
        wav_files = wav_files[:max_files]

    all_enc   = []
    all_pred  = []
    all_labels = []
    skipped = 0

    for wav_path in tqdm(wav_files, desc=f"  {split_name}"):
        try:
            enc_f, pred_f, labs = extract_from_file(
                wav_path, asr_hparams, tokenizer, device, dynchunk_config
            )
            all_enc.extend(enc_f)
            all_pred.extend(pred_f)
            all_labels.extend(labs)
        except Exception as exc:
            print(f"\n    [skip] {wav_path.name}: {exc}")
            skipped += 1

    if not all_enc:
        print(f"  {split_name}: no data extracted (skipped={skipped})")
        return

    dataset = {
        "enc_feats":  torch.stack(all_enc),           # [N, joint_dim]
        "pred_feats": torch.stack(all_pred),           # [N, joint_dim]
        "labels":     torch.tensor(all_labels, dtype=torch.long),  # [N]
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, out_path)

    n       = len(all_labels)
    n_pos   = sum(all_labels)
    n_neg   = n - n_pos
    print(
        f"  {split_name}: {n} samples  |  "
        f"boundary={n_pos} ({100*n_pos/n:.1f}%)  "
        f"non-boundary={n_neg} ({100*n_neg/n:.1f}%)  |  "
        f"skipped files={skipped}"
    )
    print(f"  Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def load_asr_model(hparams, device):
    """Load the frozen streaming ASR model and return its hparams dict."""
    with open(hparams["asr_hparams_file"], encoding="utf-8") as f:
        asr_hparams = load_hyperpyyaml(f)

    asr_modules = asr_hparams["modules"]

    # Load model weights
    ckpt_dir   = Path(hparams["asr_checkpoint"])
    model_ckpt = ckpt_dir / "model.ckpt"
    norm_ckpt  = ckpt_dir / "normalizer.ckpt"

    if model_ckpt.exists():
        state = torch.load(model_ckpt, map_location=device)
        missing, unexpected = asr_hparams["model"].load_state_dict(state, strict=False)
        if missing:
            print(f"  [ASR] Missing keys: {missing}")
        if unexpected:
            print(f"  [ASR] Unexpected keys (first 3): {unexpected[:3]}")
    else:
        recoverables = {"model": asr_hparams["model"], "normalizer": asr_hparams["normalize"]}
        checkpointer = Checkpointer(str(ckpt_dir.parent), recoverables=recoverables)
        checkpointer.recover_if_possible()

    if norm_ckpt.exists():
        norm_state = torch.load(norm_ckpt, map_location=device)
        norm_mod   = asr_modules.get("normalize")
        if norm_mod and isinstance(norm_state, dict) and "glob_mean" in norm_state:
            norm_mod.glob_mean = norm_state["glob_mean"]
            norm_mod.glob_std  = norm_state["glob_std"]
            norm_mod.count     = norm_state.get("count", 1)

    # Move all modules to device and set to eval
    asr_hparams["model"].to(device).eval()
    asr_hparams["compute_features"].to(device)
    for mod in asr_modules.values():
        if hasattr(mod, "to"):
            mod.to(device)
    torch.set_grad_enabled(False)

    return asr_hparams


def load_tokenizer(hparams, asr_hparams):
    """Load the SentencePiece tokenizer from the checkpoint."""
    tokenizer = asr_hparams["tokenizer"]
    if tokenizer is None or not hasattr(tokenizer, "Load"):
        raise RuntimeError("Tokenizer not found in ASR hparams.")

    # Already loaded?
    if getattr(tokenizer, "vocab_size", lambda: 0)() > 0:
        return tokenizer

    # Search candidates
    candidates = []
    if hparams.get("tokenizer_ckpt"):
        candidates.append(hparams["tokenizer_ckpt"])
    ckpt_dir = Path(hparams["asr_checkpoint"])
    candidates.append(str(ckpt_dir.parent.parent / "pretrained" / "tokenizer.ckpt"))
    pretrain_folder = getattr(asr_hparams.get("pretrainer", None), "collect_in", None)
    if pretrain_folder:
        candidates.append(str(Path(pretrain_folder) / "tokenizer.ckpt"))

    for cand in candidates:
        if os.path.exists(cand):
            print(f"  Loading tokenizer from: {cand}")
            tokenizer.Load(cand)
            return tokenizer

    raise FileNotFoundError(
        "tokenizer.ckpt not found. Pass tokenizer_ckpt in the YAML or --tokenizer_ckpt.\n"
        f"Searched: {candidates}"
    )


def parse_args():
    p = argparse.ArgumentParser(description="Create BoundaryClassifier dataset from LibriSpeech.")
    p.add_argument("hparams_file",
                   help="Path to boundary_classifier.yaml")
    p.add_argument("--max_files_per_split", type=int, default=None,
                   help="Limit files per split (for quick testing).")
    p.add_argument("--data_folder", type=str, default=None,
                   help="LibriSpeech root; overrides the yaml's data_folder.")
    p.add_argument("--tokenizer_ckpt", type=str, default=None,
                   help="Override tokenizer.ckpt path.")
    return p.parse_args()


def main():
    args    = parse_args()
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load config ───────────────────────────────────────────────────────
    with open(args.hparams_file, encoding="utf-8") as f:
        hparams = load_hyperpyyaml(f)

    if args.tokenizer_ckpt:
        hparams["tokenizer_ckpt"] = args.tokenizer_ckpt
    if args.data_folder:
        hparams["data_folder"] = args.data_folder

    # ── Load frozen ASR model ─────────────────────────────────────────────
    print("Loading frozen ASR model …")
    asr_hparams = load_asr_model(hparams, device)
    print("  ASR model loaded.")

    tokenizer = load_tokenizer(hparams, asr_hparams)
    print(f"  Tokenizer loaded. vocab_size={tokenizer.vocab_size()}")

    dynchunk_config = DynChunkTrainConfig(
        chunk_size=hparams["chunk_size"],
        left_context_size=hparams["left_context"],
    )
    print(
        f"  Streaming config: chunk_size={hparams['chunk_size']}, "
        f"left_context={hparams['left_context']}\n"
    )

    dataset_dir = Path(hparams["dataset_dir"])
    data_folder = Path(hparams["data_folder"])

    # ── Process each split ────────────────────────────────────────────────
    valid = hparams["valid_split"]
    if isinstance(valid, str):
        valid = [valid]
    splits_to_process = list(hparams["train_splits"]) + list(valid)

    for split_name in splits_to_process:
        split_dir = data_folder / split_name
        if not split_dir.exists():
            print(f"  [skip] Split dir not found: {split_dir}")
            continue

        wav_files = sorted(split_dir.glob("**/*.flac"))
        print(f"Processing '{split_name}'  ({len(wav_files)} files) …")

        out_path = dataset_dir / f"{split_name}.pt"
        process_split(
            split_name,
            wav_files,
            asr_hparams,
            tokenizer,
            device,
            dynchunk_config,
            out_path,
            max_files=args.max_files_per_split,
        )

    print("\nDataset creation complete.")


if __name__ == "__main__":
    main()
