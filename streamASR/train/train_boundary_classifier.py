#!/usr/bin/env python3
"""
train_boundary_classifier.py

Trains BoundaryClassifier on the (enc_feat, pred_feat, label) dataset
produced by data/create_boundary_dataset.py.

The classifier shares the same joint-network architecture as the ASR model's
Tjoint:  logits = Linear(joint_dim, 2)(GELU(enc_feat + pred_feat)).

Two output classes:
    0  –  non-boundary  (last emitted subword is part of an incomplete word)
    1  –  boundary      (last emitted subword ends a complete word)

Training uses weighted cross-entropy loss to handle the class imbalance
(word-boundary tokens are ~25-35 % of all chunk-boundary samples depending
on the BPE vocabulary).

Usage
-----
python train/train_boundary_classifier.py hparams/boundary_classifier.yaml
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm

from models.boundary_classifier import BoundaryClassifier


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_split_dataset(dataset_dir: Path, split_names):
    """
    Load and concatenate one or more split .pt files.

    Each file is a dict with keys:
        enc_feats  : Tensor [N, joint_dim]
        pred_feats : Tensor [N, joint_dim]
        labels     : Tensor [N]  (int64)

    Returns a TensorDataset.
    """
    all_enc   = []
    all_pred  = []
    all_labels = []

    if isinstance(split_names, str):
        split_names = [split_names]

    for name in split_names:
        pt_path = dataset_dir / f"{name}.pt"
        if not pt_path.exists():
            print(f"  [warn] Dataset file not found, skipping: {pt_path}")
            continue
        data = torch.load(pt_path, weights_only=True)
        all_enc.append(data["enc_feats"])
        all_pred.append(data["pred_feats"])
        all_labels.append(data["labels"])
        n = data["labels"].shape[0]
        n_pos = int(data["labels"].sum().item())
        print(
            f"  Loaded '{name}': {n} samples  "
            f"(boundary={n_pos}, non-boundary={n - n_pos})"
        )

    if not all_enc:
        raise FileNotFoundError(
            f"No dataset files found in {dataset_dir} for splits {split_names}. "
            "Run data/create_boundary_dataset.py first."
        )

    enc_feats  = torch.cat(all_enc,    dim=0)
    pred_feats = torch.cat(all_pred,   dim=0)
    labels     = torch.cat(all_labels, dim=0)
    return TensorDataset(enc_feats, pred_feats, labels)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    """Return (avg_loss, accuracy, precision, recall, f1)."""
    model.eval()
    total_loss = 0.0
    total      = 0
    tp = fp = fn = tn = 0

    for enc_f, pred_f, labs in loader:
        enc_f  = enc_f.to(device)
        pred_f = pred_f.to(device)
        labs   = labs.to(device)

        logits = model(enc_f, pred_f)          # [B, 2]
        loss   = loss_fn(logits, labs)
        total_loss += loss.item() * labs.size(0)
        total      += labs.size(0)

        preds = logits.argmax(-1)
        tp += int(((preds == 1) & (labs == 1)).sum())
        fp += int(((preds == 1) & (labs == 0)).sum())
        fn += int(((preds == 0) & (labs == 1)).sum())
        tn += int(((preds == 0) & (labs == 0)).sum())

    avg_loss  = total_loss / total if total > 0 else float("inf")
    accuracy  = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return avg_loss, accuracy, precision, recall, f1


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(hparams, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    dataset_dir = Path(hparams["dataset_dir"])

    # ── Datasets ──────────────────────────────────────────────────────────
    print("Loading datasets …")
    print("  Train splits:")
    train_dataset = load_split_dataset(dataset_dir, hparams["train_splits"])
    print("  Valid split:")
    valid_dataset = load_split_dataset(dataset_dir, hparams["valid_split"])
    print()

    train_loader = DataLoader(
        train_dataset,
        batch_size=hparams["batch_size"],
        shuffle=True,
        num_workers=hparams["num_workers"],
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=hparams["batch_size"] * 4,
        shuffle=False,
        num_workers=hparams["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = BoundaryClassifier(
        joint_dim=hparams["joint_dim"],
        hidden_dim=hparams.get("hidden_dim", 512),
        num_layers=hparams.get("num_layers", 3),
        dropout=hparams.get("dropout", 0.1),
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"BoundaryClassifier  —  {total_params:,} trainable parameters\n")

    # ── Loss ──────────────────────────────────────────────────────────────
    # Weighted cross-entropy: upweight the non-boundary class (label=0) since
    # boundary tokens are the majority class with BPE tokenization on
    # short-chunk streaming (many single-token words hit the chunk boundary).
    neg_weight   = float(hparams.get("neg_weight", 3.0))
    class_weight = torch.tensor([neg_weight, 1.0], device=device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weight)

    # Unweighted loss used only for reporting
    loss_fn_report = nn.CrossEntropyLoss()

    # ── Optimiser ─────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(hparams["lr"]),
        weight_decay=float(hparams["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # ── Checkpointing ─────────────────────────────────────────────────────
    ckpt_dir = Path(hparams["checkpoint_folder"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "best_model.pt"
    best_precision_ckpt_path = ckpt_dir / "best_precision_model.pt"
    last_ckpt_path = ckpt_dir / "last_model.pt"

    # Optionally resume from last checkpoint
    start_epoch  = 1
    best_val_loss = float("inf")
    best_val_precision = 0.0
    args.no_resume = True
    if last_ckpt_path.exists() and not args.no_resume:
        ckpt = torch.load(last_ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {ckpt['epoch']}  (best_val_loss={best_val_loss:.4f})\n")

    # ── Training log ──────────────────────────────────────────────────────
    log_path = Path(hparams["train_log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a", buffering=1)
    log_file.write("epoch,train_loss,val_loss,val_acc,val_precision,val_recall,val_f1,lr\n")

    num_epochs = int(hparams["number_of_epochs"])
    print(f"Training for {num_epochs} epochs …\n")

    for epoch in range(start_epoch, num_epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        n_train    = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{num_epochs}", leave=False)
        for enc_f, pred_f, labs in pbar:
            enc_f  = enc_f.to(device)
            pred_f = pred_f.to(device)
            labs   = labs.to(device)

            optimizer.zero_grad()
            logits = model(enc_f, pred_f)       # [B, 2]
            loss   = loss_fn(logits, labs)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labs.size(0)
            n_train    += labs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / n_train if n_train > 0 else float("inf")

        # ── Validate ──────────────────────────────────────────────────────
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(
            model, valid_loader, loss_fn_report, device
        )

        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{num_epochs}  "
            f"train_loss={avg_train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.4f}  "
            f"P={val_prec:.4f}  R={val_rec:.4f}  F1={val_f1:.4f}  "
            f"lr={lr_now:.2e}"
        )
        log_file.write(
            f"{epoch},{avg_train_loss:.6f},{val_loss:.6f},"
            f"{val_acc:.6f},{val_prec:.6f},{val_rec:.6f},{val_f1:.6f},{lr_now:.2e}\n"
        )

        # Save last checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "hparams": {
                    "joint_dim": hparams["joint_dim"],
                    "hidden_dim": hparams.get("hidden_dim", 512),
                    "num_layers": hparams.get("num_layers", 3),
                    "dropout":    hparams.get("dropout", 0.1),
                },
            },
            last_ckpt_path,
        )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "hparams": {
                        "joint_dim": hparams["joint_dim"],
                        "hidden_dim": hparams.get("hidden_dim", 512),
                        "num_layers": hparams.get("num_layers", 3),
                        "dropout":    hparams.get("dropout", 0.1),
                    },
                },
                best_ckpt_path,
            )
            print(f"  ✓ New best model saved  (val_loss={val_loss:.4f})")

        # Save best precision checkpoint
        if val_prec > best_val_precision:
            best_val_precision = val_prec
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_precision": val_prec,
                    "val_recall": val_rec,
                    "val_f1": val_f1,
                    "hparams": {
                        "joint_dim": hparams["joint_dim"],
                        "hidden_dim": hparams.get("hidden_dim", 512),
                        "num_layers": hparams.get("num_layers", 3),
                        "dropout":    hparams.get("dropout", 0.1),
                    },
                },
                best_precision_ckpt_path,
            )
            print(f"  ✓ New best precision model saved  (precision={val_prec:.4f})")

    log_file.close()
    print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}")
    print(f"Best val_precision={best_val_precision:.4f}")
    print(f"Best checkpoint : {best_ckpt_path}")
    print(f"Last checkpoint : {last_ckpt_path}")
    print(f"Training log    : {log_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Inference helper (standalone boundary prediction at inference time)
# ─────────────────────────────────────────────────────────────────────────────

def load_boundary_classifier(ckpt_path: str, device=None) -> BoundaryClassifier:
    """
    Load a trained BoundaryClassifier from a checkpoint produced by this script.

    Parameters
    ----------
    ckpt_path : str
        Path to best_model.pt or last_model.pt.
    device : torch.device, optional

    Returns
    -------
    model : BoundaryClassifier  (in eval mode, on device)

    Example usage during streaming inference
    ----------------------------------------
    ::
        classifier = load_boundary_classifier("…/best_model.pt", device)

        # At the end of each chunk, after greedy decode emits blank:
        enc_feat  = proj_enc(enc_out[t_end]).unsqueeze(0)   # [1, joint_dim]
        pred_feat = proj_dec(dec_out[u_last]).unsqueeze(0)  # [1, joint_dim]

        logits = classifier(enc_feat, pred_feat)            # [1, 2]
        is_boundary = logits.argmax(-1).item() == 1
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    hp = ckpt["hparams"]
    model = BoundaryClassifier(
        joint_dim=hp["joint_dim"],
        hidden_dim=hp.get("hidden_dim", 512),
        num_layers=hp.get("num_layers", 3),
        dropout=hp.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train BoundaryClassifier on pre-extracted chunk-boundary features."
    )
    p.add_argument("hparams_file",
                   help="Path to boundary_classifier.yaml")
    p.add_argument("--no_resume", action="store_true",
                   help="Start training from scratch even if a checkpoint exists.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(args.hparams_file, encoding="utf-8") as f:
        hparams = load_hyperpyyaml(f)

    Path(hparams["output_folder"]).mkdir(parents=True, exist_ok=True)

    train(hparams, args)
