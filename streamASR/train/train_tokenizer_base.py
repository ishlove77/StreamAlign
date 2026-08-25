#!/usr/bin/env python3

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import warnings
import random

# Suppress torchaudio deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", message=".*StreamingMediaDecoder.*")
warnings.filterwarnings("ignore", message=".*TorchCodec.*")

from tqdm import tqdm
import sys
from hyperpyyaml import load_hyperpyyaml
from transformers import AutoTokenizer

# Allow imports from streamASR root when run from train/ subdirectory
_STREAMASR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STREAMASR_ROOT)

matcha_path = os.path.join(_STREAMASR_ROOT, "third_party", "Matcha-TTS")
if matcha_path not in sys.path:
    sys.path.append(matcha_path)

from utils.data_utils_cosyvoice import (
    NoiseAugmentor, LibriTTSDataset, EmiliaTextGridDataset,
    unified_collate_fn, load_emilia_wavpaths,
)
from utils.train_utils_cosyvoice import (
    is_main_process,
    load_filtered_state,
    load_optimizer_state_flexible,
    load_subalign_overrides,
    compute_losses,
    preprocess_batch,
    process_batch,
)
from utils.visualize import visualize_validation_batch
from models.model_tokenizer import Data2VecSemanticAcousticModel

sys.path.insert(0, os.environ.get(
    "COSYVOICE_ROOT", os.path.join(_STREAMASR_ROOT, "third_party", "CosyVoice")
))
from cosyvoice.cli.frontend import CosyVoiceFrontEnd


###############################################################################
# Argument parsing
###############################################################################

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train HuBERT->Character model with word-level segmentation."
    )
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--resume_path", type=str, default=None, help="Path to checkpoint to resume from (optional).")
    parser.add_argument("--subalign_init_path", type=str, default=None,
                        help="Init non-encoder weights from a prior-phase epoch_N.pt "
                             "(fresh optimizer, epoch counter reset; encoder.* excluded). "
                             "Overridden by --resume_path if both are set.")
    parser.add_argument("--visualize", type=bool, default=True)
    parser.add_argument("--mask_prob", type=float, default=0.08)
    parser.add_argument("--train_sample_ratio", type=float, default=0.2, help="Fraction of training data to use per epoch (0.0-1.0).")
    parser.add_argument("--chunk_size", type=int, default=4)
    parser.add_argument("--left_context", type=int, default=32)
    parser.add_argument(
        "--hparams", type=str,
        default=os.path.join(_STREAMASR_ROOT, "hparams", "chunk_streaming_libritts_distill.yaml"),
    )
    parser.add_argument(
        "--checkpoint_path", type=str,
        default=os.path.join(_STREAMASR_ROOT, "results", "char_asr_ckpt"),
    )
    parser.add_argument(
        "--emilia_csv", type=str, default=None,
        help="Path to Emilia CSV manifest. If None, train on LibriTTS only.",
    )
    parser.add_argument(
        "--emilia_data_root", type=str,
        default=os.environ.get("EMILIA_ROOT", "/data/Emilia"),
        help="Root directory for Emilia audio (replaces $data_root in CSV).",
    )
    parser.add_argument(
        "--emilia_sample_ratio", type=float, default=0.1,
        help="Fraction of Emilia to sample each epoch (independent of --train_sample_ratio).",
    )
    parser.add_argument(
        "--freeze_except_quantizer", action="store_true",
        help="Freeze all parameters except the residual_vq quantizer.",
    )
    parser.add_argument(
        "--use_precomputed_features", action="store_true",
        help="Load CosyVoice3 speech tokens and speaker embeddings from per-wav "
             ".speech_tokens.pt / .spk_emb.pt cache files instead of extracting "
             "them online. Skips frontend instantiation.",
    )
    parser.add_argument(
        "--data_root", type=str,
        default=os.environ.get("LIBRITTS_ROOT", os.environ.get("LIBRI_DATA_ROOT", "/data/LibriTTS")),
        help="Root containing the train/val split directories (e.g. LibriTTS or LibriSpeech mirror).",
    )
    parser.add_argument(
        "--data_ext", type=str, default=os.environ.get("LIBRI_DATA_EXT", "wav"),
        help="Audio file extension to glob (e.g. wav for LibriTTS, flac for raw LibriSpeech).",
    )
    parser.add_argument(
        "--train_splits", type=str,
        default="train-clean-100,train-clean-360,train-other-500",
        help="Comma-separated list of split subdirectories to glob for training.",
    )
    parser.add_argument(
        "--val_splits", type=str, default="dev-clean",
        help="Comma-separated list of split subdirectories for validation.",
    )
    parser.add_argument(
        "--skip_validation", action="store_true",
        help="Disable validation loader and per-epoch validation loop.",
    )
    return parser.parse_args()


###############################################################################
# Distributed setup
###############################################################################

def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}"), local_rank


###############################################################################
# Training loop
###############################################################################

def train_regression_model(
    train_wavpaths,
    val_wavpaths,
    emilia_wavpaths=None,
    num_epochs=20,
    batch_size=8,
    learning_rate=1e-4 * 0.2,
    weight_decay=1e-5,
    log_dir="./logs/Hubert_Character_Model",
    local_rank=None,
    resume_path=None,
    args=None,
    device=None,
):
    
    writer = SummaryWriter(log_dir=log_dir) if is_main_process() else None

    model_dir = os.environ.get(
        "COSYVOICE_MODEL_DIR",
        os.path.join(
            os.environ.get("COSYVOICE_ROOT", os.path.join(_STREAMASR_ROOT, "third_party", "CosyVoice")),
            "pretrained_models", "Fun-CosyVoice3-0.5B",
        ),
    )

    noise_aug = NoiseAugmentor(
        os.environ.get("NOISE_ROOT", "/data/noise")
    )

    if args.use_precomputed_features:
        frontend = None
        if is_main_process():
            print("[use_precomputed_features=True] Skipping CosyVoice frontend init; "
                  "loading speech_tokens/spk_emb from .pt cache.")
    else:
        with open(f"{model_dir}/cosyvoice3.yaml", "r") as f:
            configs = load_hyperpyyaml(f)
        frontend = CosyVoiceFrontEnd(
            get_tokenizer=configs["get_tokenizer"],
            feat_extractor=configs["feat_extractor"],
            campplus_model=f"{model_dir}/campplus.onnx",
            speech_tokenizer_model=f"{model_dir}/speech_tokenizer_v3.onnx",
            allowed_special=configs["allowed_special"],
        )

    glob_tok = AutoTokenizer.from_pretrained(
        os.environ.get("LLAMA_TOKENIZER_PATH", "meta-llama/Llama-3.1-8B")
    )
    glob_tok.pad_token = glob_tok.eos_token


    # validation dataset and loader (skipped when --skip_validation)
    if args.skip_validation or not val_wavpaths:
        val_loader = None
        if is_main_process():
            print("[skip_validation] No validation loader will be built.")
    else:
        val_dataset = LibriTTSDataset(
            val_wavpaths, use_precomputed_features=args.use_precomputed_features
        )
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=8,
            collate_fn=unified_collate_fn,
            pin_memory=True,
            drop_last=True
        )

    # Model, optimizer, and (optionally) checkpoint loading
    hubert_model = Data2VecSemanticAcousticModel(
        chunk_size=args.chunk_size,
        left_context=args.left_context,
        hparams_file=args.hparams,
        checkpoint_path=args.checkpoint_path,
    ).to(device)

    if getattr(args, "subalign_init_path", None):
        load_subalign_overrides(
            hubert_model, args.subalign_init_path,
            exclude_prefixes=("encoder.",),
            name="subalign-init",
        )

    if args.freeze_except_quantizer:
        for name, param in hubert_model.named_parameters():
            if "residual_vq" not in name:
                param.requires_grad = False
        if is_main_process():
            n_train = sum(p.numel() for p in hubert_model.parameters() if p.requires_grad)
            print(f"[freeze_except_quantizer] Trainable parameters: {n_train}")

    hubert_optimizer = torch.optim.AdamW(
        hubert_model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    start_epoch = 0
    if resume_path and os.path.isfile(resume_path):
        if is_main_process():
            print(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        load_filtered_state(hubert_model, checkpoint["hubert_state_dict"], "hubert_model")
        load_optimizer_state_flexible(
            hubert_optimizer, checkpoint["optimizer_state_dict"], device,
            model=hubert_model, saved_model_state=checkpoint["hubert_state_dict"],
        )
        start_epoch = checkpoint["epoch"] + 1

    hubert_model = nn.parallel.DistributedDataParallel(
        hubert_model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )

    global_step = 0
    os.makedirs(f"checkpoints/{args.exp_name}", exist_ok=True)

    for epoch in range(start_epoch, num_epochs):
        # ── Training ──────────────────────────────────────────────────────────
        hubert_model.train()
        total_train_loss = train_loss_u = train_loss_vq = 0.0
        total_train_steps = 0

        # Sample training data
        num_samples = int(len(train_wavpaths) * args.train_sample_ratio)
        sampled_indices = random.sample(range(len(train_wavpaths)), num_samples)
        sampled_wavpaths = [train_wavpaths[i] for i in sampled_indices]
        libri_ds = LibriTTSDataset(
            sampled_wavpaths, use_precomputed_features=args.use_precomputed_features
        )

        if emilia_wavpaths:
            num_emilia = int(len(emilia_wavpaths) * args.emilia_sample_ratio)
            sampled_emilia = [emilia_wavpaths[i] for i in random.sample(range(len(emilia_wavpaths)), num_emilia)]
            emilia_ds = EmiliaTextGridDataset(
                sampled_emilia, use_precomputed_features=args.use_precomputed_features
            )
            train_ds = ConcatDataset([libri_ds, emilia_ds])
        else:
            train_ds = libri_ds

        train_sampler = DistributedSampler(train_ds, shuffle=True)
        train_sampler.set_epoch(epoch)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=8,
            collate_fn=unified_collate_fn,
            drop_last=True,
            pin_memory=True,
        )

        train_iter = (
            tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
            if is_main_process() else train_loader
        )

        for batch_data in train_iter:
            if batch_data is None:
                continue

            preprocessed = preprocess_batch(
                batch_data, frontend, glob_tok, device
            )
            #detach = global_step < 0
            model_out = process_batch(
                hubert_model, preprocessed, device,
                detach=True, noise_aug=noise_aug,
                tokenizer=glob_tok
            )
            if model_out is None:
                continue

            losses = compute_losses(
                preprocessed, hubert_model, model_out,
                model_out["char_alignment"], model_out["word_alignment"], device,
            )

            hubert_optimizer.zero_grad()
            losses["total_loss"].backward()
            hubert_optimizer.step()

            if writer:
                writer.add_scalar("Train/Total_loss_step", losses["total_loss"].item(), global_step)
                writer.add_scalar("Train/Loss_u_step", losses["loss_u"].item(), global_step)
                writer.add_scalar("Train/Loss_vq_step", losses["loss_vq"].item(), global_step)

            global_step += 1
            total_train_loss += losses["total_loss"].item()
            train_loss_u += losses["loss_u"].item()
            train_loss_vq += losses["loss_vq"].item()
            total_train_steps += 1

        avg_train_loss = total_train_loss / max(1, total_train_steps)
        avg_train_loss_u = train_loss_u / max(1, total_train_steps)
        avg_train_loss_vq = train_loss_vq / max(1, total_train_steps)

        if writer:
            writer.add_scalar("Train/Total_loss_epoch", avg_train_loss, epoch)
            writer.add_scalar("Train/Loss_u_epoch", avg_train_loss_u, epoch)
            writer.add_scalar("Train/Loss_vq_epoch", avg_train_loss_vq, epoch)

        # ── Validation ────────────────────────────────────────────────────────
        avg_val_loss = avg_val_u = avg_val_vq = 0.0
        if val_loader is None:
            if is_main_process():
                print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} (validation skipped)")
                print(f"  Train - U: {avg_train_loss_u:.4f}, VQ: {avg_train_loss_vq:.4f}")
                checkpoint_path = f"checkpoints/{args.exp_name}/epoch_{epoch+1}.pt"
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                hubert_state_dict = (
                    hubert_model.module.state_dict()
                    if isinstance(hubert_model, nn.parallel.DistributedDataParallel)
                    else hubert_model.state_dict()
                )
                torch.save({
                    "epoch": epoch,
                    "hubert_state_dict": hubert_state_dict,
                    "optimizer_state_dict": hubert_optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                }, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
            continue

        hubert_model.eval()
        total_val_loss = total_val_u = total_val_vq = 0.0
        total_val_steps = 0

        with torch.no_grad():
            val_iter = (
                tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
                if is_main_process() else val_loader
            )

            for val_batch_idx, batch_data in enumerate(val_iter):
                if batch_data is None:
                    continue
                try:
                    preprocessed = preprocess_batch(
                        batch_data, frontend, glob_tok, device
                    )
                    model_out = process_batch(
                        hubert_model, preprocessed, device,
                        tokenizer=glob_tok
                    )
                    if model_out is None:
                        continue

                    val_losses = compute_losses(
                        preprocessed, hubert_model, model_out,
                        model_out["char_alignment"], model_out["word_alignment"], device,
                    )
                    total_val_loss += val_losses["total_loss"].item()
                    total_val_u += val_losses["loss_u"].item()
                    total_val_vq += val_losses["loss_vq"].item()
                    total_val_steps += 1

                    if val_batch_idx == 0 and writer and args.visualize and is_main_process():
                        try:
                            visualize_validation_batch(
                                writer=writer,
                                global_step=global_step,
                                batch_idx=0,
                                model_out=model_out,
                                preprocessed_batch=preprocessed,
                                char_alignment=model_out["char_alignment"],
                                word_alignment=model_out["word_alignment"],
                                waveform=preprocessed["waveforms"],
                            )

                            writer.add_scalar("Validation/Total_loss", val_losses["total_loss"].item(), global_step)
                            writer.add_scalar("Validation/Loss_u", val_losses["loss_u"].item(), global_step)
                            writer.add_scalar("Validation/Loss_vq", val_losses["loss_vq"].item(), global_step)

                            if preprocessed.get("char_data"):
                                cd_item = preprocessed["char_data"][0]
                                if "char_sequence" in cd_item:
                                    writer.add_text(
                                        "Validation/Sample_Text",
                                        "".join(cd_item["char_sequence"][:100]),
                                        global_step,
                                    )
                                if "tokens" in cd_item:
                                    writer.add_text(
                                        "Validation/Sample_Tokens",
                                        " ".join(cd_item["tokens"][:20]),
                                        global_step,
                                    )

                                char_align = model_out["char_alignment"][0]
                                word_align = model_out["word_alignment"][0]
                                writer.add_scalar("Validation/Char_Alignment_Density",
                                                  char_align.float().mean().item(), global_step)
                                writer.add_scalar("Validation/Char_Alignment_Coverage",
                                                  (char_align.sum(dim=1) > 0).float().mean().item(), global_step)
                                writer.add_scalar("Validation/Word_Alignment_Density",
                                                  word_align.float().mean().item(), global_step)
                                writer.add_scalar("Validation/Word_Alignment_Coverage",
                                                  (word_align.sum(dim=1) > 0).float().mean().item(), global_step)

                                if "d_pred" in model_out["forward_output"]:
                                    d_pred = model_out["forward_output"]["d_pred"][0]
                                    d_gt = char_align.sum(dim=1).float()
                                    writer.add_scalar("Validation/Duration_Mean_Pred", d_pred.mean().item(), global_step)
                                    writer.add_scalar("Validation/Duration_Mean_GT", d_gt.mean().item(), global_step)
                                    if len(d_pred) > 1 and len(d_gt) > 1:
                                        valid_mask = (d_gt > 0) & (d_pred > 0)
                                        if valid_mask.sum() > 1:
                                            corr = torch.corrcoef(
                                                torch.stack([d_pred[valid_mask], d_gt[valid_mask]])
                                            )[0, 1]
                                            if not torch.isnan(corr):
                                                writer.add_scalar("Validation/Duration_Correlation",
                                                                  corr.item(), global_step)

                            print(f"Validation visualization completed for epoch {epoch+1}")
                        except Exception as e:
                            print(f"Validation visualization failed: {e}")
                            import traceback
                            traceback.print_exc()

                except Exception as e:
                    print(f"Validation batch {val_batch_idx} failed: {e}")
                    continue

        avg_val_loss = total_val_loss / max(1, total_val_steps)
        avg_val_u = total_val_u / max(1, total_val_steps)
        avg_val_vq = total_val_vq / max(1, total_val_steps)

        if writer and is_main_process():
            writer.add_scalar("Validation/Avg_Total_loss_epoch", avg_val_loss, epoch)
            writer.add_scalar("Validation/Avg_Loss_u_epoch", avg_val_u, epoch)
            writer.add_scalar("Validation/Avg_Loss_vq_epoch", avg_val_vq, epoch)

        if is_main_process():
            print(f"[Epoch {epoch+1}/{num_epochs}] "
                  f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            print(f"  Train - U: {avg_train_loss_u:.4f}, VQ: {avg_train_loss_vq:.4f}")
            print(f"  Val   - U: {avg_val_u:.4f}, VQ: {avg_val_vq:.4f}")

            checkpoint_path = f"checkpoints/{args.exp_name}/epoch_{epoch+1}.pt"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            hubert_state_dict = (
                hubert_model.module.state_dict()
                if isinstance(hubert_model, nn.parallel.DistributedDataParallel)
                else hubert_model.state_dict()
            )
            torch.save({
                "epoch": epoch,
                "hubert_state_dict": hubert_state_dict,
                "optimizer_state_dict": hubert_optimizer.state_dict(),
                "train_loss": avg_train_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    if writer:
        writer.close()
    dist.destroy_process_group()
    print("Training complete!")
    return hubert_model


###############################################################################
# Entry point
###############################################################################

def main():
    args = parse_args()
    device, local_rank = (
        setup_distributed() if torch.cuda.is_available()
        else (torch.device("cpu"), 0)
    )
    torch.backends.cudnn.benchmark = True
    train_wavpaths, val_wavpaths = [], []

    import glob
    train_splits = [s.strip() for s in args.train_splits.split(",") if s.strip()]
    for dataset in train_splits:
        pattern = f"{args.data_root}/{dataset}/*/*/*.{args.data_ext}"
        train_wavpaths.extend(glob.glob(pattern))

    if not args.skip_validation:
        val_splits = [s.strip() for s in args.val_splits.split(",") if s.strip()]
        for dataset in val_splits:
            pattern = f"{args.data_root}/{dataset}/*/*/*.{args.data_ext}"
            val_wavpaths.extend(glob.glob(pattern))

    emilia_wavpaths = []
    if args.emilia_csv:
        emilia_wavpaths = load_emilia_wavpaths(args.emilia_csv, args.emilia_data_root)

    if local_rank == 0:
        print(f"Training data: {len(train_wavpaths)} LibriTTS, "
              f"{len(emilia_wavpaths)} Emilia, "
              f"{len(val_wavpaths)} validation (LibriTTS dev-clean)")

    hubert_model = train_regression_model(
        train_wavpaths=train_wavpaths,
        val_wavpaths=val_wavpaths,
        emilia_wavpaths=emilia_wavpaths if emilia_wavpaths else None,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        log_dir="logs/" + args.exp_name,
        local_rank=local_rank,
        resume_path=args.resume_path,
        args=args,
        device=device,
    )

    final_path = f"checkpoints/{args.exp_name}/final.pt"
    hubert_state_dict = (
        hubert_model.module.state_dict()
        if isinstance(hubert_model, nn.parallel.DistributedDataParallel)
        else hubert_model.state_dict()
    )
    torch.save(hubert_state_dict, final_path)
    print(f"Final model saved to {final_path}")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
