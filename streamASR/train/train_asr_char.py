#!/usr/bin/env python3
"""
Recipe for training a Character-level Transducer ASR system with LibriTTS.

The system uses:
- Conformer encoder with Dynamic Chunk Training for streaming
- LSTM decoder
- Transducer loss + optional CTC
- Character-level tokens (no BPE/SentencePiece needed)
- CER evaluation

Usage:
    python train_asr.py <streamASR>/hparams/chunk_streaming_char.yaml

Authors
 * Based on SpeechBrain LibriSpeech Transducer recipe
 * Modified for character-level prediction
"""

import os
import sys
from pathlib import Path

# Allow imports from streamASR root when run from train/ subdirectory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.distributed import if_main_process, run_on_main
from speechbrain.utils.logger import get_logger
from models.model import CharacterTokenizer

logger = get_logger(__name__)

from speechbrain.utils.fetching import fetch  # 이걸로!

def load_pretrained_encoder_only(hparams, device, load_enc=True):
    src = hparams["pretrained_source"]
    savedir = Path(hparams["pretrained_save"])
    savedir.mkdir(parents=True, exist_ok=True)

    model_ckpt = fetch("model.ckpt", source=src, savedir=str(savedir))
    norm_ckpt  = fetch("normalizer.ckpt", source=src, savedir=str(savedir))

    # normalizer
    norm_state = torch.load(norm_ckpt, map_location="cpu")
    if isinstance(norm_state, dict) and "state_dict" in norm_state:
        norm_state = norm_state["state_dict"]
    hparams["normalize"].load_state_dict(norm_state, strict=False)

    # model
    ckpt = torch.load(model_ckpt, map_location="cpu")
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # ModuleList:
    # 0 CNN / 1 enc / 2 emb / 3 dec / 4 proj_enc / 5 proj_dec / 6 proj_ctc / 7 transducer_lin
    if load_enc:
        keep_prefixes = ("0.", "1.", "4.")   # CNN + enc + proj_enc
    else:
        keep_prefixes = ("0.", "4.")         # CNN + proj_enc 

    filtered = {k: v for k, v in sd.items() if k.startswith(keep_prefixes)}

    missing, unexpected = hparams["model"].load_state_dict(filtered, strict=False)
    print("[Pretrained] loaded:", len(filtered))
    print("[Pretrained] missing:", len(missing))
    print("[Pretrained] unexpected:", len(unexpected))
    

    hparams["model"].to(device)
    hparams["normalize"].to(device)


# ============================================================================
# ASR Brain
# ============================================================================
class ASR(sb.Brain):
    """Brain class for character-level Transducer ASR training."""
    
    def compute_forward(self, batch, stage):
        """Forward computations from waveform to output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_with_bos, token_with_bos_lens = batch.tokens_bos

        # Waveform augmentation (training only)
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "wav_augment"):
                wavs, wav_lens = self.hparams.wav_augment(wavs, wav_lens)
                tokens_with_bos = self.hparams.wav_augment.replicate_labels(
                    tokens_with_bos
                )

        # Feature extraction
        feats = self.hparams.compute_features(wavs)

        # Feature augmentation (training only)
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "fea_augment"):
            feats, fea_lens = self.hparams.fea_augment(feats, wav_lens)
            tokens_with_bos = self.hparams.fea_augment.replicate_labels(
                tokens_with_bos
            )

        current_epoch = self.hparams.epoch_counter.current

        # Dynamic Chunk Training configuration
        if hasattr(self.hparams, "streaming") and self.hparams.streaming:
            dynchunktrain_config = self.hparams.dynchunktrain_config_sampler(
                stage
            )
        else:
            dynchunktrain_config = None

        # Normalize features
        feats = self.modules.normalize(feats, wav_lens, epoch=current_epoch)

        # CNN frontend
        src = self.modules.CNN(feats)
        
        # Conformer encoder with optional chunk-based streaming
        x = self.modules.enc(
            src,
            wav_lens,
            pad_idx=self.hparams.pad_index,
            dynchunktrain_config=dynchunktrain_config,
        )
        x = self.modules.proj_enc(x)

        # Decoder (prediction network)
        e_in = self.modules.emb(tokens_with_bos)
        e_in = torch.nn.functional.dropout(
            e_in,
            self.hparams.dec_emb_dropout,
            training=(stage == sb.Stage.TRAIN),
        )
        h, _ = self.modules.dec(e_in)
        h = torch.nn.functional.dropout(
            h, self.hparams.dec_dropout, training=(stage == sb.Stage.TRAIN)
        )
        h = self.modules.proj_dec(h)

        # Joint network
        joint = self.modules.Tjoint(x.unsqueeze(2), h.unsqueeze(1))

        # Transducer output
        logits_transducer = self.modules.transducer_lin(joint)

        # Compute outputs based on stage
        if stage == sb.Stage.TRAIN:
            p_ctc = None

            if (
                self.hparams.ctc_weight > 0.0
                and current_epoch <= self.hparams.number_of_ctc_epochs
            ):
                out_ctc = self.modules.proj_ctc(x)
                p_ctc = self.hparams.log_softmax(out_ctc)

            return p_ctc, logits_transducer, wav_lens

        elif stage == sb.Stage.VALID:
            best_hyps, scores, _, _ = self.hparams.Greedysearcher(x)
            return logits_transducer, wav_lens, best_hyps
        else:
            # Test: use beam search
            best_hyps, best_scores, nbest_hyps, nbest_scores = self.hparams.Beamsearcher(x)
            return logits_transducer, wav_lens, best_hyps

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (Transducer + CTC) given predictions and targets."""
        ids = batch.id
        tokens, token_lens = batch.tokens
        tokens_eos, token_eos_lens = batch.tokens_eos

        if stage == sb.Stage.TRAIN:
            p_ctc, logits_transducer, wav_lens = predictions
        else:
            logits_transducer, wav_lens, predicted_tokens = predictions

        # Handle augmentation label replication
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "fea_augment"):
                (
                    tokens,
                    token_lens,
                    tokens_eos,
                    token_eos_lens,
                ) = self.hparams.fea_augment.replicate_multiple_labels(
                    tokens, token_lens, tokens_eos, token_eos_lens
                )

        # Compute loss
        if stage == sb.Stage.TRAIN:
            CTC_loss = 0.0
            if p_ctc is not None:
                CTC_loss = self.hparams.ctc_cost(
                    p_ctc, tokens, wav_lens, token_lens
                )
            
            loss_transducer = self.hparams.transducer_cost(
                logits_transducer, tokens, wav_lens, token_lens
            )
            
            loss = (
                self.hparams.ctc_weight * CTC_loss
                + (1 - self.hparams.ctc_weight) * loss_transducer
            )
        else:
            loss = self.hparams.transducer_cost(
                logits_transducer, tokens, wav_lens, token_lens
            )

        # Compute CER (validation/test)
        if stage != sb.Stage.TRAIN:
            # Decode predictions to characters
            predicted_chars = [
                list(self.tokenizer.decode(utt_seq))
                for utt_seq in predicted_tokens
            ]
            
            # Reference characters
            target_chars = [list(lbl.lower()) for lbl in batch.wrd]
            
            # Compute CER
            self.cer_metric.append(ids, predicted_chars, target_chars)

        return loss

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """Apply Noam annealing at the end of optimizer step."""
        if should_step:
            self.hparams.noam_annealing(self.optimizer)

    def on_stage_start(self, stage, epoch):
        """Initialize metrics at stage start."""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Log stats and save checkpoints at stage end."""
        stage_stats = {"loss": stage_loss}
        
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")

        if stage == sb.Stage.VALID:
            lr = self.hparams.noam_annealing.current_lr
            steps = self.optimizer_step
            optimizer = self.optimizer.__class__.__name__

            epoch_stats = {
                "epoch": epoch,
                "lr": lr,
                "steps": steps,
                "optimizer": optimizer,
            }

            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            
            self.checkpointer.save_and_keep_only(
                meta={"CER": stage_stats["CER"], "epoch": epoch},
                min_keys=["CER"],
                num_to_keep=self.hparams.avg_checkpoints,
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            
            if if_main_process():
                with open(
                    self.hparams.test_cer_file, "w", encoding="utf-8"
                ) as w:
                    self.cer_metric.write_stats(w)

            # Save averaged checkpoint
            self.checkpointer.save_and_keep_only(
                meta={"CER": -0.1, "epoch": epoch},
                min_keys=["CER"],
                num_to_keep=1,
            )

    def on_evaluate_start(self, max_key=None, min_key=None):
        """Perform checkpoint averaging if needed."""
        super().on_evaluate_start()

        ckpts = self.checkpointer.find_checkpoints(
            max_key=max_key,
            min_key=min_key,
        )
        ckpt = sb.utils.checkpoints.average_checkpoints(
            ckpts, recoverable_name="model"
        )

        self.hparams.model.load_state_dict(ckpt, strict=True)
        self.hparams.model.eval()


# ============================================================================
# Data Preparation
# ============================================================================
def dataio_prepare(hparams, tokenizer):
    """Prepare datasets with LibriSpeech CSV loading."""
    from speechbrain.dataio.dataio import load_data_csv

    data_folder = hparams["data_folder"]
    replacements = {"data_root": data_folder}

    # Merge all train CSVs into a single DynamicItemDataset
    train_csvs = hparams["train_csv"]
    if not isinstance(train_csvs, list):
        train_csvs = [train_csvs]
    train_dict = {}
    for csv in train_csvs:
        train_dict.update(load_data_csv(csv, replacements))
    train_data = sb.dataio.dataset.DynamicItemDataset(train_dict)

    if hparams["sorting"] == "ascending":
        train_data = train_data.filtered_sorted(sort_key="duration")
        hparams["train_dataloader_opts"]["shuffle"] = False
    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        hparams["train_dataloader_opts"]["shuffle"] = False
    elif hparams["sorting"] == "random":
        pass
    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements=replacements,
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    # Test: may be a list of CSV paths
    test_csvs = hparams["test_csv"]
    if not isinstance(test_csvs, list):
        test_csvs = [test_csvs]
    missing = [c for c in test_csvs if not os.path.exists(c)]
    if missing:
        print(f"[data] skipping missing test csvs: {missing}")
        test_csvs = [c for c in test_csvs if os.path.exists(c)]
    test_datasets = {}
    for csv_path in test_csvs:
        name = Path(csv_path).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_path, replacements=replacements
        )
        test_datasets[name] = test_datasets[name].filtered_sorted(
            sort_key="duration"
        )

    datasets = [train_data, valid_data] + list(test_datasets.values())

    # Audio pipeline
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # Text pipeline (character-level, uppercase encoding)
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.encode(wrd.upper())
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + tokens_list)
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # Set output keys
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens"],
    )

    # Dynamic batching
    train_batch_sampler = None
    valid_batch_sampler = None

    if hparams["dynamic_batching"]:
        from speechbrain.dataio.sampler import DynamicBatchSampler

        dynamic_hparams = hparams["dynamic_batch_sampler"]
        num_buckets = dynamic_hparams["num_buckets"]

        train_batch_sampler = DynamicBatchSampler(
            train_data,
            dynamic_hparams["max_batch_len"],
            num_buckets=num_buckets,
            length_func=lambda x: x["duration"],
            shuffle=dynamic_hparams["shuffle_ex"],
            batch_ordering=dynamic_hparams["batch_ordering"],
        )

        valid_batch_sampler = DynamicBatchSampler(
            valid_data,
            dynamic_hparams["max_batch_len_val"],
            num_buckets=num_buckets,
            length_func=lambda x: x["duration"],
            shuffle=dynamic_hparams["shuffle_ex"],
            batch_ordering=dynamic_hparams["batch_ordering"],
        )

    return (
        train_data,
        valid_data,
        test_datasets,
        train_batch_sampler,
        valid_batch_sampler,
    )


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    # Parse arguments
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # CPU fallback for torchaudio
    device = getattr(run_opts, "device", "cuda")
    if device == "cpu":
        if "use_torchaudio: True" in overrides:
            overrides = overrides.replace("use_torchaudio: True", "use_torchaudio: False")
        else:
            overrides += "\nuse_torchaudio: True"

    # DDP init
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Create character tokenizer
    tokenizer = CharacterTokenizer()
    logger.info(f"Character vocabulary size: {tokenizer.vocab_size}")

    # Prepare datasets
    (
        train_data,
        valid_data,
        test_datasets,
        train_bsampler,
        valid_bsampler,
    ) = dataio_prepare(hparams, tokenizer)

    # Initialize ASR brain
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Attach tokenizer
    asr_brain.tokenizer = tokenizer

    # Load pretrained encoder weights
    load_pretrained_encoder_only(hparams, asr_brain.device, load_enc=True)

    # Dataloader options
    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]

    if train_bsampler is not None:
        train_dataloader_opts = {
            "batch_sampler": train_bsampler,
            "num_workers": hparams["num_workers"],
        }
    if valid_bsampler is not None:
        valid_dataloader_opts = {"batch_sampler": valid_bsampler}

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=train_dataloader_opts,
        valid_loader_kwargs=valid_dataloader_opts,
    )

    # Testing
    os.makedirs(hparams["output_wer_folder"], exist_ok=True)
    for k in test_datasets.keys():
        asr_brain.hparams.test_wer_file = os.path.join(
            hparams["output_wer_folder"], f"wer_{k}.txt"
        )
        asr_brain.evaluate(
            test_datasets[k],
            test_loader_kwargs=hparams["test_dataloader_opts"],
            min_key="CER",
        )
