#!/usr/bin/env python3
"""
Recipe for training a Character-level Transducer ASR system with LibriTTS.

The system uses:
- Conformer encoder with Dynamic Chunk Training for streaming
- LSTM decoder
- Transducer loss + optional CTC
- Character-level tokens (no BPE/SentencePiece needed)
- CER evaluation


Authors
 * Based on SpeechBrain LibriTTS Transducer recipe
 * Modified for character-level prediction
"""

import os
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*Grid size.*will likely result in GPU under-utilization.*")

# Allow imports from streamASR root when run from train/ subdirectory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.distributed import if_main_process, run_on_main
from speechbrain.utils.logger import get_logger
from nemo.collections.asr.losses.rnnt import RNNTLoss

logger = get_logger(__name__)

from speechbrain.utils.fetching import fetch  

# FastEmit-regularized RNN-T loss (NeMo warprnnt_numba). lambda=0.04 gives the
# best streaming WER in our runs; best checkpoint is reached around epoch 20.
_rnnt_loss = RNNTLoss(num_classes=0, reduction='mean_batch', loss_name='warprnnt_numba', loss_kwargs={"fastemit_lambda": 0.04})


# ============================================================================
# ASR Brain
# ============================================================================
class ASR(sb.Brain):
    """Brain class for character-level Transducer ASR training."""

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
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
        fea_lens = wav_lens
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
        feats = self.modules.normalize(feats, fea_lens, epoch=current_epoch)

        # CNN frontend
        src = self.modules.CNN(feats)

        # Conformer encoder with optional chunk-based streaming
        x = self.modules.enc(
            src,
            fea_lens,
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

            T = logits_transducer.shape[1]
            act_lens = (wav_lens * T).round().to(torch.int64)
            label_lens = (token_lens * tokens.shape[1]).round().to(torch.int64)
            loss_transducer = _rnnt_loss(
                log_probs=logits_transducer,
                targets=tokens,
                input_lengths=act_lens, target_lengths=label_lens,
            )

            loss = (
                self.hparams.ctc_weight * CTC_loss
                + (1 - self.hparams.ctc_weight) * loss_transducer
            )
        else:
            T = logits_transducer.shape[1]
            act_lens = (wav_lens * T).round().to(torch.int64)
            label_lens = (token_lens * tokens.shape[1]).round().to(torch.int64)
            loss = _rnnt_loss(
                log_probs=logits_transducer,
                targets=tokens,
                input_lengths=act_lens, target_lengths=label_lens,
            )

        # Compute WER (validation/test)
        if stage != sb.Stage.TRAIN:
            predicted_words = [
                self.tokenizer.decode_ids(utt_seq).lower().split()
                for utt_seq in predicted_tokens
            ]
            target_words = [wrd.lower().split() for wrd in batch.wrd]

            # Compute WER
            self.wer_metric.append(ids, predicted_words, target_words)

        return loss

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------
    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        pass

    def on_stage_start(self, stage, epoch):
        """Initialize metrics at stage start."""
        if stage != sb.Stage.TRAIN:
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Log stats and save checkpoints at stage end."""
        stage_stats = {"loss": stage_loss}

        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        if stage == sb.Stage.VALID:
            epoch_stats = {
                "epoch": epoch,
                "lr": self.hparams.lr,
                "optimizer": self.optimizer.__class__.__name__,
            }

            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"], "epoch": epoch},
                min_keys=["WER"],
                num_to_keep=self.hparams.avg_checkpoints,
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

            if if_main_process():
                with open(
                    self.hparams.test_wer_file, "w", encoding="utf-8"
                ) as w:
                    self.wer_metric.write_stats(w)

            # Save averaged checkpoint
            self.checkpointer.save_and_keep_only(
                meta={"WER": -0.1, "epoch": epoch},
                min_keys=["WER"],
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

    # Load Emilia training data if configured
    if "emilia_train_csv" in hparams:
        emilia_csvs = hparams["emilia_train_csv"]
        if not isinstance(emilia_csvs, list):
            emilia_csvs = [emilia_csvs]
        emilia_replacements = {"data_root": hparams["emilia_data_folder"]}
        for csv in emilia_csvs:
            train_dict.update(load_data_csv(csv, emilia_replacements))

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

    # Text pipeline (LibriSpeech uses "wrd" column for transcription)
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        # Normalize to match LibriSpeech tokenizer format: uppercase, no punctuation
        wrd = re.sub(r"[^A-Z\s']", "", wrd.upper()).strip()
        yield wrd
        tokens_list = tokenizer.encode_as_ids(wrd)
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


    tokenizer = hparams["tokenizer"]
    logger.info(f"Character vocabulary size: {tokenizer.vocab_size}")

    # Prepare datasets
    (
        train_data,
        valid_data,
        test_datasets,
        train_bsampler,
        valid_bsampler,
    ) = dataio_prepare(hparams, tokenizer)

    hparams["pretrainer"].collect_files()
    hparams["pretrainer"].load_collected()

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
            min_key="WER",
        )
