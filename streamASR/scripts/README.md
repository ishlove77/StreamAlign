# StreamAlign · streamASR training pipeline

This directory holds the end-to-end scripts for building the StreamAlign speech
tokenizer, in the order they should be run. Every script locates the repo from
its own path (`BASH_SOURCE` / `__file__`), so it can be run from anywhere after
cloning; only the **data / checkpoint** paths near the top of each script are
machine-specific and may need editing.

Run everything from a conda env with the project deps (`conda activate streamASR`)
and, if `~/.local` shadows the conda PyTorch, prefix commands with
`PYTHONNOUSERSITE=1`.

## Pipeline order

### 1. Streaming ASR (RNN-Transducer)

| Script | What it trains |
| --- | --- |
| `run_train_char_asr.sh` | Character-level streaming RNN-T ASR. |
| `run_train_asr_word_fastemit.sh` | Word-level FastEmit RNN-T ASR that guides alignment. |

Key ASR config (`hparams/chunk_streaming_word_fastemit.yaml` + trainer):
- **Chunk size 160 ms** (`chunk_size: 4` frames, `left_context: 32` chunks).
- **FastEmit λ = 0.04** (`train/train_asr_word_fastemit.py`).
- Best streaming WER is reached around **epoch 20** — use that checkpoint.

### 2. Alignment data

| Script | Output |
| --- | --- |
| `generate_textgrids.sh` | Forced-alignment TextGrids used as alignment supervision. |
| `create_boundary_dataset.sh` | Word-boundary classifier dataset, built from the TextGrids. |

### 3. Proactive word-boundary classifier

| Script | Notes |
| --- | --- |
| `train_boundary_classifier.sh` | Trains the boundary classifier (cuts streaming latency 560 ms → 270 ms). |
| `run_boundary_pipeline.sh` | Convenience wrapper: dataset + classifier in one go. |

### 4. Precompute CosyVoice features (optional, recommended)

Caching the CosyVoice acoustic features/tokens makes tokenizer training much
faster (pass `--use_precomputed_features`).

- `precompute/launch_precompute_features*.sh` — acoustic features (LibriSpeech / Emilia).
- `precompute/run_precompute_tokens*.sh` — speech tokens.
- `precompute/verify_precomputed_features.py` — integrity check.

### 5. Tokenizer stage 1 — word distillation

| Script | What it trains |
| --- | --- |
| `train_word_distill.sh` | Initializes the subword acoustic embedding by distilling from the word ASR. |

### 6. Tokenizer stage 2 — RVQ pipeline

**R=32 is the tokenizer used for speech reconstruction** (the paper's
reconstruction results); **R=16 is used for spoken language modeling** and is
what the released weights contain.

Shared by both: the plain RVQ trainers (`train_tokenizer.py`, and
`train_tokenizer_cosine.py` for the decay phase) with the default EMA-updated
codebook, `RVQ_CODEBOOK_SIZE=512`, code dimension 256 (leave
`RVQ_CODEBOOK_DIM` unset so it defaults to `feat_dim`), commitment weight 1.0,
`--chunk_size=4 --left_context=32`, and the three-phase schedule at peak LRs
1e-4, 1e-4, 1e-5.

They differ only in quantizer depth and how phases are chained: R=32 uses
`--subalign_init_path` (fresh optimizer, epoch reset), R=16 uses
`--resume_path`.

#### 6a. R=32 — speech reconstruction (paper recipe)

`train_tokenizer_r32_pipeline.sh <stage>` covers the whole recipe, including the
alignment stage and the reconstruction evaluation:

```bash
# Stage 1 — char-level RNN-T aligner, then the boundary classifier
bash scripts/train_tokenizer_r32_pipeline.sh char_asr
bash scripts/train_tokenizer_r32_pipeline.sh boundary

# Stage 2 — continuous (quantizer bypassed) -> R=32 RVQ -> cosine decay
bash scripts/train_tokenizer_r32_pipeline.sh continuous
PHASE_CKPT=<continuous_ckpt>.pt bash scripts/train_tokenizer_r32_pipeline.sh r32
PHASE_CKPT=<r32_ckpt>.pt        bash scripts/train_tokenizer_r32_pipeline.sh cosine

# Streaming reconstruction on test-clean + whisper-large-v3 WER/CER
CKPT=<cosine_ckpt>.pt bash scripts/train_tokenizer_r32_pipeline.sh eval
```

`RVQ_R=32`. Phases B and C initialize their non-encoder weights from the
previous phase via `--subalign_init_path`, with a fresh optimizer.
Reference: WER 4.43% / CER 1.92% / UTMOS 4.23 / SECS 0.585 at cosine epoch 13
(paper: WER 4.41%); the cosine tail past ~13 epochs plateaus, so select there.

Note on resuming the cosine phase: the scheduler step is not checkpointed, so
set `LEARNING_RATE` to the LR at the interruption and `COSINE_TOTAL_STEPS` to
the *remaining* steps.

#### 6b. R=16 — spoken language modeling

`train_tokenizer_r16_pipeline.sh <stage>` runs the three-stage tokenizer training:

```bash
# Stage 1 — continuous representation (RVQ stop-grad passthrough, no hard codes)
bash scripts/train_tokenizer_r16_pipeline.sh continuous

# Stage 2 — R=16 RVQ enabled
RESUME_PATH=<stage1_ckpt>.pt bash scripts/train_tokenizer_r16_pipeline.sh r16

# Stage 3 — cosine LR decay fine-tune (peak -> 0)
RESUME_PATH=<stage2_ckpt>.pt bash scripts/train_tokenizer_r16_pipeline.sh cosine
```

`RVQ_R=16`, otherwise identical to the R=32 recipe. Phases are chained with
`--resume_path`.

### 7. Inference / evaluation

- `inference_stream.sh` — streaming reconstruction from StreamAlign units.
- `inference/run_inference_rvq_example.sh`, `inference/run_inference_subset.sh` — RVQ inference examples.
- Non-streaming variant: `inference/inference_nonstream.py`
  (`NONSTREAM_FULL_ATTN=1` for full bidirectional attention; boundary classifier
  is not used in non-streaming mode).

## Reference

Setup and notation follow the StreamAlign paper (streaming char/word RNN-T ASR →
speech-character & speech-subword alignment → character-level attention-pooling
aggregator → subword acoustic embedding `z_m` → RVQ with `R` codebook layers →
StreamAlign units, with a proactive boundary classifier and two-stage CosyVoice
flow+HiFT decoding for reconstruction).
