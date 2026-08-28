# Official Implementation of StreamAlign: Streaming Text-Aligned Speech Tokenization

[Kang-wook Kim](https://kwkim.me/)<sup>&ast;</sup>,
[Jinyoung Park](https://www.linkedin.com/in/jinyoung-park-3841b4384/)<sup>&ast;</sup>,
[Jinsoo Kim](https://www.linkedin.com/in/%EC%A7%84%EC%88%98-%EA%B9%80-b769832a1/),
[Sehun Lee](https://yhytoto12.github.io),
[Tony Woo](https://tonywoo.me/),
[Gunhee Kim](https://vision.snu.ac.kr/gunhee/)

<sup>&ast;</sup>Equal contribution

[![Venue](https://img.shields.io/badge/EMNLP_2026-Findings-b31b1b.svg)](https://2026.emnlp.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![R32](https://img.shields.io/badge/🤗-Streamalign--R32-yellow.svg)](https://huggingface.co/dd3434/Streamalign-R32)
[![Tokenizer](https://img.shields.io/badge/🤗-Streamalign--R16-yellow.svg)](https://huggingface.co/dd3434/Streamalign-R16)
[![SLM](https://img.shields.io/badge/🤗-Streamalign--SLM--R16-yellow.svg)](https://huggingface.co/dd3434/Streamalign-SLM-R16)

Accepted to **Findings of EMNLP 2026**.

This repository contains the **full StreamAlign stack**: the streaming speech
tokenizer that aligns acoustic units to text as audio arrives (`streamASR`),
and the speech language model trained on those units (`streamSLM`).

Speech tokenizers usually emit units on a fixed frame grid, which leaves the
units unaligned with the text a language model consumes. StreamAlign instead
aggregates encoder frames into **one unit per subword**, streaming, using a
character-level RNN-Transducer ASR for alignment and a proactive word-boundary
classifier to commit words early. Each subword `m` becomes a tuple
`(w_m, q_m, d_m)`: the subword id, `R` residual-VQ codes, and a duration in
encoder frames.

**The number of residual stages `R` differs by task.** Following the paper, the
tokenizer used for **speech reconstruction is R=32**, while spoken language
modeling uses **R=16**. Both share the same alignment stage, ASR models, and
boundary classifier; only the quantizer depth differs, so each has its own
training pipeline below.

## Repository layout

| Directory | What it holds |
|---|---|
| [`streamASR/`](streamASR) | The tokenizer: streaming ASR, alignment, boundary classifier, RVQ quantizer, and CosyVoice-based resynthesis. |
| [`streamSLM/`](streamSLM) | The speech LM: unit extraction, training, generation, and evaluation (SALMon, StoryCloze, continuation). |
| [`examples/`](examples) | Runnable reconstruction and continuation demos against the released weights. |
| [`docs/`](docs) | The demo page, published at [ishlove77.github.io/StreamAlign](https://ishlove77.github.io/StreamAlign/): side-by-side audio for reconstruction and continuation. |

## Pipeline overview

Training runs in two stages. **Stage 1 learns the alignment**: the streaming
ASR models, the forced alignments they produce, and the boundary classifier
that decides when a word can be committed. **Stage 2 learns the tokenizer**:
the subword acoustic embedding and the residual-VQ codebooks that turn it into
discrete units. The speech LM is trained afterwards on the resulting units.

### Stage 1 — Alignment

| Step | What it does | Where |
|---|---|---|
| 1.1 Streaming ASR | Train the char- and word-level RNN-T models. Chunk 160 ms, FastEmit λ 0.04. | [`run_train_char_asr.sh`](streamASR/scripts/run_train_char_asr.sh), [`run_train_asr_word_fastemit.sh`](streamASR/scripts/run_train_asr_word_fastemit.sh) |
| 1.2 Alignment data | Generate TextGrids and the boundary-classifier dataset. | [`generate_textgrids.sh`](streamASR/scripts/generate_textgrids.sh), [`create_boundary_dataset.sh`](streamASR/scripts/create_boundary_dataset.sh) |
| 1.3 Boundary classifier | Train the proactive word-boundary detector, which cuts commit latency. | [`train_boundary_classifier.sh`](streamASR/scripts/train_boundary_classifier.sh) |

### Stage 2 — Tokenizer

| Step | What it does | Where |
|---|---|---|
| 2.1 Word distillation | Initialize the subword acoustic embedding from the word ASR. | [`train_word_distill.sh`](streamASR/scripts/train_word_distill.sh) |
| 2.2a **R=32 tokenizer — speech reconstruction** | Paper recipe: continuous → R=32 RVQ → cosine decay. This is the configuration behind the paper's reconstruction results. | [`train_tokenizer_r32_pipeline.sh`](streamASR/scripts/train_tokenizer_r32_pipeline.sh) |
| 2.2b **R=16 tokenizer — spoken language modeling** | The same three phases at R=16. Produces the units the SLM is trained on. | [`train_tokenizer_r16_pipeline.sh`](streamASR/scripts/train_tokenizer_r16_pipeline.sh) |

### Speech LM

| Step | What it does | Where |
|---|---|---|
| 3.1 Unit extraction + training | Extract `(w, q, d)` units with the **R=16** tokenizer from step 2.2b, then train the SLM on them. **R=16 is the configuration used for spoken language modeling**, and is what the released SLM is paired with. | [`streamSLM/scripts/run_train_slm_r16.sh`](streamSLM/scripts/run_train_slm_r16.sh) |

Dataset manifests come from `streamASR/utils/librispeech_prepare.py` and
`libritts_prepare.py`, which write the CSV/JSON the stages read.

Full details: [tokenizer pipeline](streamASR/scripts/README.md) ·
[speech LM](streamSLM/README.md) · [examples](examples/README.md)

## Setup

```bash
conda create -n streamASR python=3.10 && conda activate streamASR
pip install -r streamASR/requirements.txt
```

SpeechBrain is **vendored** at `streamASR/speechbrain`, so `import speechbrain`
resolves there whenever `streamASR/` is on the path; do not install it
separately. Resynthesis needs a [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
checkout for the flow + HiFT decoder, pointed to by `COSYVOICE_ROOT`.

If `import torchaudio` fails with an `undefined symbol` error, a user-site
PyTorch is shadowing the environment's. Prefix commands with
`PYTHONNOUSERSITE=1`.

## Quickstart

Both demos run against the released weights and need no training.

```bash
# Fetch the tokenizer stack and the SLM from the Hub
bash examples/download_weights.sh

# Tokenizer: encode LibriSpeech to R=32 units and resynthesize
bash examples/tokenizer_reconstruction_librispeech.sh 10

# SLM: continue LibriTTS prompts and decode to audio
bash examples/slm_continuation_libritts.sh 5
```

## Training

Run the stages in the order of the table above. Pick the tokenizer pipeline
that matches your task: **R=32 for speech reconstruction**, R=16 for the SLM.

Both pipelines share the same shape. An alignment stage trains the speech
encoder for chunk-level speech-to-text alignment; the encoder is then frozen
and the remaining modules are trained in three phases — continuous acoustic
features with the quantizer bypassed, then RVQ enabled, then a cosine decay to
zero.

**Shared by both:** the same trainers (`train_tokenizer.py`, then
`train_tokenizer_cosine.py` for the decay phase), a plain EMA-updated RVQ
codebook of size 512 at code dimension 256 (leave `RVQ_CODEBOOK_DIM` unset so it
defaults to `feat_dim`), commitment weight 1.0, chunk 4 / left context 32, and
peak learning rates 1e-4, 1e-4, 1e-5 across the three phases.

**Where they differ:**

| | R=32 — reconstruction | R=16 — SLM |
|---|---|---|
| Residual stages | 32 | 16 |
| Phase chaining | `--subalign_init_path` (fresh optimizer, epoch reset) | `--resume_path` |

### R=32 — speech reconstruction (paper recipe)

This is the tokenizer the paper's reconstruction results come from.

```bash
cd streamASR
# Stage 1 — char-level RNN-T aligner, and the boundary classifier
bash scripts/train_tokenizer_r32_pipeline.sh char_asr
bash scripts/train_tokenizer_r32_pipeline.sh boundary

# Stage 2 — continuous → R=32 RVQ → cosine decay
bash scripts/train_tokenizer_r32_pipeline.sh continuous
PHASE_CKPT=<continuous_ckpt>.pt bash scripts/train_tokenizer_r32_pipeline.sh r32
PHASE_CKPT=<r32_ckpt>.pt        bash scripts/train_tokenizer_r32_pipeline.sh cosine

# Streaming reconstruction on LibriSpeech test-clean + whisper-large-v3 WER
CKPT=<cosine_ckpt>.pt bash scripts/train_tokenizer_r32_pipeline.sh eval
```

The `eval` stage reproduces the reconstruction metric end to end: it resynthesizes test-clean,
pairs the output with the reference transcripts, and scores it with
whisper-large-v3. For reference, the paper reports **WER 4.41%** at a 2.97 Hz
unit rate with 270 ms latency; this pipeline reaches WER 4.43% / CER 1.92% /
UTMOS 4.23 at cosine-phase epoch 13, which is the checkpoint to select — the
remaining cosine tail plateaus.

### R=16 — spoken language modeling

```bash
cd streamASR
bash scripts/train_tokenizer_r16_pipeline.sh continuous
RESUME_PATH=<stage1_ckpt>.pt bash scripts/train_tokenizer_r16_pipeline.sh r16
RESUME_PATH=<stage2_ckpt>.pt bash scripts/train_tokenizer_r16_pipeline.sh cosine
```

Identical to the R=32 recipe apart from the quantizer depth. The resulting
tokenizer produces the units the SLM is trained on.

## Pretrained weights

| Repo | Contents |
|---|---|
| [`dd3434/Streamalign-R32`](https://huggingface.co/dd3434/Streamalign-R32) | **R=32 tokenizer** for speech reconstruction, plus the char alignment model, word streaming ASR, and boundary classifier. |
| [`dd3434/Streamalign-R16`](https://huggingface.co/dd3434/Streamalign-R16) | R=16 tokenizer, char alignment model, word streaming ASR, boundary classifier. |
| [`dd3434/Streamalign-SLM-R16`](https://huggingface.co/dd3434/Streamalign-SLM-R16) | Speech LM, Llama-3.2-1B backbone, hierarchical AR with duration-first ordering. |

Only the tokenizer itself is R-specific; the ASR, alignment, and boundary
components are shared across R8/R16/R32. Set `RVQ_R` to match the checkpoint you
load (32 or 16) with `RVQ_CODEBOOK_SIZE=512`, and leave `RVQ_CODEBOOK_DIM` unset.

**Use R=32 for reconstruction and R=16 for the SLM**, matching how each was
trained: the reconstruction demo in `examples/` runs the R=32 release, while the
continuation demo runs the R=16 stack because the released SLM was trained on
R=16 units. To reproduce the paper's reconstruction numbers from scratch, train
with `train_tokenizer_r32_pipeline.sh` and run its `eval` stage.

## License

Code is released under the **Apache License 2.0** (see [`LICENSE`](LICENSE)).
This repository builds on [SpeechBrain](https://github.com/speechbrain/speechbrain)
(Apache 2.0), vendored at `streamASR/speechbrain`, and uses
[CosyVoice](https://github.com/FunAudioLLM/CosyVoice) for resynthesis.

## Citation

```bibtex
@inproceedings{kim2026streamalign,
  title     = {{StreamAlign: Streaming Text-Aligned Speech Tokenization}},
  author    = {Kim, Kang-wook and Park, Jinyoung and Kim, Jinsoo and
               Lee, Sehun and Woo, Tony and Kim, Gunhee},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  year      = {2026}
}
```
