# StreamSLM

> **Paper reproduction (StreamAlign-SLM, R16).** Reported: SALMon 69.1,
> StoryCloze 72.1. Pretrained weights (HF): `Streamalign-R16` (tokenizer
> stack), `Streamalign-SLM-R16` (the SLM).
>
> 1. **Extract** RVQ units: `bash streamSLM/scripts/run_extract_full_seq.sh`
>    (`RVQ_R=16 RVQ_CODEBOOK_SIZE=512`, teacher = `Streamalign-R16`).
> 2. **Train** (unified two-phase): `PHASE=all bash streamSLM/scripts/run_train_slm_r16.sh`
>    - Phase 1 pretrain — `loss_w_text=1.0`, `text_kl_weight=0.9`, 300k steps.
>    - Phase 2 finetune — resume phase-1 @180k with `loss_w_text=5.0`,
>      `text_kl_weight=0.5`, → 210k. The ~185k checkpoint is the reported model.
>    Recipe: RVQ R16/C512, hierarchical AR (duration-first), delay=2, audioboost,
>    Llama-3.2-1B backbone + text-KL distillation. Needs `CACHE_ROOT`, `MOSHI_ROOT`.
> 3. **Evaluate**: `run_eval_storycloze_parallel.sh`, `run_eval_salmon_parallel.sh`,
>    and `scripts/examples/run_speech_continuation.sh` +
>    `python -m streamSLM.eval.continuation_eval.run` (UTMOS / SECS / GPT-4o).

Speech language model trained on RVQ subword units extracted from a
StreamAlign teacher. Each subword `n` is represented by a tuple
`(w_n, q_n, d_n)`:

| Field             | Shape    | Meaning                                                  |
|-------------------|----------|----------------------------------------------------------|
| `subword_ids`     | `(N,)`   | LLM-tokenizer token id (Llama-3.2 by default)            |
| `q_codes`         | `(N, R)` | RVQ codebook indices, `R` quantizers, codebook size `C`     |
| `duration_frames` | `(N,)`   | Encoder-frame count per subword (25 Hz, 40 ms / frame)   |

Default tokenization is **RVQ R=16, C=512**, teacher
`weights/Streamalign-R16/rvq_teacher/epoch_22.pt`
(released as `Streamalign-R16`, `rvq_teacher/epoch_22.pt`).

The model is a Llama-3.2-1B backbone + audio embedding fusion + delayed-prediction
heads (`text`, `acoustic`, `duration`).

```
streamSLM/
├── extract/           per-shard unit extractor (see top-level CLAUDE.md)
├── train/train.py     training entry point
├── inference/
│   ├── generate.py        AR sampling of (w, q, d) -> .units.pt
│   ├── reconstruct.py     .units.pt -> waveform via StreamAlign + CosyVoice
│   └── synthesize.py      end-to-end CLI (text/units prompt -> wav)
├── eval/              StoryCloze, SALMon, token accuracy, test-loss
├── scripts/           launchers (run_train_*, run_eval_*, etc.)
├── docs/              extraction / experiment notes
└── config.py units.py model/ data/
```

The repo CLAUDE.md is the source of truth for tokenization. This README only
covers **training** and **inference**.

---

## 1. Prerequisites

- Slurm via the `sr` wrapper (see project `CLAUDE.md` — never call `srun`/`sbatch`).
- A populated unit cache (RVQ for the default config). Standard layout:

  ```
  cache/streamSLM_units_<TAG>/
    librispeech/{train-clean-100, train-clean-360, train-other-500}/manifest_shard{R}_of{W}.csv
    emilia/{400h, full}/manifest_shard{R}_of{W}.csv
    .../<rel-audio-path>.units.pt
  ```

  Run `streamSLM/scripts/run_extract_full_seq.sh` (16-way) or
  `submit_extract_emilia_full.sh` (48-way) to (re)build the cache.

- A StreamAlign teacher checkpoint (`.pt`) — required for inference / eval and
  for any extraction config you re-run.

---

## 2. Training

### Quick launch — RVQ C=512 / R=16 (released configuration)

```
PHASE=all bash streamSLM/scripts/run_train_slm_r16.sh
```

Two phases: pretrain with `loss_w_text=1.0`, `text_kl_weight=0.9` to 300k steps,
then finetune from the 180k checkpoint with `loss_w_text=5.0`,
`text_kl_weight=0.5` to 210k. The ~185k checkpoint is the released model
(`Streamalign-SLM-R16`). Needs `CACHE_ROOT` and `MOSHI_ROOT`.

Defaults: Llama-3.2-1B backbone, RVQ R=16 / C=512, hierarchical AR with
duration-first ordering, `delay=2`, audioboost, bf16, KL distillation against a
frozen Llama-3.2-1B.

### Distill / packed / DDP variants

The `_abl_common.sh` orchestrator exposes the full knob set via env vars.
Common toggles:

| Env var                       | Purpose                                                |
|-------------------------------|--------------------------------------------------------|
| `RVQ_NQ` `RVQ_C`              | RVQ quantizers / codebook size (released: 16 / 512)    |
| `PACKING=1` `PACK_MAX_TOKENS` | B=1 packed sequences (TASTE-style)                     |
| `TEXT_KL_WEIGHT`              | Stage-2 KL distillation weight (0 drops `ref_model`)   |
| `ACOUSTIC_TARGET`             | `rvq` (default) / `continuous`                         |
| `ACOUSTIC_LAYER_MIX`          | `last` (default) / `weighted`                          |
| `PREDICTOR_TYPE`              | `depth_transformer` (default) / `mlp`                  |
| `DEPFORMER_{DIM,HEADS,LAYERS,FF}` | Depth-transformer geometry                         |
| `NGPU`                        | `>1` ⇒ launch via `torchrun --nproc_per_node=$NGPU`    |
| `LAUNCHER`                    | Job-submission prefix (empty = run directly; e.g. a Slurm wrapper) |
| `STREAMSLM_ATTN_IMPL`         | `flash_attention_2` (Ampere+) / `sdpa` (eval pool)     |
| `STREAMSLM_GRADIENT_CHECKPOINTING=1` | Drop activation memory for big backbones        |
| `RESUME`                      | Path to `step_*.pt` to resume from                     |
| `RESUME_MODEL_ONLY=1`         | Warm-start: load weights only, fresh optim/start_step  |
| `LAUNCH_FG=1`                 | Run synchronously instead of `nohup` (for watchdogs)   |

Direct (no shell wrapper):

```
python -u -m streamSLM.train.train \
    --manifest cache/streamSLM_units_C512_R16_rvq/librispeech/train-clean-100/manifest_shard*_of32.csv \
                cache/streamSLM_units_C512_R16_rvq/.../manifest_shard*_of32.csv \
    --out_dir checkpoints/streamSLM/my_run \
    --backbone meta-llama/Llama-3.2-1B \
    --token_type rvq --rvq_num_quantizers 16 --rvq_codebook_size 512 \
    --delay 1 --batch_size 8 --grad_accum 4 --lr 2e-4 \
    --warmup_steps 1000 --max_steps 50000 \
    --save_every 2000 --log_every 50 --bf16
```

DDP single-node, 4-GPU:

```
sr 4 48 --qos=q-low torchrun --standalone --nproc_per_node=4 \
    -m streamSLM.train.train  ...same flags...
```

### Outputs

```
checkpoints/streamSLM/<RUN_NAME>/
├── step_00002000.pt          model + optim state, every SAVE_EVERY
├── step_00004000.pt
├── ...
├── latest.pt                 symlink/copy of most recent
└── config.json               run-level config dump
logs/streamSLM_train/<RUN_NAME>.log
```

A run is healthy when the log shows steady `loss/text`, `loss/acoustic`,
`loss/duration` decreases plus periodic `[val]` lines every `VAL_EVERY` steps.

---

## 3. Inference

### End-to-end (text or speech prompt → wav)

`streamSLM.inference.synthesize` chains AR generation and waveform reconstruction.

```
python -m streamSLM.inference.synthesize \
    --slm_checkpoint checkpoints/streamSLM/<RUN_NAME>/step_00010000.pt \
    --streamalign_ckpt weights/Streamalign-R16/rvq_teacher/epoch_22.pt \
    --variant rvq \
    --speaker_wav prompts/jane.wav \
    --text_prompt "Once upon a time" \
    --max_new_tokens 96 \
    --out_wav out/sample.wav
```

Speech continuation: replace `--text_prompt` with `--prompt_units`:

```
    --prompt_units cache/streamSLM_units_C512_R16_rvq/librispeech/.../foo.flac.units.pt \
    --prompt_max_subwords 8 \
    --max_new_tokens 64 \
```

(`--text_prompt` and `--prompt_units` are mutually exclusive; omit both for a
BOS-only prompt.)

Common flags:

| Flag                              | Default                   | Notes                              |
|-----------------------------------|---------------------------|------------------------------------|
| `--slm_checkpoint`                | required                  | StreamSLM `.pt`                    |
| `--streamalign_ckpt`              | required                  | Teacher used at extraction time    |
| `--variant`                       | `rvq`                     | Only `rvq` is supported            |
| `--speaker_wav`                   | required                  | Reference for spk emb + flow prompt |
| `--cosyvoice_model_dir`           | Fun-CosyVoice3-0.5B       | CosyVoice flow + hift              |
| `--max_new_tokens`                | 128                       | AR length cap                      |
| `--temperature_text` `--top_p_text` | 0.9 / 0.95             | Text head sampling                 |
| `--temperature_aco` `--top_p_aco`   | 1.0 / 0.9              | Acoustic head sampling             |
| `--save_units path.pt`            | off                       | Also dump generated `SubwordUnits` |
| `--out_wav`                       | required                  | Output `.wav`                      |

A `<out_wav>.txt` sidecar is written with the decoded text and per-subword IDs.

### AR generation only (units, no audio)

```
python -m streamSLM.inference.generate \
    --slm_checkpoint .../step_00050000.pt \
    --text_prompt "Once upon a time" \
    --max_new_tokens 96 \
    --out_units out/sample.units.pt
```

### Reconstruct from existing `.units.pt`

```
python -m streamSLM.inference.reconstruct \
    --units_pt out/sample.units.pt \
    --streamalign_ckpt weights/Streamalign-R16/rvq_teacher/epoch_22.pt \
    --variant rvq \
    --speaker_wav prompts/jane.wav \
    --out_wav out/sample.wav
```

Useful for re-rendering a generation with a different speaker or for sanity
checks against ground-truth `.units.pt` from the extraction cache.

---

## 4. Evaluation

### Zero-shot benchmarks

```
SLM_CKPT=checkpoints/streamSLM/<RUN_NAME>/step_00050000.pt \
TEACHER_CKPT=weights/Streamalign-R16/rvq_teacher/epoch_22.pt \
WORLD=8 \
bash streamSLM/scripts/run_eval_storycloze_parallel.sh
```

Fans out to `WORLD` parallel `${LAUNCHER}` shards, then merges
per-shard JSONs into `results/storycloze/<ckpt-tag>/<scoring_mode>/`.

For SALMon: `bash streamSLM/scripts/run_eval_salmon.sh`.

**On 24 GB GPUs always export `STREAMSLM_ATTN_IMPL=sdpa`** — FA2 requires
Ampere+ but the 24 GB pool includes Turing/Volta cards. The `_parallel.sh`
launcher already sets it; one-off scripts must export it explicitly.

### Token accuracy / test loss

```
python -m streamSLM.eval.test_token_acc --slm_checkpoint ... --val_manifest ...
python -m streamSLM.eval.test_loss      --slm_checkpoint ... --val_manifest ...
```

---

## 5. Tips & gotchas

- **Always use `sr`, never `sbatch`/`srun`.** QoS rules per global `CLAUDE.md`:
  default mid quota; `q-low` for long-running training/extraction; high quota
  only for short experiments.
- **C and R must match between extraction, training, and inference.**
  Mismatched `rvq_codebook_size`/`rvq_num_quantizers` will silently load
  wrong-shape audio embeddings or crash at the AR feedback step.
- **`delay` is part of the SLM contract**, not a sampler knob — train and
  generate with the same value. The first `delay` slots are zero-filled at
  inference, mirroring the learned PAD branch in `_delay_shift_audio`.
- **`speaker_wav`** is the reference for both the speaker embedding and the
  CosyVoice flow prompt. Output sample rate is always 24 kHz.
- **Continuous-acoustic models** (`acoustic_target=continuous`) need a teacher
  quantizer for AR feedback, passed in when calling `generate.generate(...)`
  directly.
- **Resuming**: `--resume path.pt` loads model+optim+step; add
  `--resume_model_only` for warm-starting a different-arch run from a baseline
  ckpt (layers up to the smaller `R` load, the rest reset).
