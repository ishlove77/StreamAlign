# Examples

Two runnable examples: **reconstruction** through the R=32 tokenizer, and
**continuation** with the (R=16) speech LM.

## 0. Get the weights

```bash
bash examples/download_weights.sh          # -> ./weights
```

This pulls `dd3434/Streamalign-R32` (R=32 tokenizer stack, ~1.6 GB:
the R=32 RVQ tokenizer, the char-level alignment model, the word-level
streaming ASR, and the boundary classifier), plus `dd3434/Streamalign-R16` /
`dd3434/Streamalign-SLM-R16` for the SLM continuation example.

Both examples also need a [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
checkout for the flow + HiFT decoder that turns units back into audio. Point
`COSYVOICE_ROOT` at it (default: `streamASR/third_party/CosyVoice`).

## 1. Tokenizer: LibriSpeech reconstruction (R=32)

Encodes utterances to R=32 units and resynthesizes them, so you can hear what
survives tokenization.

```bash
bash examples/tokenizer_reconstruction_librispeech.sh 10
```

Writes one `.flac` per utterance to `outputs/reconstruction/test-clean/`,
mirroring the LibriSpeech directory layout.

The argument selects the manifest `$LIBRISPEECH/csv/test-clean.first<N>.csv`, so
it only works for an `N` you actually have a manifest for. Point `TEST_CSV` at
any CSV with columns `ID,duration,wav,spk_id,wrd` to use your own.

Each output mirrors the input path, so you can play a reconstruction against
its original to hear what survives tokenization:

```
outputs/reconstruction/test-clean/3575/170457/3575-170457-0021.flac   # reconstructed
$LIBRISPEECH/test-clean/3575/170457/3575-170457-0021.flac            # original
```

Reference numbers for this checkpoint (full test-clean, streaming, scored with
whisper-large-v3 / versa): WER 4.43% · CER 1.92% · UTMOS 4.23 · SECS 0.585.

## 2. SLM: LibriTTS continuation (R=16)

The released SLM was trained on R=16 units, so this example still runs on the
`dd3434` R=16 stack (an R=32 SLM would need retraining on R=32 units).

Conditions on the first few subwords of a LibriTTS prompt and lets the SLM
continue it, then decodes the result to audio.

```bash
bash examples/slm_continuation_libritts.sh 5
```

Prompts are sampled from `LIBRITTS/test-clean` into
`outputs/continuation_prompts/`; point `PROMPT_DIR` at a directory of your own
`.wav` files to skip that. Outputs land in `outputs/continuation/`.

`PROMPT_MAX_SUBWORDS` (default 20) sets how much of the prompt conditions the
model and `MAX_NEW_TOKENS` (default 16) how many subwords it generates.

## Notes

- `RVQ_R=32` and `RVQ_CODEBOOK_SIZE=512` must match the R=32 checkpoint (the
  reconstruction script sets them). Leave `RVQ_CODEBOOK_DIM` unset:
  `codebook_dim` defaults to `feat_dim=256`, which is what the released
  weights expect. Setting it explicitly causes a shape mismatch when loading.
- The reconstruction script uses `boundary_classifier/best_model.pt` (the
  checkpoint the reference numbers were measured with);
  `best_precision_model.pt` is also included if you want fewer false-positive
  word commits.
- If `import torchaudio` fails with an `undefined symbol` error, a user-site
  PyTorch is shadowing the environment's. Prefix commands with
  `PYTHONNOUSERSITE=1`.
- The SLM path pins `STREAMSLM_ATTN_IMPL=sdpa`; the `flash_attention_2` default
  hangs at startup on mixed-generation GPUs.
