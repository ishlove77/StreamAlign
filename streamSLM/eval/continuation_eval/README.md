# continuation_eval

Standalone evaluation for StreamSLM speech-continuation outputs. Computes:

- **UTMOS** — UTMOS22-strong (same `ftshijt/SpeechMOS:main` checkpoint VERSA uses).
- **SECS** — Speaker Embedding Cosine Similarity using ESPnet2's
  RawNet3 (`espnet/voxcelebs12_rawnet3`), L2-normalized cosine.
  Reference audio = the CosyVoice speaker prompt
  (`input_wavs/<stem>.wav`); generated audio = `<stem>_with_prompt.wav`.
- **GPT-4o** — semantic-coherence judge over transcripts
  (`whisper_continuation` vs. ground-truth prompt text); model
  `gpt-4o-2024-08-06`, 1–5 rubric in `prompts/semantic_coherence.md`.

The OpenAI API key is read from `~/.env` (`export OPENAI_API_KEY="..."`) and is
never echoed.

## Setup (one-time)

```bash
cd streamSLM/eval/continuation_eval
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -e .
```

## Run

Default model dirs (R16 + R32_d16 delay-2 emilia4000h latest):

```bash
.venv/bin/python -m streamSLM.eval.continuation_eval.run
```

Pick metrics / smoke-test on a few utts:

```bash
.venv/bin/python -m streamSLM.eval.continuation_eval.run \
  --metrics utmos,secs \
  --max_utts 3
```

Outputs land at `logs/continuation_eval/<model_dir>/{utmos,secs,gpt}.jsonl`,
`per_utt.csv`, `summary.json`, plus an overall `all_summaries.json`.

The driver runs on one GPU (UTMOS + SECS). The GPT judge is CPU/network only
and can be split off via a second invocation with `--metrics gpt`.
