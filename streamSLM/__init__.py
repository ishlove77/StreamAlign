"""streamSLM: spoken language model trained on StreamAlign per-subword tokens.

Each unit is (w_n, q_n, d_n):
    w_n  subword id (Llama / Qwen3 BPE)
    q_n  R-dim RVQ code for the subword
    d_n  encoder-frame count (modeled as log-frames, L1 loss; see ModelConfig)

Delayed prediction:
    step n   -> predict w_n
    step n+1 -> predict (q_n, d_n)
"""
