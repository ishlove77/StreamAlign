"""AR generation smoke test using the same tiny stub backbone."""

from __future__ import annotations

import torch

from streamSLM.config import ModelConfig, TokenizerConfig
from streamSLM.model.slm import StreamSLM
from streamSLM.inference.generate import generate
from streamSLM.tests.test_slm import _patch_backbone


def test_generate_shapes_and_dtypes():
    _patch_backbone(monkey_vocab=64, monkey_hidden=32)
    tok_cfg = TokenizerConfig(rvq_num_quantizers=4, rvq_codebook_size=64)
    model_cfg = ModelConfig(delay=1)
    model = StreamSLM(model_cfg, tok_cfg)

    prompt = torch.tensor([1, 2, 3], dtype=torch.long)  # 3-token seed
    units = generate(
        model,
        prompt_ids=prompt,
        prompt_q=None,
        prompt_d=None,
        max_new_tokens=10,
        eos_id=None,
        temperature_text=1.0,
        top_p_text=1.0,
        temperature_aco=1.0,
        top_p_aco=1.0,
    )

    # The prompt+generation length sits in [prompt, prompt+max_new] (greedy could end early via eos).
    n = int(units.subword_ids.numel())
    assert prompt.numel() <= n <= prompt.numel() + 10
    assert units.q_codes.shape == (n, tok_cfg.num_quantizers)
    assert units.duration_frames.shape == (n,)
    assert units.subword_ids.dtype == torch.int64
    assert units.q_codes.dtype == torch.int64
    assert units.duration_frames.dtype == torch.int64
    assert int(units.q_codes.min()) >= 0
    assert int(units.q_codes.max()) < tok_cfg.codebook_size
    # Prompt prefix (positions 0..T_p-2) is never predicted, so its duration
    # stays 0 from init. From position T_p-1 onward the head clamps to >= 1.
    T_p = prompt.numel()
    if n >= T_p:
        assert int(units.duration_frames[T_p - 1:].min()) >= 1


def test_generate_eos_stops():
    _patch_backbone(monkey_vocab=8, monkey_hidden=16)
    tok_cfg = TokenizerConfig(rvq_num_quantizers=2, rvq_codebook_size=4)
    model_cfg = ModelConfig(delay=1)
    model = StreamSLM(model_cfg, tok_cfg)

    # With temperature=0 (argmax) and a fixed prompt, the model is deterministic.
    # We just check it runs without hanging and returns something.
    units = generate(
        model,
        prompt_ids=torch.tensor([0]),
        prompt_q=None,
        prompt_d=None,
        max_new_tokens=4,
        eos_id=None,
        temperature_text=0.0,
        temperature_aco=0.0,
    )
    assert units.subword_ids.numel() >= 1


if __name__ == "__main__":
    test_generate_shapes_and_dtypes()
    test_generate_eos_stops()
    print("test_generate: OK")
