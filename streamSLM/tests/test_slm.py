"""StreamSLM smoke test using a tiny in-process Causal-LM stub.

Avoids any HF Hub download; substitutes a minimal backbone with the same
public interface that StreamSLM relies on (config.hidden_size /
config.vocab_size, get_input_embeddings, forward returning logits +
hidden_states).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from streamSLM.config import ModelConfig, TokenizerConfig
from streamSLM.model import slm as slm_mod
from streamSLM.model.slm import StreamSLM


# --------------------------------------------------------------------------- #
# Tiny stand-in for AutoModelForCausalLM
# --------------------------------------------------------------------------- #
class _TinyCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 64, hidden: int = 32, n_layers: int = 2):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=hidden,
            vocab_size=vocab_size,
            num_hidden_layers=n_layers,
        )
        self.embed = nn.Embedding(vocab_size, hidden)
        self.layers = nn.ModuleList(
            [nn.TransformerEncoderLayer(hidden, nhead=4, batch_first=True) for _ in range(n_layers)]
        )
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, inputs_embeds, attention_mask=None, position_ids=None,
                output_hidden_states=False, use_cache=False, return_dict=True):
        h = inputs_embeds
        # causal mask for an honest streaming-style stub
        T = h.size(1)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=h.device), diagonal=1)
        hidden_states = [h]
        for layer in self.layers:
            h = layer(h, src_mask=causal)
            hidden_states.append(h)
        logits = self.lm_head(h)
        return SimpleNamespace(
            logits=logits,
            hidden_states=tuple(hidden_states) if output_hidden_states else (h,),
        )


def _patch_backbone(monkey_vocab: int = 64, monkey_hidden: int = 32):
    """Patch AutoModelForCausalLM.from_pretrained inside slm.py."""
    class _Stub:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return _TinyCausalLM(vocab_size=monkey_vocab, hidden=monkey_hidden)
    slm_mod.AutoModelForCausalLM = _Stub


# --------------------------------------------------------------------------- #
def test_streamslm_forward_and_loss():
    _patch_backbone(monkey_vocab=64, monkey_hidden=32)

    model_cfg = ModelConfig(delay=1)
    tok_cfg = TokenizerConfig(
        rvq_num_quantizers=4, rvq_codebook_size=64,
    )

    model = StreamSLM(model_cfg, tok_cfg)
    model.eval()

    B, T, R, K = 2, 12, tok_cfg.num_quantizers, tok_cfg.codebook_size
    subword_ids = torch.randint(0, 64, (B, T))
    q_codes = torch.randint(0, K, (B, T, R))
    duration_frames = torch.randint(1, 25, (B, T))
    attn = torch.ones(B, T, dtype=torch.bool)
    attn[1, T - 3:] = False  # pretend last sample has 3 pad slots

    out = model(subword_ids, q_codes, duration_frames, attn)
    assert out["text_logits"].shape == (B, T, 64), out["text_logits"].shape
    assert out["acoustic_logits"].shape == (B, T, R, K), out["acoustic_logits"].shape
    assert out["duration_pred"].shape == (B, T), out["duration_pred"].shape

    loss = model.compute_loss(out, subword_ids, q_codes, duration_frames, attn)
    for k in ("loss", "loss_text", "loss_acoustic", "loss_duration"):
        v = loss[k]
        assert torch.isfinite(v).all(), f"{k} non-finite: {v}"

    # backward sanity
    loss["loss"].backward()
    grads_seen = sum(int(p.grad is not None) for p in model.parameters() if p.requires_grad)
    assert grads_seen > 0


if __name__ == "__main__":
    test_streamslm_forward_and_loss()
    print("test_streamslm_forward_and_loss: OK")
