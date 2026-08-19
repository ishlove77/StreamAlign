"""End-to-end smoke: dataset -> collator -> StreamSLM forward+backward+step.

Uses the same _TinyCausalLM stub as test_slm.py to avoid HF-Hub downloads.
Builds a synthetic manifest on the fly.
"""

from __future__ import annotations

import csv
import os
import tempfile

import torch
from torch.utils.data import DataLoader

from streamSLM.config import ModelConfig, TokenizerConfig
from streamSLM.data.dataset import SubwordUnitsDataset, PadCollator
from streamSLM.model import slm as slm_mod
from streamSLM.model.slm import StreamSLM
from streamSLM.units import SubwordUnits

# Reuse the patcher from test_slm by inlining (avoids cross-test fragility).
from streamSLM.tests.test_slm import _patch_backbone


def _make_corpus(root: str, n_utts: int, R: int, K: int):
    rng = torch.Generator().manual_seed(123)
    manifest = os.path.join(root, "manifest.csv")
    with open(manifest, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rel_path", "n_subwords", "n_frames_total", "units_pt"])
        for i in range(n_utts):
            n = int(torch.randint(8, 20, (1,), generator=rng).item())
            sids = torch.randint(0, 64, (n,), generator=rng)
            qcs = torch.randint(0, K, (n, R), generator=rng)
            durs = torch.randint(2, 25, (n,), generator=rng)
            p = os.path.join(root, f"utt{i:03d}.units.pt")
            SubwordUnits(sids, qcs, durs).save(p)
            w.writerow([f"utt{i:03d}.wav", n, int(durs.sum().item()), p])
    return manifest


def test_one_train_step():
    _patch_backbone(monkey_vocab=64, monkey_hidden=32)
    tok_cfg = TokenizerConfig(rvq_num_quantizers=4, rvq_codebook_size=64)
    model_cfg = ModelConfig(delay=1)
    R, K = tok_cfg.num_quantizers, tok_cfg.codebook_size

    with tempfile.TemporaryDirectory() as tmp:
        manifest = _make_corpus(tmp, n_utts=6, R=R, K=K)
        ds = SubwordUnitsDataset(manifest, min_subwords=1, max_subwords=64)
        loader = DataLoader(
            ds, batch_size=3,
            collate_fn=PadCollator(pad_token_id=0, eos_token_id=1, delay=model_cfg.delay),
        )

        model = StreamSLM(model_cfg, tok_cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

        first_loss = None
        last_loss = None
        for step, batch in enumerate(loader):
            out = model(batch["subword_ids"], batch["q_codes"],
                        batch["duration_frames"], batch["attention_mask"])
            losses = model.compute_loss(out, batch["subword_ids"], batch["q_codes"],
                                        batch["duration_frames"], batch["attention_mask"])
            loss = losses["loss"]
            assert torch.isfinite(loss).all()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if first_loss is None:
                first_loss = float(loss)
            last_loss = float(loss)
        # Loss should not have exploded; with 2 mini-batches we don't require strict descent.
        assert last_loss < first_loss * 5.0, (first_loss, last_loss)


if __name__ == "__main__":
    test_one_train_step()
    print("test_one_train_step: OK")
