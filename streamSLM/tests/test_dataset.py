"""Synthetic-data smoke for SubwordUnitsDataset + PadCollator."""

from __future__ import annotations

import csv
import os
import tempfile

import torch

from streamSLM.units import SubwordUnits
from streamSLM.data.dataset import SubwordUnitsDataset, PadCollator


def _make_fake_corpus(root: str, n_utts: int = 5, R: int = 32) -> str:
    """Write n_utts fake .units.pt files + a manifest CSV. Returns manifest path."""
    rng = torch.Generator().manual_seed(0)
    manifest_path = os.path.join(root, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rel_path", "n_subwords", "n_frames_total", "units_pt"])
        for i in range(n_utts):
            n = int(torch.randint(5, 30, (1,), generator=rng).item())
            sids = torch.randint(0, 32_000, (n,), generator=rng)
            qcs = torch.randint(0, 64, (n, R), generator=rng)
            durs = torch.randint(2, 25, (n,), generator=rng)
            u = SubwordUnits(sids, qcs, durs)
            p = os.path.join(root, f"utt{i:03d}.units.pt")
            u.save(p)
            w.writerow([f"utt{i:03d}.wav", n, int(durs.sum().item()), p])
    return manifest_path


def test_dataset_collator_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        manifest = _make_fake_corpus(tmp, n_utts=4, R=32)
        ds = SubwordUnitsDataset(manifest, min_subwords=1, max_subwords=64)
        assert len(ds) == 4
        item = ds[0]
        assert item["q_codes"].shape[1] == 32

        # Symmetric-delay collator: per-sample trailing slot count = delay.
        DELAY = 2
        EOS = 128_002
        PAD = 128_001
        collate = PadCollator(pad_token_id=PAD, eos_token_id=EOS, delay=DELAY)
        batch = collate([ds[i] for i in range(len(ds))])

        B, T = batch["subword_ids"].shape
        assert B == 4
        assert batch["q_codes"].shape == (B, T, 32)
        assert batch["duration_frames"].shape == (B, T)
        assert batch["attention_mask"].shape == (B, T)
        assert batch["text_label_mask"].shape == (B, T)
        assert batch["aco_label_mask"].shape == (B, T)
        # Each row has its original L_i tokens, an EOS at slot L_i, then
        # DELAY-1 PAD slots. q/d are zero outside [0, L_i). Masks pre-baked.
        for i in range(B):
            L = int(batch["lengths"][i])
            assert batch["subword_ids"][i, L] == EOS
            assert (batch["subword_ids"][i, L + 1 :] == PAD).all()
            assert (batch["q_codes"][i, L:] == 0).all()
            assert (batch["duration_frames"][i, L:] == 0).all()
            # attention covers [0, L_i] (real + EOS); not the trailing PAD slots.
            assert batch["attention_mask"][i, : L + 1].all()
            assert not batch["attention_mask"][i, L + 1 :].any()
            # text prediction positions: [0, L_i-1].
            assert batch["text_label_mask"][i, :L].all()
            assert not batch["text_label_mask"][i, L:].any()
            # acoustic prediction positions: [delay-1, L_i+delay-2].
            assert batch["aco_label_mask"][i, DELAY - 1 : L + DELAY - 1].all()
            assert not batch["aco_label_mask"][i, : DELAY - 1].any()
            assert not batch["aco_label_mask"][i, L + DELAY - 1 :].any()


if __name__ == "__main__":
    test_dataset_collator_roundtrip()
    print("test_dataset_collator_roundtrip: OK")
