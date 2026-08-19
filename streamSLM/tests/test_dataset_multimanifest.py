"""Verify SubwordUnitsDataset accepts multiple manifest sources at once.

Builds two synthetic manifest dirs, then loads them as a single dataset
either as a list of dirs or as a glob, and confirms total length matches.
"""

from __future__ import annotations

import csv
import os
import tempfile

import torch

from streamSLM.units import SubwordUnits
from streamSLM.data.dataset import SubwordUnitsDataset


def _make_shard(root: str, prefix: str, n: int, R: int = 4):
    os.makedirs(root, exist_ok=True)
    manifest = os.path.join(root, f"{prefix}.csv")
    rng = torch.Generator().manual_seed(hash(prefix) & 0xFFFF)
    with open(manifest, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rel_path", "n_subwords", "n_frames_total", "units_pt"])
        for i in range(n):
            sids = torch.randint(0, 1000, (8,), generator=rng)
            qcs = torch.randint(0, 64, (8, R), generator=rng)
            durs = torch.randint(2, 25, (8,), generator=rng)
            p = os.path.join(root, f"{prefix}_{i:03d}.units.pt")
            SubwordUnits(sids, qcs, durs).save(p)
            w.writerow([f"{prefix}/{i:03d}.wav", 8, int(durs.sum().item()), p])


def test_dirs_list():
    with tempfile.TemporaryDirectory() as tmp:
        _make_shard(os.path.join(tmp, "ls"), "ls", n=3)
        _make_shard(os.path.join(tmp, "em"), "em", n=5)
        ds = SubwordUnitsDataset(
            [os.path.join(tmp, "ls"), os.path.join(tmp, "em")],
            min_subwords=1,
        )
        assert len(ds) == 8


def test_glob():
    with tempfile.TemporaryDirectory() as tmp:
        _make_shard(os.path.join(tmp, "ls"), "shard0", n=2)
        _make_shard(os.path.join(tmp, "ls"), "shard1", n=4)
        ds = SubwordUnitsDataset(os.path.join(tmp, "ls", "shard*.csv"), min_subwords=1)
        assert len(ds) == 6


if __name__ == "__main__":
    test_dirs_list()
    test_glob()
    print("test_dataset_multimanifest: OK")
