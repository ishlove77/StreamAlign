"""Dataset + collator for the SLM.

Reads per-utterance .units.pt files (SubwordUnits) referenced by a manifest CSV.
Produces clean per-subword sequences; the delayed-prediction shift lives in
the model, not here, so the data layer stays unit-test-friendly.

Manifest schema (rows): rel_path, n_subwords, n_frames_total, units_pt
"""

from __future__ import annotations

import csv
import getpass
import glob
import gzip
import hashlib
import os
import pickle  # noqa: F401  (kept for backward-compat readers of .pkl caches)
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from streamSLM.units import SubwordUnits


def _read_manifest(path: str, min_subwords: int, max_subwords: int) -> List[str]:
    """Return units_pt paths from a single manifest CSV, length-filtered.

    Uses pandas for the cold scan (~5–10× faster than csv.DictReader on the
    12.9M-row Emilia/full manifests). Falls back to the stdlib csv reader if
    pandas isn't installed."""
    try:
        import pandas as pd
        df = pd.read_csv(path, usecols=["n_subwords", "units_pt"])
        mask = (df["n_subwords"] >= min_subwords) & (df["n_subwords"] <= max_subwords)
        return df.loc[mask, "units_pt"].tolist()
    except ImportError:
        out: List[str] = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                n = int(row["n_subwords"])
                if min_subwords <= n <= max_subwords:
                    out.append(row["units_pt"])
        return out


def _expand_manifests(manifests: Sequence[str]) -> List[str]:
    """Accept globs, single files, or directories (treats *.csv inside as shards)."""
    paths: List[str] = []
    for m in manifests:
        if os.path.isdir(m):
            paths.extend(sorted(glob.glob(os.path.join(m, "*.csv"))))
        else:
            paths.extend(sorted(glob.glob(m)) or [m])
    return paths


# Two-tier path cache. The fast tier is node-local /tmp (typically a few
# hundred MB/s), the durable tier is NFS (slow under contention). On read we
# probe fast → durable; on miss-then-scan we write to both. The 12.9M-path
# cache is ~150 MB gzipped (vs ~1.9 GB pickled), so even the durable tier
# decodes in under a minute on a healthy NFS mount.
_PATHS_CACHE_DIR_FAST = os.environ.get(
    "STREAMSLM_PATHS_CACHE_DIR_FAST",
    f"/tmp/{getpass.getuser()}/streamSLM_paths_cache",
)
_PATHS_CACHE_DIR = os.environ.get(
    "STREAMSLM_PATHS_CACHE_DIR",
    os.path.expanduser("~/.cache/streamSLM/manifest_paths"),
)


def _paths_cache_key(manifest_files: Sequence[str], min_subwords: int, max_subwords: int) -> str:
    """SHA-256 over (abspath, size, mtime_ns) per file + filter args. Any manifest
    edit/regeneration invalidates the cache automatically via mtime/size."""
    h = hashlib.sha256()
    h.update(f"v1|min={min_subwords}|max={max_subwords}\n".encode())
    for p in sorted(manifest_files):
        ap = os.path.abspath(p)
        try:
            st = os.stat(ap)
            h.update(f"{ap}|{st.st_size}|{st.st_mtime_ns}\n".encode())
        except FileNotFoundError:
            h.update(f"{ap}|MISSING\n".encode())
    return h.hexdigest()


def _read_paths_cache(cache_path: str) -> Optional[List[str]]:
    """Return list of paths from a gzip text cache, or None on miss/corruption."""
    if not cache_path or not os.path.exists(cache_path):
        return None
    try:
        with gzip.open(cache_path, "rb") as f:
            blob = f.read()
        paths = blob.decode("utf-8").splitlines()
        return paths or None
    except (OSError, gzip.BadGzipFile, UnicodeDecodeError):
        return None


def _write_paths_cache(cache_path: str, paths: List[str]) -> None:
    """Atomically write paths to a gzip text cache. Best-effort — silent on failure."""
    if not cache_path:
        return
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp = f"{cache_path}.tmp.{os.getpid()}"
        # compresslevel=3 is ~3x faster than the default 9 with only ~5%
        # larger output on path-string data — write speed matters because the
        # writer holds up the first launch's startup until rename completes.
        with gzip.open(tmp, "wb", compresslevel=3) as f:
            f.write(("\n".join(paths) + "\n").encode("utf-8"))
        os.replace(tmp, cache_path)
    except OSError:
        pass


def _load_units_paths(manifest_files: Sequence[str], min_subwords: int, max_subwords: int) -> List[str]:
    """Parse manifests through a tiered on-disk cache.

    Probes (fast=/tmp, durable=NFS) in order; cache hit returns the gzip
    text blob directly. On miss, runs the cold pandas scan and writes to
    both tiers. Reading from durable also mirrors to fast.

    Disable via ``STREAMSLM_PATHS_CACHE_DIR=`` (empty) — useful for debugging.
    Disable just the fast tier via ``STREAMSLM_PATHS_CACHE_DIR_FAST=``.
    """
    if not _PATHS_CACHE_DIR and not _PATHS_CACHE_DIR_FAST:
        return _scan_units_paths(manifest_files, min_subwords, max_subwords)

    key = _paths_cache_key(manifest_files, min_subwords, max_subwords)
    fast = (
        os.path.join(_PATHS_CACHE_DIR_FAST, f"{key}.txt.gz")
        if _PATHS_CACHE_DIR_FAST else None
    )
    nfs = (
        os.path.join(_PATHS_CACHE_DIR, f"{key}.txt.gz")
        if _PATHS_CACHE_DIR else None
    )

    paths = _read_paths_cache(fast)
    if paths is not None:
        return paths
    paths = _read_paths_cache(nfs)
    if paths is not None:
        # Mirror to fast tier so subsequent launches on this node skip NFS.
        _write_paths_cache(fast, paths)
        return paths

    paths = _scan_units_paths(manifest_files, min_subwords, max_subwords)
    _write_paths_cache(fast, paths)
    _write_paths_cache(nfs, paths)
    return paths


def _scan_units_paths(manifest_files: Sequence[str], min_subwords: int, max_subwords: int) -> List[str]:
    out: List[str] = []
    for mf in manifest_files:
        out.extend(_read_manifest(mf, min_subwords, max_subwords))
    return out


class SubwordUnitsDataset(Dataset):
    """Loads (subword_ids, q_codes, duration_frames) per utterance.

    Parameters
    ----------
    manifests : single path / glob / dir, or list of any combination
        Each manifest is a CSV produced by streamSLM/extract/extract_tokens.py.
    min_subwords, max_subwords : keep utterances with N in [min, max].
        max_subwords doubles as the dataloader-side hard truncation safety net.
    return_pre_quant_feat : when True, also return the pre-quantizer continuous
        teacher feature (N, D) bf16 if it's saved in the .units.pt. Required
        for continuous-acoustic training (see streamSLM/model/slm.py).
    """

    def __init__(
        self,
        manifests,
        min_subwords: int = 4,
        max_subwords: int = 1024,
        return_pre_quant_feat: bool = False,
    ):
        if isinstance(manifests, (str, os.PathLike)):
            manifests = [manifests]
        manifest_files = _expand_manifests(manifests)
        if not manifest_files:
            raise FileNotFoundError(f"no manifests resolved from: {manifests}")

        self.units_paths = _load_units_paths(manifest_files, min_subwords, max_subwords)
        if not self.units_paths:
            raise RuntimeError(
                f"no utterances passed length filter [{min_subwords}, {max_subwords}] "
                f"across {len(manifest_files)} manifests"
            )
        self.max_subwords = max_subwords
        self.return_pre_quant_feat = return_pre_quant_feat

    def __len__(self) -> int:
        return len(self.units_paths)

    def __getitem__(self, idx: int):
        u = SubwordUnits.load(self.units_paths[idx])
        n = min(u.num_subwords, self.max_subwords)
        item = {
            "subword_ids":     u.subword_ids[:n],
            "q_codes":         u.q_codes[:n],
            "duration_frames": u.duration_frames[:n],
        }
        if self.return_pre_quant_feat:
            if u.pre_quant_feat is None:
                raise RuntimeError(
                    f"pre_quant_feat is missing from {self.units_paths[idx]}; "
                    "use a cache root extracted with --save_pre_quant_feat "
                    "(e.g. cache/streamSLM_units_C512_prequant)."
                )
            item["pre_quant_feat"] = u.pre_quant_feat[:n]
        return item


# --------------------------------------------------------------------------- #
# Collator
#
# Symmetric-delay convention (MusicGen-style next-step AR). For each sample of
# original length L_i, we emit T_i = L_i + delay slots:
#
#     subword_ids     : [w_0, ..., w_{L_i-1}, EOS,   PAD, ..., PAD]
#     q_codes         : [q_0, ...,  q_{L_i-1},  0,     0, ...,   0]
#     duration_frames : [d_0, ...,  d_{L_i-1},  0,     0, ...,   0]
#     attention_mask  : [  1,            1,     1,     0, ...,   0]   (EOS counted as real)
#     text_label_mask : [  1,            1,     0,     0, ...,   0]   prediction positions
#     aco_label_mask  : [  0,    ...    1,      1,     0, ...,   0]   True at [D-1, L_i+D-2]
#
# The model's compute_loss uses text_label_mask AFTER the HF causal-shift
# (so position L_i-1, whose target is the stored EOS at slot L_i, supervises
# EOS). aco targets are right-shifted by D-1 (see slm.py::compute_loss).
#
# IMPORTANT: this is a hard switch from the previous asymmetric-delay layout.
# Checkpoints trained before this change are NOT compatible because their
# inference loop wrote q_n into slot n (not slot n-D) and their training
# targeted q_n at position n directly. Retrain from scratch.
# --------------------------------------------------------------------------- #
@dataclass
class PadCollator:
    """Pad + symmetric-delay-extend to the longest sample in the batch.

    Each sample is extended by ``delay`` slots: one EOS at position L_i, then
    ``delay - 1`` padding slots (zeros). Pad values:
        subword_ids     -> pad_token_id at trailing PAD slots; EOS is its own id
        q_codes         -> 0
        duration_frames -> 0
    Also produces ``text_label_mask`` (True over prediction positions
    [0, L_i-1]) and ``aco_label_mask`` (True over [delay-1, L_i+delay-2]) so
    compute_loss can mask without recomputing per-sample lengths.

    Args:
        pad_token_id: LM tokenizer pad id used at trailing PAD slots.
        eos_token_id: id written at slot L_i of every sample.
        delay: number of trailing slots appended per sample (>= 1).
    """

    pad_token_id: int
    eos_token_id: int
    delay: int = 1

    def __call__(self, samples):
        B = len(samples)
        delay = max(1, int(self.delay))
        # Per-sample real length (without the trailing EOS+PADs).
        Ls = [int(s["subword_ids"].numel()) for s in samples]
        Tmax = max(Ls) + delay
        R = samples[0]["q_codes"].shape[1] if samples[0]["q_codes"].ndim == 2 else 0
        has_feat = "pre_quant_feat" in samples[0]
        F_dim = samples[0]["pre_quant_feat"].shape[1] if has_feat else 0

        subword_ids = torch.full((B, Tmax), self.pad_token_id, dtype=torch.long)
        q_codes = torch.zeros((B, Tmax, R), dtype=torch.long)
        duration_frames = torch.zeros((B, Tmax), dtype=torch.long)
        attn = torch.zeros((B, Tmax), dtype=torch.bool)
        text_label_mask = torch.zeros((B, Tmax), dtype=torch.bool)
        aco_label_mask = torch.zeros((B, Tmax), dtype=torch.bool)
        # Keep bf16 in pad to halve transfer bytes; consumer casts to fp32.
        pre_quant_feat = (
            torch.zeros((B, Tmax, F_dim), dtype=torch.bfloat16) if has_feat else None
        )

        for i, s in enumerate(samples):
            L = Ls[i]
            subword_ids[i, :L] = s["subword_ids"]
            subword_ids[i, L] = self.eos_token_id      # EOS at slot L_i
            q_codes[i, :L] = s["q_codes"]
            duration_frames[i, :L] = s["duration_frames"]
            # attention_mask includes the EOS slot but not the trailing PADs.
            attn[i, : L + 1] = True
            # text prediction positions: [0, L_i-1] — supervises EOS via the
            # HF shift at position L_i-1, but NOT trailing PAD prediction.
            text_label_mask[i, :L] = True
            # acoustic prediction positions: [delay-1, L_i+delay-2].
            aco_label_mask[i, delay - 1 : L + delay - 1] = True
            if has_feat:
                pre_quant_feat[i, :L] = s["pre_quant_feat"]

        out = {
            "subword_ids":     subword_ids,
            "q_codes":         q_codes,
            "duration_frames": duration_frames,
            "attention_mask":  attn,
            "text_label_mask": text_label_mask,
            "aco_label_mask":  aco_label_mask,
            # `lengths` stays the original per-sample length L_i (so downstream
            # code that uses it for stats / dataset-side checks is unaffected).
            "lengths":         torch.tensor(Ls, dtype=torch.long),
        }
        if has_feat:
            out["pre_quant_feat"] = pre_quant_feat
        return out


# --------------------------------------------------------------------------- #
# Sequence packing
#
# Concatenates multiple short utterances into one B=1 long sequence with
# `position_ids` resetting at each doc boundary. Removes ~83% padding waste at
# our default B=8/T_max=1024 with avg utt ~175 subwords.
#
# Why B=1: transformers 4.53's FA2 path only routes to varlen via
# `position_ids` resets when `query_states.shape[0] == 1` (see
# transformers/modeling_flash_attention_utils.py). With B>1, FA2 falls back to
# unpadded-but-not-cross-doc-isolated, which would let attention leak across
# docs. With B=1 + attention_mask=None + position_ids resets,
# `create_causal_mask` (transformers/masking_utils.py:700) detects the packed
# format and the FA2 path uses `cu_seqlens_q/k` from the resets.
#
# Symmetric-delay layout per doc:
#   each utterance of length L_i contributes T_i = L_i + delay tokens:
#       [w_0..w_{L_i-1}, EOS, PAD, ..., PAD]    (delay-1 trailing PADs)
#   with position_ids 0..T_i-1 then resetting for the next doc. The model's
#   _delay_shift_audio sees `position_ids < delay` and emits pad_audio_unit_embed
#   at those (per-doc) prefix positions, so each doc's audio stream is
#   isolated even though everything is one long packed B=1 row.
#
# Per-doc loss masks are pre-baked here:
#   text_label_mask True over [0, L_i-1] (the prediction positions), False at
#       the EOS slot itself and at trailing PADs.
#   aco_label_mask  True over [delay-1, L_i+delay-2].
# --------------------------------------------------------------------------- #
class PackedSubwordUnitsDataset(IterableDataset):
    """Greedily packs utterances up to ``pack_max_tokens`` per emitted sample.

    Parameters mirror ``SubwordUnitsDataset``. Sharding for DDP/workers is
    handled in ``__iter__`` via ``rank``/``world`` + torch worker info; pass a
    ``DataLoader(..., batch_size=1, sampler=None)`` to consume.

    Each yielded sample is a dict with concatenated 1-D tensors:
        subword_ids      (T,)     long   per-doc extended with [EOS, PAD*(D-1)]
        q_codes          (T, R)   long   trailing slots zero-padded
        duration_frames  (T,)     long
        position_ids     (T,)     long   resets to 0 at each utt boundary
        attention_mask   (T,)     bool   True over real + EOS; False at PAD
        text_label_mask  (T,)     bool   per-doc [0, L_i-1]
        aco_label_mask   (T,)     bool   per-doc [delay-1, L_i+delay-2]
        lengths          (Nd,)    long   per-doc L_i (Nd = number of utts)
        pre_quant_feat   (T, F)   bf16   only if return_pre_quant_feat=True

    ``pack_max_tokens`` is enforced AGAINST the extended length (L_i + delay)
    so the per-doc footprint matches what the model will actually see.
    """

    def __init__(
        self,
        manifests,
        min_subwords: int = 4,
        max_subwords: int = 1024,
        pack_max_tokens: int = 4096,
        return_pre_quant_feat: bool = False,
        seed: int = 0,
        world: int = 1,
        rank: int = 0,
        delay: int = 1,
        eos_token_id: int = 0,
        pad_token_id: int = 0,
    ):
        if isinstance(manifests, (str, os.PathLike)):
            manifests = [manifests]
        manifest_files = _expand_manifests(manifests)
        if not manifest_files:
            raise FileNotFoundError(f"no manifests resolved from: {manifests}")

        self.units_paths = _load_units_paths(manifest_files, min_subwords, max_subwords)
        if not self.units_paths:
            raise RuntimeError(
                f"no utterances passed length filter [{min_subwords}, {max_subwords}] "
                f"across {len(manifest_files)} manifests"
            )
        # Budget check uses the *extended* per-utt length.
        delay = max(1, int(delay))
        if pack_max_tokens < max_subwords + delay:
            raise ValueError(
                f"pack_max_tokens ({pack_max_tokens}) < "
                f"max_subwords + delay ({max_subwords + delay}); "
                "every utt would emit alone"
            )
        self.max_subwords = max_subwords
        self.pack_max_tokens = pack_max_tokens
        self.return_pre_quant_feat = return_pre_quant_feat
        self.seed = seed
        self.world = max(1, world)
        self.rank = rank
        self.delay = delay
        self.eos_token_id = int(eos_token_id)
        self.pad_token_id = int(pad_token_id)

    def num_utterances(self) -> int:
        """Total utts in this dataset (across all ranks)."""
        return len(self.units_paths)

    def __iter__(self):
        wi = get_worker_info()
        nw = wi.num_workers if wi is not None else 1
        wid = wi.id if wi is not None else 0
        global_id = self.rank * nw + wid
        global_size = self.world * nw

        # Per-worker epoch counter: each call to __iter__ (epoch boundary)
        # advances the shuffle so we don't replay the same order forever.
        # Different ranks share the same shuffle (so the global ordering is
        # consistent) and pick disjoint strides via global_id::global_size.
        epoch = getattr(self, "_epoch", 0)
        self._epoch = epoch + 1

        rng = random.Random(self.seed + epoch)
        idxs = list(range(len(self.units_paths)))
        rng.shuffle(idxs)
        idxs = idxs[global_id::global_size]

        cur: List = []
        cur_T = 0
        D = self.delay
        for i in idxs:
            try:
                u = SubwordUnits.load(self.units_paths[i])
            except Exception as e:
                # bad files are rare but fatal in iter -- skip + continue
                print(f"[packed] skip {self.units_paths[i]}: {e!r}", flush=True)
                continue
            n = min(int(u.num_subwords), self.max_subwords)
            T_i = n + D
            if self.return_pre_quant_feat and u.pre_quant_feat is None:
                raise RuntimeError(
                    f"pre_quant_feat is missing from {self.units_paths[i]}; "
                    "use a cache root extracted with --save_pre_quant_feat."
                )
            if T_i > self.pack_max_tokens:
                # one utt larger than the budget -- emit alone (truncated to n)
                yield self._pack([(u, n)])
                continue
            if cur_T + T_i > self.pack_max_tokens and cur:
                yield self._pack(cur)
                cur = []
                cur_T = 0
            cur.append((u, n))
            cur_T += T_i
        if cur:
            yield self._pack(cur)

    def _pack(self, items):
        D = self.delay
        R = items[0][0].q_codes.shape[1]
        T_total = sum(n + D for _, n in items)

        sw = torch.full((T_total,), self.pad_token_id, dtype=torch.long)
        qc = torch.zeros((T_total, R), dtype=torch.long)
        df = torch.zeros((T_total,), dtype=torch.long)
        pi = torch.zeros((T_total,), dtype=torch.long)
        attn = torch.zeros((T_total,), dtype=torch.bool)
        text_mask = torch.zeros((T_total,), dtype=torch.bool)
        aco_mask = torch.zeros((T_total,), dtype=torch.bool)
        lens = []
        if self.return_pre_quant_feat:
            F_dim = items[0][0].pre_quant_feat.shape[1]
            feat = torch.zeros((T_total, F_dim), dtype=torch.bfloat16)
        else:
            feat = None

        cursor = 0
        for u, n in items:
            T_i = n + D
            sw[cursor : cursor + n] = u.subword_ids[:n]
            sw[cursor + n] = self.eos_token_id        # EOS at slot L_i within doc
            qc[cursor : cursor + n] = u.q_codes[:n]
            df[cursor : cursor + n] = u.duration_frames[:n]
            pi[cursor : cursor + T_i] = torch.arange(T_i, dtype=torch.long)
            attn[cursor : cursor + n + 1] = True
            text_mask[cursor : cursor + n] = True
            aco_mask[cursor + D - 1 : cursor + n + D - 1] = True
            if feat is not None:
                feat[cursor : cursor + n] = u.pre_quant_feat[:n]
            lens.append(n)
            cursor += T_i

        out = {
            "subword_ids":     sw,
            "q_codes":         qc,
            "duration_frames": df,
            "position_ids":    pi,
            "attention_mask":  attn,
            "text_label_mask": text_mask,
            "aco_label_mask":  aco_mask,
            "lengths":         torch.tensor(lens, dtype=torch.long),
        }
        if feat is not None:
            out["pre_quant_feat"] = feat
        return out


@dataclass
class PackingCollator:
    """Trivial collator for packed B=1 samples: unsqueeze the batch dim.

    The DataLoader must be configured with ``batch_size=1`` so each call
    receives exactly one packed sample.
    """

    def __call__(self, samples):
        if len(samples) != 1:
            raise ValueError(
                f"PackingCollator expects batch_size=1, got {len(samples)} samples"
            )
        s = samples[0]
        out = {
            "subword_ids":     s["subword_ids"].unsqueeze(0),
            "q_codes":         s["q_codes"].unsqueeze(0),
            "duration_frames": s["duration_frames"].unsqueeze(0),
            "position_ids":    s["position_ids"].unsqueeze(0),
            "attention_mask":  s["attention_mask"].unsqueeze(0),
            "text_label_mask": s["text_label_mask"].unsqueeze(0),
            "aco_label_mask":  s["aco_label_mask"].unsqueeze(0),
            "lengths":         s["lengths"],
        }
        if "pre_quant_feat" in s:
            out["pre_quant_feat"] = s["pre_quant_feat"].unsqueeze(0)
        return out
