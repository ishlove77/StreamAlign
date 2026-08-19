"""Constrained RNN-T alignment using chunk-level TextGrid waypoints.

Each wav file in the dataset has a corresponding TextGrid in chunk_textgrids_word_model.
The TextGrid's "words" tier has intervals with xmin/xmax in encoder-frame units
(1 frame ≈ 40 ms after CNN subsampling).  Each interval constrains which characters
the RNNT path may emit within those frames.

The constrained Viterbi finds the maximum-likelihood RNNT path that respects these
waypoints.  No gradient is computed – the result is used as an alignment signal.
"""

import re
import torch
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# TextGrid parsing
# ---------------------------------------------------------------------------

def parse_textgrid_words(
    path: str,
    frame_rate: Optional[float] = None,
) -> List[Tuple[int, int, str]]:
    """Parse the 'words' tier of a Praat TextGrid.

    Two TextGrid conventions are supported:

    - ``frame_rate=None`` (default) — xmin/xmax are already integer-valued
      encoder-frame indices stored as floats (e.g. ``8.000000``). This is
      the chunk_textgrids_word_model_final2 convention used by all
      streaming-ASR-aligned corpora in this repo (LibriSpeech, LibriTTS,
      Emilia).
    - ``frame_rate=<Hz>`` — xmin/xmax are wall-clock seconds (e.g.
      ``0.530``); they are multiplied by ``frame_rate`` and floored to
      produce encoder-frame indices. Legacy MFA outputs use this format,
      but the project no longer consumes them.

    Returns
    -------
    list of (xmin_frame, xmax_frame, text)
        Only the *words* tier is returned.  Empty-text intervals are included so
        that the caller can reason about silent/blank regions.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Locate the first IntervalTier named "words"
    tier_blocks = re.split(r'item\s*\[\d+\]\s*:', content)
    words_block = None
    for block in tier_blocks:
        if '"words"' in block:
            words_block = block
            break
    if words_block is None:
        return []

    # Extract all intervals from that tier
    interval_pattern = re.compile(
        r'xmin\s*=\s*([\d.eE+\-]+)\s*\n\s*xmax\s*=\s*([\d.eE+\-]+)\s*\n\s*text\s*=\s*"([^"]*)"',
        re.MULTILINE,
    )
    scale = 1.0 if frame_rate is None else float(frame_rate)
    intervals = []
    for m in interval_pattern.finditer(words_block):
        xmin = int(float(m.group(1)) * scale)
        xmax = int(float(m.group(2)) * scale)
        text = m.group(3)
        intervals.append((xmin, xmax, text))

    return intervals


# ---------------------------------------------------------------------------
# Constraint building
# ---------------------------------------------------------------------------

def build_char_constraints(
    intervals: List[Tuple[int, int, str]],
    char_to_idx: dict,
) -> Tuple[List[int], List[Tuple[int, int, int, int]]]:
    """Convert TextGrid intervals into character-level RNNT waypoint constraints.

    Parameters
    ----------
    intervals : list of (t_start, t_end, text)
    char_to_idx : mapping from character to vocab index (lowercase keys expected)

    Returns
    -------
    full_char_ids : list[int]
        Character IDs for the full text (concatenation of all non-empty intervals).
    constraints : list of (t_start, t_end, u_start, u_end)
        For encoder frames in ``[t_start, t_end)``, only characters with index in
        ``[u_start, u_end)`` may be emitted by the RNNT path.
        Intervals with empty text are not included; those frames are blank-only.
    """
    full_char_ids: List[int] = []
    constraints: List[Tuple[int, int, int, int]] = []

    for t_start, t_end, text in intervals:
        # NOTE: keep a leading-space if present. The streaming-ASR TextGrid uses
        # a leading " " on each interval text to mark "this chunk commits the
        # start of a new word"; intervals without a leading space continue the
        # previous word (e.g. " WAS TURN" then "ERS" -> "WAS TURNERS"). Calling
        # .strip() here would erase that boundary signal, causing downstream
        # tokenization to treat every interval as its own word.
        text_lower = text.lower().rstrip()
        if not text_lower.strip():
            continue

        char_ids = [char_to_idx[ch] for ch in text_lower if ch in char_to_idx]
        if not char_ids:
            continue

        u_start = len(full_char_ids)
        u_end = u_start + len(char_ids)
        full_char_ids.extend(char_ids)
        constraints.append((t_start, t_end, u_start, u_end))

    return full_char_ids, constraints


# ---------------------------------------------------------------------------
# Teacher-forced joint log-probs
# ---------------------------------------------------------------------------

def _rnnt_joint_forward_batch(
    hparams: dict,
    enc_out: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Batched RNN-T joint forward pass.

    Parameters
    ----------
    hparams : dict
        Hparams dict with ``emb``, ``dec``, ``proj_dec``, ``Tjoint``,
        ``transducer_lin``, ``log_softmax``.
    enc_out : torch.Tensor
        Encoder outputs ``[B, T, D]``.
    input_ids : torch.Tensor
        Teacher-forced predictor inputs ``[B, U_max+1]`` (int32, blank-padded).

    Returns
    -------
    log_probs : torch.Tensor
        ``[B, T, U_max+1, V]`` in log-probability space.
    """
    emb_out  = hparams["emb"](input_ids)           # [B, U_max+1, emb_dim]
    dec_out, _ = hparams["dec"](emb_out)            # [B, U_max+1, 512]
    dec_proj = hparams["proj_dec"](dec_out)         # [B, U_max+1, 640]

    enc_exp = enc_out.unsqueeze(2)                  # [B, T, 1, 640]
    dec_exp = dec_proj.unsqueeze(1)                 # [B, 1, U_max+1, 640]

    joint   = hparams["Tjoint"](enc_exp, dec_exp)   # [B, T, U_max+1, 640]
    logits  = hparams["transducer_lin"](joint)      # [B, T, U_max+1, V]
    return   hparams["log_softmax"](logits)         # [B, T, U_max+1, V]


# ---------------------------------------------------------------------------
# Constrained Viterbi DP
# ---------------------------------------------------------------------------

def constrained_viterbi_single(
    log_probs: torch.Tensor,
    char_ids: List[int],
    constraints: List[Tuple[int, int, int, int]],
    blank_id: int,
) -> Tuple[List[int], float]:
    """Constrained Viterbi alignment for a single utterance.

    The path must obey: within encoder frames ``[t_s, t_e)`` only characters with
    label index in ``[u_s, u_e)`` may be emitted.  Frames not covered by any
    constraint interval only allow blank emissions.

    Parameters
    ----------
    log_probs : torch.Tensor
        Shape ``[T, U+1, V]``.
    char_ids : list[int]
        Target character IDs, length U.
    constraints : list of (t_start, t_end, u_start, u_end)
    blank_id : int

    Returns
    -------
    frame_positions : list[int]
        Length U.  ``frame_positions[i]`` is the encoder frame at which character
        ``i`` was emitted along the optimal constrained path (same format expected
        by ``_build_alignment_from_rnnt``).
    path_logprob : float
        Total log-probability of the optimal path.
    """
    T = log_probs.shape[0]
    U = len(char_ids)

    NEG_INF = float("-inf")

    # valid_emit[t] = set of char indices u that may be emitted at encoder frame t
    valid_emit: List[set] = [set() for _ in range(T)]
    for t_s, t_e, u_s, u_e in constraints:
        t_s_clamped = min(max(0, t_s), T - 1)
        t_e_clamped = max(min(t_e, T), t_s_clamped + 1)
        for t in range(t_s_clamped, t_e_clamped):
            for u in range(max(0, u_s), min(U, u_e)):
                valid_emit[t].add(u)

    # DP table and backpointer
    # dp[t][u]: max log-prob of any path from (0,0) to state (t frames via blank, u labels emitted)
    dp = [[NEG_INF] * (U + 1) for _ in range(T + 1)]
    # bp[t][u]: 'B' (blank at frame t-1 brought us here) or 'L' (label u-1 at frame t)
    bp = [[""] * (U + 1) for _ in range(T + 1)]

    dp[0][0] = 0.0

    lp = log_probs.cpu()  # avoid repeated .item() GPU round-trips

    for t in range(T):
        for u in range(U + 1):
            cur = dp[t][u]
            if cur == NEG_INF:
                continue

            # Blank: advance time (t, u) → (t+1, u)
            blank_lp = lp[t, u, blank_id].item()
            val = cur + blank_lp
            if val > dp[t + 1][u]:
                dp[t + 1][u] = val
                bp[t + 1][u] = "B"

            # Label u: stay at frame t, advance label (t, u) → (t, u+1)
            if u < U and u in valid_emit[t]:
                label_lp = lp[t, u, char_ids[u]].item()
                val = cur + label_lp
                if val > dp[t][u + 1]:
                    dp[t][u + 1] = val
                    bp[t][u + 1] = "L"

    # Backtrack from (T, U) to recover the per-character frame assignments
    fps: dict = {}  # char_index → frame_t where it was emitted
    t, u = T, U
    while t > 0 or u > 0:
        action = bp[t][u]
        if action == "B":
            t -= 1
        elif action == "L":
            # char u-1 emitted at encoder frame t (before the blank that will advance from t)
            fps[u - 1] = t
            u -= 1
        else:
            # Unreachable state – path is infeasible; return best-effort result
            raise Exception(f"Invalid Viterbi backtrace at (t={t}, u={u})")

    frame_positions = [fps.get(i, 0) for i in range(U)]
    return frame_positions, dp[T][U]


# ---------------------------------------------------------------------------
# Batch alignment entry-point
# ---------------------------------------------------------------------------

@torch.no_grad()
def constrained_rnnt_align(
    enc_out_batch: torch.Tensor,
    enc_feat_lens: torch.Tensor,
    char_ids_list: List[Optional[List[int]]],
    constraints_list: List[Optional[List[Tuple[int, int, int, int]]]],
    hparams: dict,
    blank_id: int,
) -> Tuple[List[List[int]], List[List[int]]]:
    """Run constrained RNNT alignment for an entire batch.

    The RNN-T joint network (emb → dec → proj_dec → Tjoint → transducer_lin →
    log_softmax) is evaluated once for the whole batch, producing
    ``[B_valid, T_max, U_max+1, V]`` in a single GPU call — mirroring how the
    standard transducer loss processes the full batch at once.  Per-sample
    constrained Viterbi DPs are then run on the resulting log-prob slices.

    Samples without TextGrid data (``None`` entries) are skipped.

    Parameters
    ----------
    enc_out_batch : torch.Tensor
        Shape ``[B, T_max, D]``.
    enc_feat_lens : torch.Tensor
        Actual encoder-frame length per sample, shape ``[B]``.
    char_ids_list : list of list[int] or None
        Per-sample character IDs from TextGrid.
    constraints_list : list of list[...] or None
        Per-sample RNNT waypoint constraints.
    hparams : dict
        Loaded hparams dict with RNN-T modules.
    blank_id : int

    Returns
    -------
    predictions : list[list[int]]
        Per-sample character ID sequences (from TextGrid, not decoded).
    frame_positions : list[list[int]]
        Per-sample ``fps`` arrays, one entry per character.
    """
    B = enc_out_batch.size(0)
    device = enc_out_batch.device
    predictions: List[List[int]] = [[] for _ in range(B)]
    frame_positions: List[List[int]] = [[] for _ in range(B)]

    # Gather samples that have valid TextGrid data
    valid_idx = [
        b for b in range(B)
        if char_ids_list[b] is not None
        and constraints_list[b] is not None
        and len(char_ids_list[b]) > 0
    ]
    if not valid_idx:
        return predictions, frame_positions

    # Pad all teacher-forced predictor inputs to [B_valid, U_max+1]
    U_max = max(len(char_ids_list[b]) for b in valid_idx)
    input_ids = torch.full(
        (len(valid_idx), U_max + 1), blank_id, dtype=torch.int32, device=device
    )
    for i, b in enumerate(valid_idx):
        cids = char_ids_list[b]
        input_ids[i, 1:1 + len(cids)] = torch.tensor(
            cids, dtype=torch.int32, device=device
        )

    # Single batched RNN-T forward pass → [B_valid, T_max, U_max+1, V]
    log_probs_batch = _rnnt_joint_forward_batch(
        hparams,
        enc_out_batch[valid_idx],   # [B_valid, T_max, D]
        input_ids,
    )

    # Per-sample constrained Viterbi on pre-computed log-prob slices
    for i, b in enumerate(valid_idx):
        T_actual = int(enc_feat_lens[b].item())
        lp = log_probs_batch[i, :T_actual]    # [T_actual, U_max+1, V]
        char_ids = char_ids_list[b]
        constraints = constraints_list[b]
        if constraints:
            t_s, t_e, u_s, u_e = constraints[-1]
            if t_e < T_actual:
                constraints = constraints[:-1] + [(t_s, T_actual, u_s, u_e)]
        try:
            fps, _ = constrained_viterbi_single(lp, char_ids, constraints, blank_id)
            predictions[b] = list(char_ids)
            frame_positions[b] = fps
        except Exception as exc:
            import warnings
            warnings.warn(f"constrained_rnnt_align failed for sample {b}: {exc}")

    return predictions, frame_positions
