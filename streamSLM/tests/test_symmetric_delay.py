r"""Indexing test for the symmetric (MusicGen-style) delay refactor.

Pins the wire-level convention the rest of the refactor must implement.

Convention recap (N = number of real subwords, D = delay >= 1):

    stored text  : [w_0, ..., w_{N-1}, EOS, PAD, ..., PAD]   length L_i+D = N+D
    stored q     : [q_0, ...,  q_{N-1},   0,   0,  ...,  0]
    stored d     : [d_0, ...,  d_{N-1},   0,   0,  ...,  0]
    attn_mask    : [   1,           1,    1,   0,  ...,  0]   # EOS counted as real
    text_mask    : [   1,           1,    0,   0,  ...,  0]   # prediction positions
                   #                ^ True at [0, L_i-1]; predicts EOS via HF shift
    aco_mask     : [   0,  1, ..., 1,     1,   0,  ...,  0]   # True at [D-1, L_i+D-2]

After the model's _delay_shift_audio (delay=D) on the *extended* sequence:

    delayed audio input  : [PAD, ..., PAD, q_0, q_1, ..., q_{N-1}]
                            \_____D_____/

For training targets:

    text       : HF causal shift  -> position n predicts subword_ids[n+1]
                 With text_mask True at [0, L_i-1]:
                   - position L_i-1 predicts subword_ids[L_i] = EOS         (supervised)
                   - position L_i predicts PAD                              (IGNORE)
                 So EOS is supervised even though it lives at slot L_i.
    acoustic   : aco_targets = F.pad(q_codes, (0,0,D-1,0))[:, :T, :]
                 valid positions n in [D-1, L_i+D-2] predict q_0..q_{N-1}.

For inference (greedy/sampled AR), with prompt length T_p and delay D:

    iteration loop body (L := len(w) at the start of the iteration):
        1) feed w (length L) to the model -> logits at positions 0..L-1
        2) sample w_next (predicted by position L-1; target = w_L)
           sample q_next (predicted by position L-1; target = q_{L-D})
           sample d_next (predicted by position L-1; target = d_{L-D})
        3) if L >= max(D, n_real_q + D):                 # not in real-prompt region
               q[L-D] = q_next ; d[L-D] = d_next         # backfill
        4) append w_next, 0, 0 to w, q, d                # new len = L+1
        5) if EOS just sampled: n_eos_pos := L           # slot where EOS lands
        6) stop:
             EOS path : len(w) >= n_eos_pos + D
             no-EOS   : len(w) >= T_p + max_new_tokens + D
        7) trim:
             EOS path : end = n_eos_pos      # keeps slots 0..N-1, drops EOS + tail
             no-EOS   : end = len(w) - D     # keeps T_p + max_new_tokens slots

The last real audio q_{N-1} is produced (predicted) at iteration L = N + D - 1
because that position's acoustic-head target is q_{L - D} = q_{N-1}. To run
that iteration, the stop check must wait until at least one *more* append
brings len(w) up to N + D. Hence the EOS stop is `len(w) >= n_eos_pos + D`
(equivalent to top-of-loop `T_in >= n_eos_pos + D` with T_in measured pre-feed).

NOTE — old asymmetric-delay checkpoints are NOT compatible with the symmetric
implementation: under asymmetric delay, position n predicted q_n directly,
and at inference the head's just-sampled q_n was written into slot n (which
in this convention would be q[L-1] not q[L-D]). Retraining is required.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from streamSLM.units import SubwordUnits


# --------------------------------------------------------------------------- #
# Constants for the canonical N=3, D=2 test setup
# --------------------------------------------------------------------------- #
N = 3                  # number of real subwords (w_0, w_1, w_2)
D = 2                  # delay
L = N + D              # extended sequence length = 5
PAD_W = 0              # text pad id (collator pad)
EOS_W = 99             # eos token id
W = [11, 12, 13]       # symbolic w_0, w_1, w_2
Q = [
    [101, 201],
    [102, 202],
    [103, 203],
]                      # q_0, q_1, q_2 with R=2 codebooks
DUR = [4, 5, 6]        # d_0, d_1, d_2


# --------------------------------------------------------------------------- #
# Reference helpers — these encode the spec, mirroring what the refactored
# collator + model + generate loop must produce.
# --------------------------------------------------------------------------- #
def _ref_extend_sample(subword_ids: torch.Tensor,
                       q_codes: torch.Tensor,
                       duration_frames: torch.Tensor,
                       delay: int,
                       eos_token_id: int,
                       pad_token_id: int = 0):
    """Single-sample reference for the symmetric-delay collator extension.

    Returns the same dict layout the new PadCollator must emit for a batch
    of one (modulo the leading batch dim added by the collator).
    """
    L_i = int(subword_ids.numel())
    R = q_codes.shape[-1]
    T = L_i + delay

    sw = torch.full((T,), pad_token_id, dtype=torch.long)
    qc = torch.zeros((T, R), dtype=torch.long)
    df = torch.zeros((T,), dtype=torch.long)
    sw[:L_i] = subword_ids
    sw[L_i] = eos_token_id
    qc[:L_i] = q_codes
    df[:L_i] = duration_frames

    attn = torch.zeros((T,), dtype=torch.bool)
    attn[: L_i + 1] = True            # real + EOS slot; trailing PADs are 0

    text_mask = torch.zeros((T,), dtype=torch.bool)
    text_mask[: L_i] = True           # prediction positions [0, L_i-1]

    aco_mask = torch.zeros((T,), dtype=torch.bool)
    aco_mask[delay - 1 : L_i + delay - 1] = True   # [D-1, L_i+D-2]

    return dict(
        subword_ids=sw,
        q_codes=qc,
        duration_frames=df,
        attention_mask=attn,
        text_label_mask=text_mask,
        aco_label_mask=aco_mask,
        length=L_i,
    )


# --------------------------------------------------------------------------- #
# 1. Stored layout: text, q, durations, masks.
# --------------------------------------------------------------------------- #
def test_stored_layout_N3_D2():
    sw = torch.tensor(W, dtype=torch.long)
    qc = torch.tensor(Q, dtype=torch.long)
    df = torch.tensor(DUR, dtype=torch.long)
    out = _ref_extend_sample(sw, qc, df, delay=D, eos_token_id=EOS_W,
                             pad_token_id=PAD_W)

    # stored text  : [w_0, w_1, w_2, EOS, PAD]
    assert out["subword_ids"].tolist() == [W[0], W[1], W[2], EOS_W, PAD_W]
    # stored q     : [q_0, q_1, q_2, 0, 0]
    assert out["q_codes"].tolist() == [Q[0], Q[1], Q[2], [0, 0], [0, 0]]
    # stored d     : [d_0, d_1, d_2, 0, 0]
    assert out["duration_frames"].tolist() == [DUR[0], DUR[1], DUR[2], 0, 0]
    # attention_mask True over real + EOS, False at trailing PAD slots
    assert out["attention_mask"].tolist() == [True] * (N + 1) + [False] * (D - 1)
    # text_label_mask True at prediction positions [0, L_i-1] only.
    # Position L_i-1=2 therefore is the (last) text prediction position; its
    # HF-shifted target is subword_ids[L_i]=EOS, so EOS is supervised.
    assert out["text_label_mask"].tolist() == [True, True, True, False, False]
    # aco_label_mask True at acoustic-target positions [D-1, L_i+D-2] = [1,2,3]
    assert out["aco_label_mask"].tolist() == [False, True, True, True, False]


# --------------------------------------------------------------------------- #
# 2. Delay-shift on the extended sequence.
# Mirrors StreamSLM._delay_shift_audio: front-pad by D with a learned pad
# vector. With T = L_i + D and audio_e length T, the model slices
# audio_e[:, :T-D] which keeps the real codes only.
# --------------------------------------------------------------------------- #
def test_delay_shift_indexing():
    audio_dim = 4
    R = 2
    # Build a fake audio_e tensor where each slot's vector is uniquely
    # identifiable (entry encodes the slot index along T).
    T = L
    audio_e = torch.stack([torch.full((audio_dim,), float(i)) for i in range(T)])  # (T, D_a)
    audio_e = audio_e.unsqueeze(0)  # (1, T, D_a)

    pad_vec = torch.full((audio_dim,), -1.0)
    pref = pad_vec.view(1, 1, audio_dim).expand(1, D, audio_dim)
    shifted = torch.cat([pref, audio_e[:, : T - D]], dim=1)  # (1, T, D_a)

    # delayed audio input  : [PAD, PAD, e_q0, e_q1, e_q2]
    expected = [
        [-1.0] * audio_dim,
        [-1.0] * audio_dim,
        [0.0] * audio_dim,
        [1.0] * audio_dim,
        [2.0] * audio_dim,
    ]
    assert shifted.squeeze(0).tolist() == expected


# --------------------------------------------------------------------------- #
# 3. Text prediction positions (HF causal shift + text_label_mask).
# --------------------------------------------------------------------------- #
def test_text_prediction_positions_and_eos_supervision():
    out = _ref_extend_sample(
        torch.tensor(W, dtype=torch.long),
        torch.tensor(Q, dtype=torch.long),
        torch.tensor(DUR, dtype=torch.long),
        delay=D, eos_token_id=EOS_W, pad_token_id=PAD_W,
    )
    IGNORE = -100
    text_mask_full = out["text_label_mask"]  # (T,)
    labels_full = out["subword_ids"]         # (T,)

    # HF shift: position n predicts labels[n+1]. Effective labels for the
    # loss are labels[1:], gated by text_mask[:-1] (mask defined on
    # prediction positions, not stored-label positions).
    shift_labels = labels_full[1:].clone()
    shift_mask = text_mask_full[:-1]
    shift_labels = torch.where(shift_mask, shift_labels,
                               torch.full_like(shift_labels, IGNORE))

    # Effective per-position targets (length T-1 = N+D-1 = 4):
    #   pos 0 predicts subword_ids[1] = w_1   (supervised)
    #   pos 1 predicts subword_ids[2] = w_2   (supervised)
    #   pos 2 predicts subword_ids[3] = EOS   (supervised)
    #   pos 3 predicts subword_ids[4] = PAD   (IGNORE -- this is the key:
    #                                          text_label_mask is False here)
    assert shift_labels.tolist() == [W[1], W[2], EOS_W, IGNORE]

    # The set of supervised prediction positions is exactly {0, 1, 2}.
    supervised_positions = (shift_labels != IGNORE).nonzero(as_tuple=True)[0].tolist()
    assert supervised_positions == [0, 1, 2]


# --------------------------------------------------------------------------- #
# 4. Acoustic target alignment + mask.
# --------------------------------------------------------------------------- #
def test_acoustic_target_alignment():
    out = _ref_extend_sample(
        torch.tensor(W, dtype=torch.long),
        torch.tensor(Q, dtype=torch.long),
        torch.tensor(DUR, dtype=torch.long),
        delay=D, eos_token_id=EOS_W, pad_token_id=PAD_W,
    )
    qc_full = out["q_codes"]      # (T, R)
    aco_mask = out["aco_label_mask"]  # (T,)
    R = qc_full.shape[-1]
    T = qc_full.shape[0]

    # Right-shift by D-1 to align targets:
    #   aco_targets[n] = q_codes[n - (D-1)]    for n >= D-1
    aco_targets = F.pad(qc_full.unsqueeze(0), (0, 0, D - 1, 0))[:, :T, :].squeeze(0)
    assert aco_targets.tolist() == [
        [0, 0],     # pos 0 (masked)
        Q[0],       # pos 1 predicts q_0
        Q[1],       # pos 2 predicts q_1
        Q[2],       # pos 3 predicts q_2
        [0, 0],     # pos 4 (masked; q at stored slot 3 is zero)
    ]
    # Mask True exactly at the slots predicting real codes.
    assert aco_mask.tolist() == [False, True, True, True, False]

    # The expected set of acoustic predictions, per mask:
    pred_slots = aco_mask.nonzero(as_tuple=True)[0].tolist()
    assert pred_slots == [1, 2, 3]
    # And targets at those slots are exactly q_0, q_1, q_2:
    assert aco_targets[pred_slots].tolist() == [Q[0], Q[1], Q[2]]


# --------------------------------------------------------------------------- #
# 5. Inference loop — drives a deterministic mock model and verifies the
# symmetric backfill + stop + trim semantics return exactly N aligned items.
#
# The mock returns predetermined w_next / q_next / d_next per iteration so the
# trim/backfill arithmetic is observable from the output values.
# --------------------------------------------------------------------------- #
class _MockSymmetricGenerator:
    """Drives the symmetric AR loop without a real model.

    Implements the same control flow the new generate.py must implement.
    The model is replaced by an oracle that yields scripted samples.
    """

    def __init__(self, delay: int, eos_id: int):
        self.delay = delay
        self.eos_id = eos_id

    def run(self,
            prompt_ids: torch.Tensor,
            prompt_q: Optional[torch.Tensor],
            prompt_d: Optional[torch.Tensor],
            scripted_w: List[int],
            scripted_q: List[List[int]],
            scripted_d: List[int],
            max_new_tokens: int) -> SubwordUnits:
        D = self.delay
        eos = self.eos_id
        R = scripted_q[0].__len__() if scripted_q else 0

        if prompt_q is None:
            w = prompt_ids[:1].clone()
            q = torch.zeros(1, R, dtype=torch.long)
            d = torch.zeros(1, dtype=torch.long)
            n_real_q = 0
            T_p = int(prompt_ids.numel())
        else:
            w = prompt_ids.clone()
            q = prompt_q.clone()
            d = prompt_d.clone()
            n_real_q = int(prompt_ids.numel())
            T_p = n_real_q

        n_eos_pos: Optional[int] = None
        step_idx = 0
        text_only_prompt = (prompt_q is None)

        while True:
            L_now = int(w.numel())
            # If a text-only prompt is in flight, the loop force-feeds the
            # remaining prompt tokens (same behaviour as generate.py).
            force_feed = text_only_prompt and L_now < T_p
            if force_feed:
                w_next = int(prompt_ids[L_now].item())
                q_next = [0] * R
                d_next = 0
            else:
                w_next = int(scripted_w[step_idx])
                q_next = list(scripted_q[step_idx])
                d_next = int(scripted_d[step_idx])
                step_idx += 1

            # Symmetric backfill: logits at position L_now-1 predict q_{L_now-D}.
            backfill_idx = L_now - D
            if backfill_idx >= n_real_q and backfill_idx >= 0:
                q[backfill_idx] = torch.tensor(q_next, dtype=torch.long)
                d[backfill_idx] = d_next

            # Append the new w / placeholder q,d so len grows by 1.
            w = torch.cat([w, torch.tensor([w_next], dtype=torch.long)])
            q = torch.cat([q, torch.zeros(1, R, dtype=torch.long)], dim=0)
            d = torch.cat([d, torch.zeros(1, dtype=torch.long)])

            if (not force_feed) and w_next == eos and n_eos_pos is None:
                n_eos_pos = L_now    # EOS just landed at slot L_now

            # Bottom-of-loop stop check:
            #   EOS path  : continue D-1 more iterations after EOS so the
            #               last backfill (q_{N-1}, d_{N-1}) runs, then stop.
            #               That's `len(w) >= n_eos_pos + D`.
            #   no-EOS    : keep producing until we can trim to T_p+max_new.
            if n_eos_pos is not None:
                if int(w.numel()) >= n_eos_pos + D:
                    break
            else:
                if int(w.numel()) >= T_p + max_new_tokens + D:
                    break

        if n_eos_pos is not None:
            end = n_eos_pos
        else:
            end = int(w.numel()) - D

        return SubwordUnits(
            subword_ids=w[:end].clone(),
            q_codes=q[:end].clone(),
            duration_frames=d[:end].clone(),
        )


def test_inference_backfill_returns_n_aligned_items_with_eos():
    """Text-only prompt = [w_0]. EOS sampled at iteration L=N+1 with N=3.

    After the EOS-sampling iteration, D-1=1 more iteration runs to backfill
    q_{N-1}=q_2; then the loop stops at len(w)=N+D=5 and trims to N=3.
    """
    gen = _MockSymmetricGenerator(delay=D, eos_id=EOS_W)

    # The first iter (L=1) predicts w_1 + q_{-1} (invalid backfill).
    # Subsequent iters predict (w_2, q_0), (EOS, q_1), (junk_w, q_2).
    scripted_w = [W[1], W[2], EOS_W, 9999]
    scripted_q = [Q[0], Q[0], Q[1], Q[2]]    # q_{-1} unused; rest = q_0,q_1,q_2
    scripted_d = [0,     DUR[0], DUR[1], DUR[2]]

    units = gen.run(
        prompt_ids=torch.tensor([W[0]], dtype=torch.long),
        prompt_q=None,
        prompt_d=None,
        scripted_w=scripted_w,
        scripted_q=scripted_q,
        scripted_d=scripted_d,
        max_new_tokens=10,
    )

    assert units.subword_ids.numel() == N
    assert units.q_codes.shape == (N, 2)
    assert units.duration_frames.numel() == N

    # Exactly the N aligned (w_n, q_n, d_n) tuples the spec promises.
    assert units.subword_ids.tolist() == [W[0], W[1], W[2]]
    assert units.q_codes.tolist() == [Q[0], Q[1], Q[2]]
    assert units.duration_frames.tolist() == [DUR[0], DUR[1], DUR[2]]


def test_inference_no_eos_trims_by_delay():
    """No-EOS path: stop at len(w) = T_p + max_new + D, trim end = len(w) - D."""
    gen = _MockSymmetricGenerator(delay=D, eos_id=EOS_W)

    # Prompt T_p=1, request max_new_tokens=2. Total iters needed:
    #   L=1 -> append (no backfill yet)
    #   L=2 -> backfill q[0]
    #   L=3 -> backfill q[1]
    #   L=4 -> backfill q[2]; after append, len(w)=5=T_p+max_new+D -> stop.
    # Trim end = 5 - D = 3 = T_p + max_new_tokens.
    scripted_w = [W[1], W[2], 7777, 8888]
    scripted_q = [Q[0], Q[0], Q[1], Q[2]]    # iter-1 q is unused (backfill skipped)
    scripted_d = [0,     DUR[0], DUR[1], DUR[2]]

    units = gen.run(
        prompt_ids=torch.tensor([W[0]], dtype=torch.long),
        prompt_q=None,
        prompt_d=None,
        scripted_w=scripted_w,
        scripted_q=scripted_q,
        scripted_d=scripted_d,
        max_new_tokens=2,
    )

    assert units.subword_ids.numel() == 1 + 2  # T_p + max_new_tokens
    assert units.subword_ids.tolist() == [W[0], W[1], W[2]]
    assert units.q_codes.tolist() == [Q[0], Q[1], Q[2]]
    assert units.duration_frames.tolist() == [DUR[0], DUR[1], DUR[2]]


def test_inference_speech_prompt_skips_real_q_region():
    """Speech-prompt (prompt_q given) must NOT overwrite real q's during
    the first D iterations (slots inside [0, T_p-1])."""
    gen = _MockSymmetricGenerator(delay=D, eos_id=EOS_W)

    # T_p=2 real prompt: (w_0, q_0, d_0) and (w_1, q_1, d_1).
    # We then sample one new subword (max_new_tokens=1) with EOS at the next
    # step. Expected aligned output: 2 prompt + 1 new = 3 items.
    prompt_w = torch.tensor([W[0], W[1]], dtype=torch.long)
    prompt_q = torch.tensor([Q[0], Q[1]], dtype=torch.long)
    prompt_d = torch.tensor([DUR[0], DUR[1]], dtype=torch.long)

    # Iteration trace (D=2, T_p=2, n_real_q=2):
    #   L=2 -> predicts w_2 + q_{0}; backfill_idx=0 < n_real_q -> SKIP backfill
    #   L=3 -> predicts EOS + q_{1}; backfill_idx=1 < n_real_q -> SKIP backfill
    #          n_eos_pos = 3
    #   L=4 -> predicts (junk_w) + q_{2}; backfill_idx=2 >= n_real_q -> backfill q[2]
    #          after append len(w)=5 >= n_eos_pos+D=5 -> stop.
    #   trim end = n_eos_pos = 3.
    scripted_w = [W[2], EOS_W, 7777]
    scripted_q = [Q[0], Q[0], Q[2]]
    scripted_d = [DUR[0], DUR[0], DUR[2]]

    units = gen.run(
        prompt_ids=prompt_w,
        prompt_q=prompt_q,
        prompt_d=prompt_d,
        scripted_w=scripted_w,
        scripted_q=scripted_q,
        scripted_d=scripted_d,
        max_new_tokens=10,
    )

    assert units.subword_ids.tolist() == [W[0], W[1], W[2]]
    # First two q's came from the prompt (not from scripted_q backfills),
    # confirming the real-prompt region was not overwritten.
    assert units.q_codes.tolist() == [Q[0], Q[1], Q[2]]
    assert units.duration_frames.tolist() == [DUR[0], DUR[1], DUR[2]]


if __name__ == "__main__":
    test_stored_layout_N3_D2()
    test_delay_shift_indexing()
    test_text_prediction_positions_and_eos_supervision()
    test_acoustic_target_alignment()
    test_inference_backfill_returns_n_aligned_items_with_eos()
    test_inference_no_eos_trims_by_delay()
    test_inference_speech_prompt_skips_real_q_region()
    print("test_symmetric_delay: OK")
