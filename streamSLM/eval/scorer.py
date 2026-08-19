"""StreamSLM scoring for SubwordUnits.

For a sequence of N subwords with R RVQ codebooks of size K, the trained
StreamSLM models the joint:

    log p(w, q) = sum_n log p(w_{n+1} | w_<=n, q_<n, d_<n)        # text head
                + sum_n sum_r log p(q_n_r | h_n[, q_n_<r])         # acoustic head

The duration term is excluded — L1/MSE regression has no proper density.

Two scoring modes are supported, both producing a ``score`` field
(higher = better fit) so the same pos/neg comparison pipeline is reused:

* ``likelihood`` (default): joint log-likelihood with sum reduction over
  valid positions and codebooks::

      score = -(text_NLL_sum + acoustic_NLL_sum)

  Longer sequences naturally have larger magnitudes; this is the
  classical SLM "log-likelihood" used in SALMon / sBLIMP / StoryCloze
  papers.

* ``loss``: TASTE-SpokenLM convention (see
  ``STAGE1_TRAIN/SpokenLM/taslm/modeling_taslm.py::calculate_log_likelihood``).
  Cross-entropy is averaged with the default ``reduction='mean'``,
  computed **independently per stream** and summed::

      score = -(mean_CE(text)
                + sum_{r=0..R-1} mean_CE(acoustic codebook r))

  Each speech channel contributes its own per-token mean, so the score
  is length-normalised. This matches TASTE's
  ``-loss_fct(text_logits, text_label_ids)`` plus the per-channel
  ``-loss_fct(cur_speech_logits, cur_speech_label_ids)`` terms.

Both modes share the single forward pass through ``StreamSLM.forward``
under the **symmetric-delay** convention: an utterance with L real
subwords is extended to T = L + D by appending EOS at slot L followed
by (D-1) PADs (same layout as ``PadCollator``). Text predictions cover
[0, L-1] (position L-1 supervises EOS) and acoustic targets are right-
shifted by D-1 so position n predicts q_{n-(D-1)} for n in
[D-1, L+D-2]. **INCOMPATIBILITY NOTE:** checkpoints trained under the
old asymmetric convention (no EOS, q-targets unshifted) cannot be
scored with this scorer.

The text and acoustic terms are each scaled by ``text_weight`` /
``acoustic_weight`` (both default 1.0) before being summed into
``score``, so the text:acoustic ratio is configurable without
changing the per-stream definitions above. In ``loss`` mode the R
acoustic codebooks may additionally be combined with per-codebook
weights (``acoustic_ch_weights``, e.g. the training
``rvq_loss_weights``); when unset they are summed uniformly.
"""

from __future__ import annotations

import os
import sys
from contextlib import nullcontext as _nullcontext
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

# Match extractor.py — keep sys.path aligned so we can import streamSLM.*.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from streamSLM.config import ModelConfig, TokenizerConfig  # noqa: E402
from streamSLM.model.slm import StreamSLM, load_lm_tokenizer  # noqa: E402
from streamSLM.model.slm_hier import StreamSLMHier  # noqa: E402
from streamSLM.units import SubwordUnits  # noqa: E402


def _build_configs_from_ckpt(saved_args: dict) -> Tuple[TokenizerConfig, ModelConfig]:
    tok_cfg = TokenizerConfig(
        token_type=saved_args.get("token_type", "rvq"),
        rvq_num_quantizers=saved_args.get("rvq_num_quantizers", 16),
        rvq_codebook_size=saved_args.get("rvq_codebook_size", 512),
        text_tokenizer=saved_args.get("text_tokenizer", "llama"),
    )
    dep_ff = saved_args.get("depformer_dim_feedforward", 0)
    model_cfg = ModelConfig(
        backbone=saved_args.get("backbone", "meta-llama/Llama-3.2-1B"),
        delay=saved_args.get("delay", 1),
        duration_loss=saved_args.get("duration_loss", "l1"),
        acoustic_predictor_type=saved_args.get(
            "acoustic_predictor_type", "multihead"
        ),
        depformer_dim=saved_args.get("depformer_dim", 128),
        depformer_num_heads=saved_args.get("depformer_num_heads", 4),
        depformer_num_layers=saved_args.get("depformer_num_layers", 2),
        depformer_dim_feedforward=(dep_ff or None),
        depformer_multi_linear=saved_args.get("depformer_multi_linear", False),
        depformer_weights_per_step=saved_args.get("depformer_weights_per_step", False),
        depformer_norm=saved_args.get("depformer_norm", False),
        fusion_mode=saved_args.get("fusion_mode", "gated_softmax"),
        audio_emb_init_scale=saved_args.get("audio_emb_init_scale", "auto"),
        duration_head_input=saved_args.get("duration_head_input", "llm"),
        duration_loss_type=saved_args.get("duration_loss_type", "regression"),
        duration_num_buckets=saved_args.get("duration_num_buckets", 256),
        feed_duration_to_depformer=saved_args.get("feed_duration_to_depformer", False),
        acoustic_target=saved_args.get("acoustic_target", "rvq"),
        acoustic_feat_dim=saved_args.get("acoustic_feat_dim", 256),
        acoustic_feat_loss=saved_args.get("acoustic_feat_loss", "l1"),
        acoustic_layer_mix=saved_args.get("acoustic_layer_mix", "last"),
        model_arch=saved_args.get("model_arch", "streamslm"),
        hier_ar_order=saved_args.get("hier_ar_order", "duration_last"),
    )
    return tok_cfg, model_cfg


SCORING_MODES = ("likelihood", "loss")


class StreamSLMScorer:
    """Wraps a trained StreamSLM checkpoint and exposes a per-utterance score.

    Parameters
    ----------
    slm_checkpoint : str
        Path to a streamSLM ``.pt`` checkpoint (must contain ``args`` and
        ``model`` keys produced by ``streamSLM.train.train``).
    scoring_mode : {"likelihood", "loss"}, default "likelihood"
        See module docstring. ``likelihood`` returns the joint
        log-likelihood (sum reduction). ``loss`` returns the TASTE-style
        sum of per-stream mean cross-entropy values, length-normalised.
    text_weight, acoustic_weight : float, default 1.0
        Linear weights applied to the text-stream and acoustic-stream
        terms when combining them into ``score`` (both modes). The pair
        score becomes ``-(text_weight * text_term + acoustic_weight *
        acoustic_term)``. The defaults give the original 1:1 combination.
    acoustic_ch_weights : sequence of float, optional
        Per-codebook weights (length R) applied to the acoustic CEs in
        ``loss`` mode before they are summed into the acoustic term.
        ``None`` (default) sums the R codebooks uniformly.
    """

    def __init__(
        self,
        slm_checkpoint: str,
        device: Optional[torch.device] = None,
        bf16: bool = True,
        scoring_mode: str = "likelihood",
        text_weight: float = 1.0,
        acoustic_weight: float = 1.0,
        acoustic_ch_weights: Optional[Sequence[float]] = None,
    ):
        if scoring_mode not in SCORING_MODES:
            raise ValueError(
                f"scoring_mode must be one of {SCORING_MODES}, got {scoring_mode!r}"
            )
        if text_weight < 0.0 or acoustic_weight < 0.0:
            raise ValueError(
                f"text_weight/acoustic_weight must be >= 0, got "
                f"{text_weight!r}/{acoustic_weight!r}"
            )
        self.scoring_mode = scoring_mode
        self.text_weight = float(text_weight)
        self.acoustic_weight = float(acoustic_weight)
        self._acoustic_ch_weights_arg = acoustic_ch_weights
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.bf16 = bf16

        ckpt = torch.load(slm_checkpoint, map_location="cpu", weights_only=False)
        saved_args = ckpt.get("args", {})
        self.tok_cfg, self.model_cfg = _build_configs_from_ckpt(saved_args)

        model_cls = StreamSLMHier if self.model_cfg.model_arch == "streamslm_hier" else StreamSLM
        self.model = model_cls(self.model_cfg, self.tok_cfg).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        self.lm_tok = load_lm_tokenizer(self.tok_cfg)
        self.delay = self.model.delay
        self.R = self.model.R
        self.K = self.model.K
        self.vocab_size = self.model.vocab_size
        self.eos_token_id = int(self.lm_tok.eos_token_id)
        pad_id = self.lm_tok.pad_token_id
        self.pad_token_id = int(pad_id if pad_id is not None else self.lm_tok.eos_token_id)
        self.acoustic_target = getattr(self.model, "acoustic_target", "rvq")
        self.acoustic_feat_loss = getattr(self.model, "acoustic_feat_loss", "l1")
        self.acoustic_feat_dim = getattr(self.model, "acoustic_feat_dim", 0)

        # Per-codebook acoustic weights for 'loss' mode. ``None`` -> the R
        # codebook CEs are summed uniformly (back-compat). When supplied
        # (e.g. the training ``rvq_loss_weights``) the per-channel mean CEs
        # are combined as a weighted sum instead.
        if self._acoustic_ch_weights_arg is None:
            self.acoustic_ch_weights: Optional[List[float]] = None
        else:
            w = [float(x) for x in self._acoustic_ch_weights_arg]
            if len(w) != self.R:
                raise ValueError(
                    f"acoustic_ch_weights has {len(w)} entries but the "
                    f"checkpoint has R={self.R} codebooks"
                )
            if any(x < 0.0 for x in w):
                raise ValueError(f"acoustic_ch_weights must be >= 0, got {w}")
            self.acoustic_ch_weights = w

    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def score_units(
        self,
        units: SubwordUnits,
        scoring_mode: Optional[str] = None,
    ) -> Dict[str, float]:
        """Return per-utterance scores under the requested mode.

        Parameters
        ----------
        units : SubwordUnits
            Output of the unit extractor for one utterance.
        scoring_mode : {"likelihood", "loss"}, optional
            Override the instance default for this call. Both modes share
            a single forward pass.

        Returns
        -------
        dict
            ``score`` is the comparable scalar (higher = better) — its
            definition depends on the mode. The dict also exposes raw
            sums (``text_nll``, ``acoustic_nll``) and per-stream means
            (``text_loss``, ``acoustic_loss``, ``acoustic_loss_per_ch``)
            so downstream analyses can compare modes without re-running
            the forward.
        """
        mode = scoring_mode or self.scoring_mode
        if mode not in SCORING_MODES:
            raise ValueError(
                f"scoring_mode must be one of {SCORING_MODES}, got {mode!r}"
            )

        if units.num_subwords < 2:
            # Need >=2 tokens for a single text-CE shift step. Mark as undefined.
            return {
                "score":                  0.0,
                "scoring_mode":           mode,
                "text_nll":               0.0,
                "acoustic_nll":           0.0,
                "text_loss":              0.0,
                "acoustic_loss":          0.0,
                "acoustic_loss_per_ch":   [0.0] * self.R,
                "n_text_tokens":          0,
                "n_acoustic_positions":   0,
            }

        device = self.device
        # Symmetric-delay extension: append EOS at slot L, then (D-1) PADs, so
        # T = L + D. Mirrors PadCollator. Text predictions cover [0, L-1] and
        # acoustic predictions cover [D-1, L+D-2] (right-shifted q/d targets).
        L = int(units.num_subwords)
        D = int(self.delay)
        T = L + D
        sub_real = units.subword_ids.to(device).long()                # (L,)
        q_real   = units.q_codes.to(device).long()                    # (L, R)
        d_real   = units.duration_frames.to(device).long()            # (L,)

        sub = sub_real.new_full((1, T), self.pad_token_id)
        sub[0, :L] = sub_real
        sub[0, L] = self.eos_token_id
        q = q_real.new_zeros((1, T, self.R))
        q[0, :L] = q_real
        d = d_real.new_zeros((1, T))
        d[0, :L] = d_real
        attn = torch.zeros((1, T), dtype=torch.bool, device=device)
        attn[0, : L + 1] = True  # real + EOS, NOT the trailing (D-1) PADs

        # Pre-baked masks over *prediction* positions (matches collator).
        text_lm = torch.zeros((1, T), dtype=torch.bool, device=device)
        text_lm[0, :L] = True
        aco_lm = torch.zeros((1, T), dtype=torch.bool, device=device)
        aco_lm[0, D - 1 : L + D - 1] = True

        ctx = (
            torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.bf16 and device.type == "cuda"
            else _nullcontext()
        )
        with ctx:
            out = self.model(sub, q, d, attn)

        # ---- text: HF causal-LM shift. logits[..., :-1] predicts labels[..., 1:].
        # Gate by the prediction-position mask [0, L-1] so the trailing PAD
        # slots (which have no valid next-token target) are ignored.
        text_logits_2d = out["text_logits"][..., :-1, :].float().reshape(
            -1, self.vocab_size
        )
        text_shift_mask = text_lm[..., : T - 1]  # (1, T-1)
        text_labels = sub[..., 1:].masked_fill(~text_shift_mask, -100)
        text_labels_1d = text_labels.reshape(-1)
        text_nll_sum = F.cross_entropy(
            text_logits_2d, text_labels_1d,
            ignore_index=-100, reduction="sum",
        )
        text_loss_mean = F.cross_entropy(
            text_logits_2d, text_labels_1d,
            ignore_index=-100, reduction="mean",
        )
        n_text_tokens = int(text_shift_mask.sum().item())

        # ---- acoustic: per-position scoring under the symmetric convention.
        # The model at prediction position n in [D-1, L+D-2] outputs q_{n-(D-1)},
        # so we right-shift q targets by D-1. (Duration is not scored.)
        shift_pad = D - 1
        if shift_pad > 0:
            q_targets_full = F.pad(q, (0, 0, shift_pad, 0))[:, :T, :]      # (1, T, R)
        else:
            q_targets_full = q

        aco_mask = aco_lm
        n_aco_positions = int(aco_mask.sum().item())

        if self.acoustic_target == "continuous":
            if units.pre_quant_feat is None:
                raise RuntimeError(
                    "scoring a continuous-acoustic model needs units.pre_quant_feat; "
                    "the RVQUnitExtractor must be re-run so it propagates the "
                    "teacher's pre-quantizer feature."
                )
            feat_pred = out["acoustic_feat_pred"].float()        # (1, T, D)
            feat_tgt_real = (units.pre_quant_feat.to(device).float()
                             .unsqueeze(0))                       # (1, L, D)
            # Right-shift teacher feature by D-1 so target at position n
            # corresponds to pre_quant_feat_{n-(D-1)}; trailing positions
            # (n > L+D-2) are masked out by aco_mask anyway.
            feat_tgt = F.pad(feat_tgt_real, (0, 0, shift_pad, 0))[:, :T, :]
            valid = aco_mask.to(feat_pred.dtype).unsqueeze(-1)   # (1, T, 1)
            denom = aco_mask.sum().clamp_min(1).to(feat_pred.dtype)
            if self.acoustic_feat_loss == "mse":
                err = (feat_pred - feat_tgt).pow(2).mean(dim=-1)         # (1, T)
                aco_mean = (err * aco_mask.to(err.dtype)).sum() / denom
            elif self.acoustic_feat_loss == "cosine":
                cos = F.cosine_similarity(feat_pred, feat_tgt, dim=-1)   # (1, T)
                err = 1.0 - cos
                aco_mean = (err * aco_mask.to(err.dtype)).sum() / denom
            elif self.acoustic_feat_loss == "mse_l1":
                # TASTE-style 0.5 * (MSE + L1), per-element mean over D
                # then masked-mean over valid positions.
                mse_err = (feat_pred - feat_tgt).pow(2).mean(dim=-1)
                l1_err  = (feat_pred - feat_tgt).abs().mean(dim=-1)
                m = aco_mask.to(mse_err.dtype)
                aco_mean = 0.5 * ((mse_err * m).sum() / denom) \
                         + 0.5 * ((l1_err  * m).sum() / denom)
            else:  # "l1"
                err = (feat_pred - feat_tgt).abs().mean(dim=-1)
                aco_mean = (err * aco_mask.to(err.dtype)).sum() / denom

            aco_loss_per_ch = []                                  # no per-codebook
            aco_mean_v = float(aco_mean.item())
            # Sum-style "NLL" surrogate: regression-mean × #valid positions.
            # Not a real likelihood, but keeps the schema and is monotonic
            # with aco_mean for fixed-length pairs.
            aco_nll_v = float(aco_mean_v * n_aco_positions)
            aco_loss_sum_of_means = aco_mean_v
        else:
            aco_logits = out["acoustic_logits"].float()              # (1, T, R, K)
            # Targets are the right-shifted q (q_targets_full); positions outside
            # aco_mask (the first D-1 positions and the trailing PAD slots) are
            # ignored.
            aco_targets = q_targets_full.clone()                     # (1, T, R)
            aco_targets = aco_targets.masked_fill(~aco_mask.unsqueeze(-1), -100)

            aco_nll_sum = F.cross_entropy(
                aco_logits.reshape(-1, self.K),
                aco_targets.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )

            # Per-channel mean CE (TASTE convention: each channel averaged
            # independently, then summed across R). Channel-r logits are
            # aco_logits[..., r, :], targets are aco_targets[..., r].
            aco_loss_per_ch = []
            for r in range(self.R):
                ch_logits = aco_logits[..., r, :].reshape(-1, self.K)
                ch_targets = aco_targets[..., r].reshape(-1)
                ch_loss = F.cross_entropy(
                    ch_logits, ch_targets,
                    ignore_index=-100, reduction="mean",
                )
                aco_loss_per_ch.append(float(ch_loss.item()))
            if self.acoustic_ch_weights is not None:
                aco_loss_sum_of_means = float(sum(
                    w * ch for w, ch in
                    zip(self.acoustic_ch_weights, aco_loss_per_ch)
                ))
            else:
                aco_loss_sum_of_means = float(sum(aco_loss_per_ch))
            aco_nll_v = float(aco_nll_sum.item())

        text_nll_v = float(text_nll_sum.item())
        text_loss_v = float(text_loss_mean.item())

        if mode == "likelihood":
            score = -(self.text_weight * text_nll_v
                      + self.acoustic_weight * aco_nll_v)
        else:  # "loss" — TASTE-style: sum of per-stream mean CE values.
            score = -(self.text_weight * text_loss_v
                      + self.acoustic_weight * aco_loss_sum_of_means)

        return {
            "score":                  score,
            "scoring_mode":           mode,
            "text_nll":               text_nll_v,
            "acoustic_nll":           aco_nll_v,
            "text_loss":              text_loss_v,
            "acoustic_loss":          aco_loss_sum_of_means,
            "acoustic_loss_per_ch":   aco_loss_per_ch,
            "n_text_tokens":          n_text_tokens,
            "n_acoustic_positions":   n_aco_positions,
        }
