"""Single source of truth for shapes / paths used across streamSLM."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class TokenizerConfig:
    """StreamAlign per-subword token producer."""
    token_type: Literal["rvq"] = "rvq"
    rvq_num_quantizers: int = 16
    rvq_codebook_size: int = 512
    text_tokenizer: Literal["llama", "qwen3"] = "llama"
    streamalign_ckpt: str = (
        "/home/streamalign/streamASR/checkpoints/"
        "streamalign_r16/"
        "epoch_22.pt"
    )

    @property
    def codebook_size(self) -> int:
        return self.rvq_codebook_size

    @property
    def num_quantizers(self) -> int:
        return self.rvq_num_quantizers


@dataclass
class ModelConfig:
    backbone: str = "meta-llama/Llama-3.2-1B"
    duration_loss: Literal["l1", "mse"] = "l1"
    duration_target: Literal["log_frames", "frames"] = "log_frames"
    delay: int = 1
    loss_w_text: float = 1.0
    loss_w_acoustic: float = 1.0
    loss_w_duration: float = 0.1
    # Text-loss blend: text_loss = kl_w * KL(ref||spoken) + (1 - kl_w) * CE.
    # Default 0.9 matches TASTE-SpokenLM stage-2 (modeling_taste.py:903).
    # Setting 0.0 drops the KL term entirely; the train launcher then also
    # skips loading the ref_model so its memory + fwd cost are recovered.
    text_kl_weight: float = 0.9
    # Residual-VQ codebook predictor:
    #   "multihead"        — legacy flat Linear, no inter-codebook coupling.
    #   "depth_transformer"— Moshi-style depth-wise AR predictor.
    #   "moss_local_depth" — MOSS-TTS-Local-style depth predictor: Qwen3-block
    #                        (RMSNorm + SwiGLU + no-PE attention) with a SwiGLU
    #                        MLP bridge global->local. Separate code path from
    #                        "depth_transformer" by design — see predictors.py.
    acoustic_predictor_type: Literal[
        "multihead", "depth_transformer", "moss_local_depth"
    ] = "multihead"
    depformer_dim: int = 128
    depformer_num_heads: int = 4
    depformer_num_layers: int = 2
    depformer_dim_feedforward: Optional[int] = None
    depformer_multi_linear: bool = False
    depformer_weights_per_step: bool = False
    depformer_norm: bool = False
    # Dropout on the previous-codebook embedding fed into the next depth-
    # layer prediction (AR coupling branch only). Does NOT apply to the
    # LLM-hidden projection dep_in(h) nor to the BOS at k=0. 0 = off.
    prev_depth_output_dropout: float = 0.0
    # MOSS-TTS-Local depth-predictor knobs (consumed only when
    # acoustic_predictor_type == "moss_local_depth"). Defaults match
    # MossTTSLocalConfig: local_hidden=1536, num_heads=16, num_layers=4,
    # ffn=8960, bridge_ffn=2048 (MOSS's additional_mlp_ffn_hidden_size).
    moss_local_hidden: int = 1536
    moss_local_num_heads: int = 16
    moss_local_num_layers: int = 4
    moss_local_ffn: int = 8960
    moss_local_bridge_ffn: int = 2048
    # Audio<->text fusion mode at the LLM input.
    #   "gated_softmax" (default, legacy): WeightedSumFusion with learnable
    #     softmax gate init [-2, 2] -> ~[0.018, 0.982] (audio damped at start).
    #   "sum": Moshi-style direct add `Linear(audio) + text` (no gate).
    fusion_mode: Literal["gated_softmax", "sum"] = "gated_softmax"
    # Per-codebook acoustic-embedding init std. "auto" => legacy 0.02/sqrt(R)
    # so the R-summed audio_e has std ~0.02 (text-embedding scale). A float
    # string overrides per-book std directly (e.g. "0.02" => Moshi-like, each
    # book at std 0.02, R-sum ~0.02*sqrt(R)).
    audio_emb_init_scale: str = "auto"
    # Optional per-codebook training weights for the residual-VQ CE loss. If
    # set, must be length R. Validation always uses uniform mean for fair
    # cross-config comparison; only training-time `aco_loss` is reweighted.
    rvq_loss_weights: Optional[List[float]] = None
    # Duration-head input source.
    #   "llm" (default): read LLM hidden state h_n (legacy).
    #   "depformer_last": read depth-transformer's final-position output
    #     u^{R-1}_n. Requires acoustic_predictor_type='depth_transformer'.
    duration_head_input: Literal["llm", "depformer_last"] = "llm"
    # Duration-loss objective.
    #   "regression" (default): L1/MSE on log1p(frames), single scalar head.
    #   "classification": CE over `duration_num_buckets` linear frame buckets;
    #     bucket = clamp(frames, 0, num_buckets-1).
    duration_loss_type: Literal["regression", "classification"] = "regression"
    duration_num_buckets: int = 256
    # When True, embed the (training: GT; inference: predicted) duration
    # bucket and add it to every depth-transformer input z^k_n. Requires
    # depth_transformer predictor and duration_loss_type='classification'.
    feed_duration_to_depformer: bool = False
    # Acoustic prediction target.
    #   "rvq" (default): per-codebook CE on q_codes via the
    #     multihead/depth_transformer predictor.
    #   "continuous": regress LLM hidden -> a continuous teacher feature
    #     (the StreamAlign pre-quantizer encoder output, dim 256). Drops the
    #     discrete-acoustic head entirely. Cache must contain pre_quant_feat.
    acoustic_target: Literal["rvq", "continuous"] = "rvq"
    # Continuous-acoustic head dimension. Must match the cached pre_quant_feat
    # last dim (256 for the StreamAlign teacher). Ignored when acoustic_target='rvq'.
    acoustic_feat_dim: int = 256
    # Loss for the continuous-acoustic regression. l1 (default), mse,
    # cosine (1 - cos similarity), or mse_l1 (0.5*MSE + 0.5*L1, TASTE-style
    # equal-weight average of two regression losses). Ignored when
    # acoustic_target='rvq'.
    acoustic_feat_loss: Literal["l1", "mse", "cosine", "mse_l1"] = "l1"
    # Hidden-state source for the acoustic + duration heads.
    #   "last" (default, legacy): h = out.hidden_states[-1] (final post-norm layer).
    #   "weighted": learnable softmax mix over all (num_hidden_layers + 1) hidden
    #     states (embedding output + each transformer block), mirroring
    #     TASTE-SpokenLM's WeightedLayerExtract. Text head still uses out.logits
    #     (i.e., the final post-norm layer) since Llama's lm_head was trained
    #     on that representation.
    acoustic_layer_mix: Literal["last", "weighted"] = "last"
    # Model architecture variant.
    #   "streamslm" (default, legacy): the original StreamSLM with parallel
    #     text / acoustic / duration heads off of h_n.
    #   "streamslm_hier": Moshi-style step-internal hierarchical AR — text
    #     stays at lm_head(h_n) but (q_n, d_n) are produced by a depth
    #     transformer that conditions on the already-sampled w_{n+1}.
    model_arch: Literal["streamslm", "streamslm_hier"] = "streamslm"
    # Ordering of duration vs acoustic codebooks inside the hier depth-AR.
    #   "duration_last" (default): [text, q^0, ..., q^{R-1}, dur]
    #   "duration_first":          [text, dur, q^0, ..., q^{R-1}]
    # Ignored when model_arch != "streamslm_hier".
    hier_ar_order: Literal["duration_first", "duration_last"] = "duration_last"


@dataclass
class TrainConfig:
    manifest: str = "streamSLM/configs/manifests/train_combined.csv"
    cache_root: str = "cache/streamSLM_units"
    batch_size: int = 8
    grad_accum: int = 4
    max_seq_len: int = 1024
    lr: float = 2e-4
    warmup_steps: int = 1000
    max_steps: int = 50_000
    bf16: bool = True
    save_every: int = 2_000
    log_every: int = 50
    out_dir: str = "checkpoints/slm/validation"


@dataclass
class StreamSLMConfig:
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
