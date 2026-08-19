"""RVQ with learnable codebook (no EMA), so codebook entries are nn.Parameters
and can receive their own optimizer LR multiplier.

Subclasses base RVQ model. After super().__init__, rebuilds ``residual_vq``
with ``learnable_codebook=True`` and exposes the codebook size/dim from env.
"""
import os

from vector_quantize_pytorch import ResidualVQ

from models.model_tokenizer import Data2VecSemanticAcousticModel as _BaseRVQ


class Data2VecSemanticAcousticModel(_BaseRVQ):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        feat_dim = getattr(self.residual_vq, "dim", 256)
        codebook_size = int(os.environ.get("RVQ_CODEBOOK_SIZE", "512"))
        codebook_dim = int(os.environ.get("RVQ_CODEBOOK_DIM", str(feat_dim)))
        rvq_r = int(os.environ.get("RVQ_R", str(getattr(self, "rvq_r", 32))))
        commit_weight = float(os.environ.get("COMMIT_WEIGHT", "1.0"))
        self.residual_vq = ResidualVQ(
            dim=feat_dim,
            num_quantizers=rvq_r,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            kmeans_init=True,
            kmeans_iters=100,
            threshold_ema_dead_code=2,
            commitment_weight=commit_weight,
            learnable_codebook=True,   # codebook entries become nn.Parameters
            ema_update=False,          # required when learnable_codebook=True
        )
        print(f"[rvq-learnable] residual_vq with learnable_codebook=True "
              f"R={rvq_r} C={codebook_size} dim={codebook_dim}", flush=True)
