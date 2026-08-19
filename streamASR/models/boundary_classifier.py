"""
boundary_classifier.py

Joiner-style binary classifier that predicts whether the last subword token
emitted at a streaming chunk boundary ends a complete word.

Architecture
------------
1. Separate linear projections for encoder and predictor outputs.
2. Concatenation of the two projected vectors.
3. Multi-layer MLP with LayerNorm, GELU activation, and Dropout.
4. 2-class output (non-boundary / boundary).

Classes
-------
0 : non-boundary  – the last emitted subword is an incomplete word
                    (the next subword will continue the same word)
1 : boundary      – the last emitted subword ends a complete word
                    (a whitespace would follow in the full transcript)
"""

import torch
import torch.nn as nn


class BoundaryClassifier(nn.Module):
    """
    Multi-layer binary classifier for word-boundary detection in streaming ASR.

    Given the encoder hidden state at the last frame of a chunk and the
    predictor hidden state after the last emitted subword, predicts whether
    that subword completes a full word.

    Architecture
    ~~~~~~~~~~~~
    * ``proj_enc``  : Linear(joint_dim → hidden_dim) + GELU
    * ``proj_pred`` : Linear(joint_dim → hidden_dim) + GELU
    * Concatenate   : [proj_enc; proj_pred]  →  2 * hidden_dim
    * MLP           : (num_layers − 1) × [Linear → LayerNorm → GELU → Dropout]
                      followed by Linear(hidden_dim → 2)

    Parameters
    ----------
    joint_dim : int
        Dimensionality of both ``enc_out`` and ``pred_out``.  Must match
        ``joint_dim`` in the ASR hparams (default 640).
    hidden_dim : int
        Width of the hidden layers in the MLP (default 512).
    num_layers : int
        Total number of linear layers in the MLP, including the output layer
        (default 3, so 2 hidden + 1 output).
    dropout : float
        Dropout probability applied after each hidden GELU (default 0.1).
    """

    def __init__(
        self,
        joint_dim: int = 640,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        # ── Separate projections ──────────────────────────────────────────────
        self.proj_enc = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim, bias=True),
            nn.GELU(),
        )
        self.proj_pred = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim, bias=True),
            nn.GELU(),
        )

        # ── MLP over concatenated projections ─────────────────────────────────
        # First hidden layer takes 2 * hidden_dim (concatenated enc + pred).
        layers: list[nn.Module] = []
        in_dim = hidden_dim * 2
        for _ in range(num_layers - 1):
            layers += [
                nn.Linear(in_dim, hidden_dim, bias=True),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(p=dropout),
            ]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 2, bias=True))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        enc_out: torch.Tensor,
        pred_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        enc_out : Tensor, shape ``[..., joint_dim]``
            Encoder output at the last frame of the current chunk, projected
            to joint dimension via ``proj_enc`` inside the ASR joint network.
        pred_out : Tensor, shape ``[..., joint_dim]``
            Predictor output after the last emitted subword, projected to
            joint dimension via ``proj_dec`` inside the ASR joint network.

        Returns
        -------
        logits : Tensor, shape ``[..., 2]``
            Raw class logits ``[non-boundary, boundary]``.
            Use ``argmax(-1)`` for hard predictions or
            ``F.cross_entropy`` for training.
        """
        e = self.proj_enc(enc_out)    # [..., hidden_dim]
        p = self.proj_pred(pred_out)  # [..., hidden_dim]
        x = torch.cat([e, p], dim=-1) # [..., 2 * hidden_dim]
        return self.mlp(x)            # [..., 2]
