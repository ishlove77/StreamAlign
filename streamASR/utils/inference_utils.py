import torch
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from models.boundary_classifier import BoundaryClassifier

def load_boundary_classifier(ckpt_path: str, device: torch.device):
    """Load a trained BoundaryClassifier from a checkpoint file."""
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    hp    = ckpt["hparams"]
    model = BoundaryClassifier(
        joint_dim  = hp["joint_dim"],
        hidden_dim = hp.get("hidden_dim", 512),
        num_layers = hp.get("num_layers", 3),
        dropout    = hp.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

def compute_pred_feat(
    context,
    device: torch.device,
) -> torch.Tensor:
    """
    Extract the proj_dec output stored in the streaming context.

    The greedy searcher's decode_network_lst is [emb, dec, proj_dec], so
    _forward_PN runs the full emb → LSTM → proj_dec chain.  After each call
    to transducer_greedy_decode_streaming the context stores:

        context.decoder_context.hidden = (out_PN, rnn_hidden)
            out_PN    : Tensor [batch, 1, joint_dim]  — proj_dec output for
                        the token that will be used in the next blank decision
            rnn_hidden: LSTM (h, c) state

    We return out_PN[:, -1, :] directly — no re-computation needed.

    Returns
    -------
    pred_feat : Tensor [1, joint_dim]
    """
    hidden_state = context.decoder_context.hidden   # (out_PN, rnn_hidden)
    out_pn = hidden_state[0]                        # [batch, 1, joint_dim]
    return out_pn[:, -1, :].to(device)              # [1, joint_dim]