import io
import torch
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
from PIL import Image

def save_alignment_heatmap(
    path_u_t: torch.Tensor,            # [B, U, T] bool/int
    target_ext: list[int],             # len=U
    tokenizer,
    out_png: str = "align.png",
    b: int = 0,
    max_frames_for_ticks: int = 2000,
    max_y_labels: int = 60,            
    max_x_labels: int = 30,            
):
    assert path_u_t.dim() == 3

    mat = path_u_t[b].detach().to("cpu").to(torch.uint8).numpy()  # [U, T]
    Uprime, T = mat.shape

    blank_id = tokenizer.blank_id

    row_labels = []
    for tid in target_ext:
        if tid == blank_id:
            row_labels.append("<BLK>")
        else:
            try:
                s = tokenizer.decode([tid])
                row_labels.append(s if (s is not None and s != "") else str(tid))
            except Exception:
                row_labels.append(str(tid))

    if len(row_labels) < Uprime:
        row_labels += [f"u{idx}" for idx in range(len(row_labels), Uprime)]
    elif len(row_labels) > Uprime:
        row_labels = row_labels[:Uprime]


    plt.figure(figsize=(max(10, T / 200), max(6, Uprime / 10)))
    plt.imshow(mat, aspect="auto", interpolation="nearest", vmin=0, vmax=1)

    plt.xlabel("frame index (t)")
    plt.ylabel("token index (u)")

    if Uprime <= max_y_labels:
        yt = np.arange(Uprime)
    else:
        step = max(1, int(np.ceil(Uprime / max_y_labels)))
        yt = np.arange(0, Uprime, step)

    plt.yticks(yt, [row_labels[i] for i in yt])


    if T <= 120:
        xt = np.arange(T)
    else:
        step = max(1, int(np.ceil(T / max_x_labels)))
        if T > max_frames_for_ticks:
            step = max(step, int(np.ceil(T / (max_x_labels * 1.5))))
        xt = np.arange(0, T, step)

    plt.xticks(xt, [str(i) for i in xt], rotation=0)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    return out_png


# ============================================================================
# TensorBoard / spectrogram helpers
# ============================================================================

def plot_to_tensorboard(writer, tag, figure, step):
    """Convert a matplotlib figure to a TensorBoard image and log it."""
    buf = io.BytesIO()
    figure.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf)
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1)
    writer.add_image(tag, img_tensor, step, dataformats="CHW")
    plt.close(figure)


def plot_spectrogram(waveform, sr=16000, title="Mel Spectrogram"):
    """Create a mel spectrogram plot from a waveform."""
    fig, ax = plt.subplots(figsize=(10, 4))
    S = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=80, fmin=0, fmax=8000)
    S_db = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel",
                             fmin=0, fmax=8000, ax=ax)
    ax.set_title(title)
    fig.colorbar(ax.collections[0], ax=ax, format="%+2.0f dB")
    return fig


def plot_dual_alignment(char_alignment, word_alignment, char_data, title_prefix="Alignment"):
    """Plot character-level and word-level alignments side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    char_matrix = char_alignment.cpu().numpy()
    im1 = ax1.imshow(char_matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax1.set_title(f"{title_prefix} - Character Level")
    ax1.set_xlabel("Speech Frames (HuBERT)")
    ax1.set_ylabel("Character Position")
    if "char_sequence" in char_data:
        char_seq = char_data["char_sequence"]
        n_chars = min(len(char_seq), char_matrix.shape[0])
        if n_chars <= 50:
            ax1.set_yticks(range(n_chars))
            ax1.set_yticklabels([repr(c) for c in char_seq[:n_chars]], fontsize=8)
    fig.colorbar(im1, ax=ax1)

    word_matrix = word_alignment.cpu().numpy()
    im2 = ax2.imshow(word_matrix, aspect="auto", cmap="Greens", vmin=0, vmax=1)
    ax2.set_title(f"{title_prefix} - Word Level")
    ax2.set_xlabel("Speech Frames (HuBERT)")
    ax2.set_ylabel("Word Position")
    if "tokens" in char_data:
        tokens = char_data["tokens"]
        n_words = min(len(tokens), word_matrix.shape[0])
        if n_words <= 30:
            ax2.set_yticks(range(n_words))
            ax2.set_yticklabels(tokens[:n_words], fontsize=8, rotation=45, ha="right")
    fig.colorbar(im2, ax=ax2)
    plt.tight_layout()
    return fig


def plot_char_to_word_mapping(char_data, title="Character to Word Mapping"):
    """Visualize the mapping from characters to words."""
    if "char_to_word_map" not in char_data or "char_sequence" not in char_data:
        return None

    char_to_word_map = char_data["char_to_word_map"]
    char_sequence = char_data["char_sequence"]
    tokens = char_data.get("tokens", [])

    fig, ax = plt.subplots(figsize=(16, 6))
    n_chars = len(char_sequence)
    n_words = len(tokens) if tokens else max(char_to_word_map) + 1

    mapping_matrix = np.zeros((n_words, n_chars))
    for c_idx, w_idx in enumerate(char_to_word_map):
        if c_idx < n_chars and w_idx < n_words:
            mapping_matrix[w_idx, c_idx] = 1

    im = ax.imshow(mapping_matrix, aspect="auto", cmap="Oranges", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel("Character Position")
    ax.set_ylabel("Word Position")

    if n_chars <= 100:
        char_labels = [repr(c) for c in char_sequence]
        step = max(1, n_chars // 50)
        ax.set_xticks(range(0, n_chars, step))
        ax.set_xticklabels(
            [char_labels[i] for i in range(0, n_chars, step)], fontsize=6, rotation=90
        )
    if tokens and n_words <= 30:
        ax.set_yticks(range(n_words))
        ax.set_yticklabels(tokens, fontsize=8)

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig


def plot_duration_comparison(d_pred, d_gt, char_data, title="Duration Prediction vs Ground Truth"):
    """Plot predicted vs ground truth character durations."""
    d_pred_np = d_pred.cpu().numpy()
    d_gt_np = d_gt.cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    char_positions = np.arange(len(d_pred_np))
    width = 0.35
    ax1.bar(char_positions - width / 2, d_pred_np, width, label="Predicted", alpha=0.7, color="blue")
    ax1.bar(char_positions + width / 2, d_gt_np, width, label="Ground Truth", alpha=0.7, color="red")
    ax1.set_title(title)
    ax1.set_xlabel("Character Position")
    ax1.set_ylabel("Duration (frames)")
    ax1.legend()
    if "char_sequence" in char_data and len(char_data["char_sequence"]) <= 50:
        char_labels = [repr(c) for c in char_data["char_sequence"][:len(d_pred_np)]]
        ax1.set_xticks(char_positions)
        ax1.set_xticklabels(char_labels, fontsize=8, rotation=45, ha="right")

    ax2.scatter(d_gt_np, d_pred_np, alpha=0.6)
    ax2.plot([d_gt_np.min(), d_gt_np.max()], [d_gt_np.min(), d_gt_np.max()], "r--", lw=2)
    ax2.set_xlabel("Ground Truth Duration")
    ax2.set_ylabel("Predicted Duration")
    ax2.set_title("Duration Correlation")
    if len(d_gt_np) > 1:
        corr_coef = np.corrcoef(d_gt_np, d_pred_np)[0, 1]
        ax2.text(0.05, 0.95, f"Correlation: {corr_coef:.3f}", transform=ax2.transAxes,
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    return fig


def visualize_validation_batch(
    writer,
    global_step,
    batch_idx,
    model_out,
    preprocessed_batch,
    char_alignment,
    word_alignment,
    waveform,
):
    """Create comprehensive visualizations for a validation batch."""
    if batch_idx >= char_alignment.size(0):
        return

    char_data_item = preprocessed_batch["char_data"][batch_idx]
    char_align_item = char_alignment[batch_idx]
    word_align_item = word_alignment[batch_idx]
    audio_item = waveform[batch_idx]

    forward_out = model_out["forward_output"]
    d_pred_item = (
        forward_out.get("d_pred", [None])[batch_idx]
        if "d_pred" in forward_out else None
    )

    prefix = f"validation/sample_{batch_idx}"

    try:
        dual_align_fig = plot_dual_alignment(
            char_align_item, word_align_item, char_data_item,
            title_prefix=f"Sample {batch_idx} Alignment",
        )
        plot_to_tensorboard(writer, f"{prefix}/dual_alignment", dual_align_fig, global_step)
    except Exception as e:
        print(f"Warning: Could not create dual alignment plot: {e}")

    try:
        mapping_fig = plot_char_to_word_mapping(
            char_data_item, title=f"Sample {batch_idx} - Char to Word Mapping"
        )
        if mapping_fig is not None:
            plot_to_tensorboard(writer, f"{prefix}/char_word_mapping", mapping_fig, global_step)
    except Exception as e:
        print(f"Warning: Could not create char-word mapping plot: {e}")

    if d_pred_item is not None:
        try:
            d_gt_item = char_align_item.sum(dim=1)
            duration_fig = plot_duration_comparison(
                d_pred_item, d_gt_item, char_data_item,
                title=f"Sample {batch_idx} - Duration Predictions",
            )
            plot_to_tensorboard(
                writer, f"{prefix}/duration_comparison", duration_fig, global_step
            )
        except Exception as e:
            print(f"Warning: Could not create duration plot: {e}")

    try:
        spec_fig = plot_spectrogram(
            audio_item.cpu().numpy(), title=f"Sample {batch_idx} - Mel Spectrogram"
        )
        plot_to_tensorboard(writer, f"{prefix}/spectrogram", spec_fig, global_step)
    except Exception as e:
        print(f"Warning: Could not create spectrogram plot: {e}")