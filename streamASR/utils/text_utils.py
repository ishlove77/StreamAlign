from typing import List
import torch

# ── Vocabulary constants ────────────────────────────────────────────────────────
WORD_VOCAB_SIZE = 128000
SOS_ID = 6561
PAD_ID = -100
VOCAB_SIZE = SOS_ID + 1

# Character vocabulary - 72 characters (mirrors models/model.py)
_CHAR_VOCAB = " !\"'(),-.:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyzåèéíñö"
CHAR_TO_IDX: dict = {char: idx for idx, char in enumerate(_CHAR_VOCAB)}
for _up in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    CHAR_TO_IDX[_up] = CHAR_TO_IDX[_up.lower()]
IDX_TO_CHAR: dict = {idx: char for idx, char in enumerate(_CHAR_VOCAB)}
CHAR_VOCAB_SIZE: int = len(_CHAR_VOCAB)
BLANK_ID: int = CHAR_VOCAB_SIZE       # CTC blank token (index 72)
CTC_VOCAB_SIZE: int = CHAR_VOCAB_SIZE + 1  # Include blank


class CharacterTokenizer:
    """Character-level tokenizer for ASR.

    Vocabulary: 72 printable characters (space, punctuation, a-z, accented).
    Uppercase letters are folded to their lowercase indices.
    Blank (CTC) is index BLANK_ID (= CHAR_VOCAB_SIZE = 72).
    """

    def __init__(self):
        self.char2idx = CHAR_TO_IDX
        self.idx2char = IDX_TO_CHAR
        self.blank_id = BLANK_ID
        self.vocab_size = CHAR_VOCAB_SIZE

    def encode(self, text):
        """Encode text to token IDs, skipping unknown characters."""
        return [self.char2idx[ch] for ch in text if ch in self.char2idx]

    def decode(self, tokens):
        """Decode token IDs to text, skipping blank."""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return "".join(
            self.idx2char[idx] for idx in tokens
            if idx in self.idx2char and idx != self.blank_id
        )

    def decode_ids(self, tokens):
        """Alias for decode (compatibility with SpeechBrain)."""
        return self.decode(tokens)


_char_tokenizer = CharacterTokenizer()


# ── Feature-length helpers ──────────────────────────────────────────────────────

def compute_model_feat_len(waveform_samples: int) -> int:
    """Compute output frame length from waveform sample length.

    For alignment.yaml:
      1. Waveform → Fbank: hop_length=160 (10 ms), win_length=512 (32 ms)
      2. Fbank → CNN: 2 conv layers (kernel=3, stride=2)
    """
    hop_length = 160
    win_length = 512
    fbank_frames = (waveform_samples - win_length) // hop_length + 1

    L = fbank_frames
    for k, s in zip([3, 3], [2, 2]):
        L = (L - k) // s + 1
        if L < 0:
            return 0
    return L


# ── Text / character utilities ──────────────────────────────────────────────────

def text_to_char_sequence(text: str) -> List[str]:
    """Convert text to character sequence, skipping unknown characters.

    Matches CharacterTokenizer in models/model.py:
    - Converts to lowercase
    - Keeps characters present in CHAR_TO_IDX (space, punctuation, a-z, accented)
    """
    text = text.lower()
    return [ch for ch in text if ch in CHAR_TO_IDX]


def char_sequence_to_indices(char_seq: List[str]) -> List[int]:
    """Convert a character sequence to vocabulary indices."""
    return [CHAR_TO_IDX.get(ch, CHAR_TO_IDX[" "]) for ch in char_seq]


def batch_word_char_alignment(tokenizer, texts: List[str]) -> List[dict]:
    """Build character/word alignment metadata for a batch of texts.

    Returns one dict per utterance with keys:
        char_sequence      – list[str]
        char_indices       – list[int]
        char_to_word_map   – list[int]  (char index → token index)
        tokens             – list[str]
        word_token_ids     – list[int]
    """
    encodings = tokenizer(
        texts,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
        padding=False,
        truncation=False,
    )

    results = []

    for i, text in enumerate(texts):
        offsets = encodings["offset_mapping"][i]
        token_ids = encodings["input_ids"][i]
        tokens = tokenizer.convert_ids_to_tokens(token_ids)

        char_seq = text_to_char_sequence(text)
        char_idx = char_sequence_to_indices(char_seq)

        c2w = [-1] * len(char_seq)
        for tok_idx, (s, e) in enumerate(offsets):
            for pos in range(s, e):
                if pos < len(c2w):
                    c2w[pos] = tok_idx

        results.append({
            "char_sequence":    char_seq,
            "char_indices":     char_idx,
            "char_to_word_map": c2w,
            "tokens":           tokens,
            "word_token_ids":   token_ids,
        })
    return results


def char_data_from_constraints(tokenizer, char_ids_list, constraints_list) -> List[dict]:
    """Build per-utterance char_data dicts directly from RNN-T constraints.

    Intervals are concatenated **as-is** into a single string (word boundaries
    are encoded as " " characters already present in cids, thanks to the
    leading-space preservation in build_char_constraints), then tokenized
    **once** by the LLM tokenizer. This gives natural Llama BPE (e.g.
    " turners" -> Gturn + ers) instead of the per-interval-with-leading-space
    tokenization that produced Gturn + Gers.

    ``char_indices`` and ``char_sequence`` retain one-to-one correspondence
    with the input ``cids`` so the alignment loop in _build_alignment_from_rnnt
    keeps its frame-to-char mapping. ``char_to_word_map[i]`` maps each input
    char to its LLM-token index in ``word_token_ids``.
    """
    out = []
    for cids, cons in zip(char_ids_list, constraints_list):
        cids_list = list(cids) if cids is not None else []
        cons_list = list(cons) if cons is not None else []
        if not cids_list or not cons_list:
            out.append({
                "char_sequence":    [],
                "char_indices":     [],
                "char_to_word_map": [],
                "tokens":           [],
                "word_token_ids":   [],
            })
            continue

        concat_text = ""
        interval_offsets: List[tuple] = []
        for tup in cons_list:
            u_s, u_e = int(tup[2]), int(tup[3])
            if u_e <= u_s:
                continue
            interval_chars = [IDX_TO_CHAR.get(int(i), "") for i in cids_list[u_s:u_e]]
            interval_text = "".join(interval_chars)
            if not interval_text.strip():
                continue
            start = len(concat_text)
            concat_text += interval_text
            interval_offsets.append((u_s, u_e, start))

        char_seq = [IDX_TO_CHAR.get(int(i), "") for i in cids_list]
        if not concat_text:
            out.append({
                "char_sequence":    char_seq,
                "char_indices":     cids_list,
                "char_to_word_map": [-1] * len(cids_list),
                "tokens":           [],
                "word_token_ids":   [],
            })
            continue

        enc = tokenizer(
            concat_text, add_special_tokens=False, return_offsets_mapping=True,
            return_attention_mask=False, padding=False, truncation=False,
        )
        tids = list(enc["input_ids"])
        offs = list(enc["offset_mapping"])
        tokens_str = tokenizer.convert_ids_to_tokens(tids)

        pos2tok = [-1] * len(concat_text)
        for ti, (s, e) in enumerate(offs):
            for p in range(s, min(e, len(concat_text))):
                pos2tok[p] = ti

        local_c2w: List[int] = [-1] * len(cids_list)
        last_tok = (len(offs) - 1) if offs else 0
        for u_s, u_e, concat_start in interval_offsets:
            for raw_i in range(u_e - u_s):
                pos = concat_start + raw_i
                ti = pos2tok[pos] if 0 <= pos < len(pos2tok) else -1
                local_c2w[u_s + raw_i] = ti if ti >= 0 else last_tok

        out.append({
            "char_sequence":    char_seq,
            "char_indices":     cids_list,
            "char_to_word_map": local_c2w,
            "tokens":           tokens_str,
            "word_token_ids":   tids,
        })
    return out
