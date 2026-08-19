
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple
import os
import argparse
import time
import torch
import torch.nn as nn
import torchaudio
import csv
import numpy as np
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm
from typing import List
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig
from speechbrain.utils.streaming import split_fixed_chunks
import itertools
from vector_quantize_pytorch import ResidualVQ
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.inference_utils import compute_pred_feat

# ============================================================================
# Character Tokenizer
# ============================================================================

# Character vocabulary - 72 characters
CHAR_VOCAB = " !\"'(),-.:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyzåèéíñö"
CHAR_TO_IDX = {char: idx for idx, char in enumerate(CHAR_VOCAB)}
for _up in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    CHAR_TO_IDX[_up] = CHAR_TO_IDX[_up.lower()]
IDX_TO_CHAR = {idx: char for idx, char in enumerate(CHAR_VOCAB)}
CHAR_VOCAB_SIZE = len(CHAR_VOCAB)
BLANK_ID = CHAR_VOCAB_SIZE      # CTC blank token
CTC_VOCAB_SIZE = CHAR_VOCAB_SIZE + 1  # Include blank


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
# ============================================================================
# ASRStreaming Context
# ============================================================================
@dataclass
class ASRStreamingContext:
    config: DynChunkTrainConfig
    fea_extractor_context: Any
    encoder_context: Any
    decoder_context: Any
    tokenizer_context: Optional[List[Any]]

# ============================================================================
# Streaming Char Model
# ============================================================================
class StreamingCharModel(nn.Module):
    def __init__(self, hparams_file, checkpoint_path=None, output_hidden_states=False, device="cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.output_hidden_states = output_hidden_states

        with open(hparams_file) as f:
            self.hparams = load_hyperpyyaml(f)

        # Convert to nn.ModuleDict so PyTorch tracks parameters
        self.mods = nn.ModuleDict()
        for name, module in self.hparams["modules"].items():
            if isinstance(module, nn.Module):
                self.mods[name] = module

        self.filter_props = self.hparams['fea_streaming_extractor'].properties

        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

        self.tokenizer = CharacterTokenizer()

    def forward(self, wavs, wav_lens, dynchunktrain_config=None, epoch=0):
        """Forward pass for training (non-streaming).

        Arguments
        ---------
        wavs : torch.Tensor
            Audio waveforms [batch, time]
        wav_lens : torch.Tensor
            Relative lengths [batch]
        dynchunktrain_config : DynChunkTrainConfig, optional
            Dynamic chunk training configuration

        Returns
        -------
        x : torch.Tensor
            Encoder output after projection [batch, time, dim]
        """
        # Feature extraction
        feats = self.hparams['compute_features'](wavs)

        # Normalize features
        feats = self.mods['normalize'](feats, wav_lens, epoch=epoch)

        # CNN frontend
        src = self.mods['CNN'](feats)

        # Encoder
        x, h = self.mods['enc'](
            src,
            wav_lens,
            pad_idx=0,
            dynchunktrain_config=dynchunktrain_config,
        )

        # Projection
        x = self.mods['proj_enc'](x)

        return x, h

    def _load_checkpoint(self, path: str):
        """Load checkpoint - simplified version"""
        if os.path.isfile(path):
            path = os.path.dirname(path)

        # Find model.ckpt
        model_ckpt = os.path.join(path, "model.ckpt")
        norm_ckpt = os.path.join(path, "normalizer.ckpt")

        if os.path.exists(model_ckpt):
            print(f"Loading model from: {model_ckpt}")
            ckpt = torch.load(model_ckpt, map_location=self.device)
            missing, unexpected = self.hparams["model"].load_state_dict(ckpt, strict=False)
            if missing:
                print(f"  Missing keys (will use init): {missing}")
            if unexpected:
                print(f"  Unexpected keys (ignored): {unexpected[:3]}{'...' if len(unexpected) > 3 else ''}")
        else:
            # model.ckpt not found in given path; search parent dir for CKPT+* folders
            from speechbrain.utils.checkpoints import Checkpointer
            parent_dir = os.path.dirname(path)
            recoverables = {"model": self.hparams["model"]}
            normalize = self.mods["normalize"] if "normalize" in self.mods else None
            if normalize is not None:
                recoverables["normalizer"] = normalize
            checkpointer = Checkpointer(checkpoints_dir=parent_dir, recoverables=recoverables)
            checkpointer.recover_if_possible()

        # Load normalizer
        if os.path.exists(norm_ckpt):
            print(f"Loading normalizer from: {norm_ckpt}")
            norm_state = torch.load(norm_ckpt, map_location=self.device)
            normalize = self.mods["normalize"] if "normalize" in self.mods else None
            if normalize and isinstance(norm_state, dict):
                if "glob_mean" in norm_state:
                    normalize.glob_mean = norm_state["glob_mean"]
                    normalize.glob_std = norm_state["glob_std"]
                    normalize.count = norm_state.get("count", 1)
                    print("  Loaded normalizer stats")

        # Load tokenizer — look for tokenizer.ckpt in sibling pretrained/ folder
        # path is the CKPT dir (e.g. .../word_asr/save/CKPT+...)
        # so ../pretrained/ is two levels up then into pretrained/
        tokenizer = self.hparams.get("tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "Load"):
            tok_ckpt = os.path.join(path, "tokenizer.ckpt")
            if not os.path.exists(tok_ckpt):
                # fallback: pretrain_folder key in hparams
                pretrain_folder = self.hparams.get("pretrain_folder", None)
                if pretrain_folder:
                    tok_ckpt = os.path.join(pretrain_folder, "tokenizer.ckpt")
            if os.path.exists(tok_ckpt):
                print(f"Loading tokenizer from: {tok_ckpt}")
                tokenizer.Load(tok_ckpt)

        # Eval mode
        self.hparams["model"].eval()
        for module in self.mods.values():
            if hasattr(module, 'eval'):
                module.eval()
        
        print("Model loaded!")

    def _get_audio_stream(self, streamer: "torchaudio.io.StreamReader", frames_per_chunk: int, sample_rate: int = 16000):

        stream_infos = [
            streamer.get_src_stream_info(i)
            for i in range(streamer.num_src_streams)
        ]

        audio_stream_infos = [
            (i, stream_info)
            for i, stream_info in enumerate(stream_infos)
            if stream_info.media_type == "audio"
        ]

        if len(audio_stream_infos) != 1:
            raise ValueError(
                f"Expected stream to have only 1 stream (with any number of channels), got {len(audio_stream_infos)} (with streams: {stream_infos})"
            )

        # find the index of the first (and only) audio stream
        audio_stream_index = audio_stream_infos[0][0]

        # output stream #0
        streamer.add_basic_audio_stream(
            frames_per_chunk=frames_per_chunk,
            stream_index=audio_stream_index,
            sample_rate= sample_rate,
            format="fltp",  # torch.float32
            num_channels=1,
        )

        for (chunk,) in streamer.stream():
            chunk = chunk.squeeze(-1)  # we deal with mono, remove that dim
            chunk = chunk.unsqueeze(0)  # create a fake batch dim
            yield chunk

    def get_chunk_size_frames(
        self, dynchunktrain_config: DynChunkTrainConfig
    ) -> int:
        """Returns the chunk size in actual audio samples, i.e. the exact
        expected length along the time dimension of an input chunk tensor (as
        passed to :meth:`~StreamingASR.encode_chunk` and similar low-level
        streaming functions).

        Arguments
        ---------
        dynchunktrain_config : DynChunkTrainConfig
            The streaming configuration to determine the chunk frame count of.

        Returns
        -------
        chunk size
        """

        return (self.filter_props.stride - 1) * dynchunktrain_config.chunk_size

    def transcribe_file_streaming(
        self,
        path,
        dynchunktrain_config: DynChunkTrainConfig,
        use_torchaudio_streaming: bool = True,
        **kwargs,
    ):
        """Transcribes the given audio file into a sequence of words, in a
        streaming fashion, meaning that text is being yield from this
        generator, in the form of strings to concatenate.

        Arguments
        ---------
        path : str
            URI/path to the audio to transcribe. When
            ``use_torchaudio_streaming`` is ``False``, uses SB fetching to allow
            fetching from HF or a local file. When ``True``, resolves the URI
            through ffmpeg, as documented in
            :class:`torchaudio.io.StreamReader`.
        dynchunktrain_config : DynChunkTrainConfig
            Streaming configuration. Sane values and how much time chunks
            actually represent is model-dependent.
        use_torchaudio_streaming : bool
            Whether the audio file can be loaded in a streaming fashion. If not,
            transcription is still performed through chunks of audio, but the
            entire audio file is fetched and loaded at once.
            This skips the usual fetching method and instead resolves the URI
            using torchaudio (via ffmpeg).
        **kwargs : dict
            Arguments forwarded to ``load_audio``

        Yields
        ------
        generator of str
            An iterator yielding transcribed chunks (strings). There is a yield
            for every chunk, even if the transcribed string for that chunk is an
            empty string.
        """

        chunk_size = self.get_chunk_size_frames(dynchunktrain_config)

        if use_torchaudio_streaming:
            streamer = torchaudio.io.StreamReader(path)
            chunks = self._get_audio_stream(streamer, chunk_size)
        else:
            waveform = self.load_audio(path, **kwargs)
            batch = waveform.unsqueeze(0)  # create batch dim
            chunks = split_fixed_chunks(batch, chunk_size)

        rel_length = torch.tensor([1.0])
        context = self.make_streaming_context(dynchunktrain_config)

        final_chunks = (
            [torch.zeros((1, chunk_size), device=self.device)]
            * self.hparams['fea_streaming_extractor'].get_recommended_final_chunk_count(
                chunk_size
            )
        )

        for chunk in itertools.chain(chunks, final_chunks):
            predicted_words = self.transcribe_chunk(context, chunk, rel_length)
            yield predicted_words[0]

    def transcribe_chunk(
        self,
        context: ASRStreamingContext,
        chunk: torch.Tensor,
        chunk_len: Optional[torch.Tensor] = None,
    ):
        """Transcription of a batch of audio chunks into transcribed text.
        Must be called over a given context in the correct order of chunks over
        time.

        Arguments
        ---------
        context : ASRStreamingContext
            Mutable streaming context object, which must be specified and reused
            across calls when streaming.
            You can obtain an initial context by calling
            `asr.make_streaming_context(config)`.
        chunk : torch.Tensor
            The tensor for an audio chunk of shape `[batch size, time]`.
            The time dimension must strictly match
            `asr.get_chunk_size_frames(config)`.
            The waveform is expected to be in the model's expected format (i.e.
            the sampling rate must be correct).
        chunk_len : torch.Tensor, optional
            The relative chunk length tensor of shape `[batch size]`. This is to
            be used when the audio in one of the chunks of the batch is ending
            within this chunk.
            If unspecified, equivalent to `torch.ones((batch_size,))`.

        Returns
        -------
        str
            Transcribed string for this chunk, might be of length zero.
        """

        if chunk_len is None:
            chunk_len = torch.ones((chunk.size(0),))

        chunk = chunk.float()
        chunk, chunk_len = chunk.to(self.device), chunk_len.to(self.device)

        x = self.encode_chunk(context, chunk, chunk_len)
        words, _ = self.decode_chunk(context, x)

        return words

    @torch.no_grad()
    def encode_chunk(
        self,
        context: ASRStreamingContext,
        chunk: torch.Tensor,
        chunk_len: Optional[torch.Tensor] = None,
    ):
        if chunk_len is None:
            chunk_len = torch.ones((chunk.size(0),))

        chunk = chunk.float()
        chunk, chunk_len = chunk.to(self.device), chunk_len.to(self.device)

        # Feature extraction (streaming)
        x = self.hparams['fea_streaming_extractor'](
            chunk, context=context.fea_extractor_context, lengths=chunk_len
        )

        # Encoder (streaming)
        x, hidden = self.mods["enc"].forward_streaming(x, context.encoder_context)
        x = self.mods["proj_enc"](x)
        hidden = torch.cat(hidden, dim=0)
        if self.output_hidden_states:
            return x, hidden  # [B, T, D], [num_layers, B, D]
        else:
            return x  # [B, T, D]
    
    @torch.no_grad()
    def decode_chunk(
        self, context: ASRStreamingContext, x: torch.Tensor
    ) -> Tuple[List[str], List[List[int]]]:
        """Decodes the output of the encoder into tokens and the associated
        transcription.
        Must be called over a given context in the correct order of chunks over
        time.

        Arguments
        ---------
        context : ASRStreamingContext
            Mutable streaming context object, which should be the same object
            that was passed to `encode_chunk`.

        x : torch.Tensor
            The output of `encode_chunk` for a given chunk.

        Returns
        -------
        list of str
            Decoded tokens of length `batch_size`. The decoded strings can be
            of 0-length.
        list of list of output token hypotheses
            List of length `batch_size`, each holding a list of tokens of any
            length `>=0`.
        """
        tokens = self.hparams['decoding_function'](x, context.decoder_context)
        tokenizer_decode_streaming = self.hparams.get('tokenizer_decode_streaming', None)
        if tokenizer_decode_streaming is not None and context.tokenizer_context is not None:
            words = [
                tokenizer_decode_streaming(self.tokenizer, cur_tokens, context.tokenizer_context)
                for cur_tokens in tokens
            ]
        else:
            words = [self.tokenizer.decode(cur_tokens) for cur_tokens in tokens]

        return words, tokens
    
    @torch.no_grad()
    def get_ctc_log_probs(self, enc_out):
        """Get CTC log probs from encoder output."""
        ctc_out = self.mods["proj_ctc"](enc_out)
        log_probs = F.log_softmax(ctc_out, dim=-1)
        return log_probs  # [B, T, vocab]
    
    def make_streaming_context(self, config: DynChunkTrainConfig):
        fea_streaming_extractor = self.hparams.get("fea_streaming_extractor", None)
        make_decoder_streaming_context = self.hparams.get('make_decoder_streaming_context', None)
        make_tokenizer_streaming_context = self.hparams.get('make_tokenizer_streaming_context', None)

        return ASRStreamingContext(
            config=config,
            fea_extractor_context=fea_streaming_extractor.make_streaming_context(),
            encoder_context=self.mods["enc"].make_streaming_context(config),
            decoder_context=make_decoder_streaming_context(),
            tokenizer_context=make_tokenizer_streaming_context() if make_tokenizer_streaming_context else None,
        )


# ============================================================================
# Acoustic Head
# ============================================================================

class AcousticHead(nn.Module):
    def __init__(self, dim=768, out_dim=256, bottleneck=256, dropout=0.1):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=1),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.mlp_block = nn.Sequential(
            nn.Linear(dim, bottleneck * 2),
            nn.GLU(),
            nn.Linear(bottleneck, out_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = self.conv_block(x.transpose(1, 2))
        x = x.transpose(1, 2)
        x = self.norm1(x)
        x = self.mlp_block(x)
        x = self.norm2(x)
        return x


# ============================================================================
# Data2Vec Semantic-Acoustic Model
# ============================================================================

class Data2VecSemanticAcousticModel(nn.Module):
    def __init__(self, vocab_size: int = None, *,
                 nhead: int = 8, num_encoder_layers: int = 4, dim_feedforward: int = 4096,
                 dropout: float = 0.1, spk_emb_dim: int = 192, chunk_size: int = 16,
                 left_context: int = 8, hparams_file: str = None, checkpoint_path: str = None,
                 codebook_size: int = 512):
        from utils.text_utils import (
            VOCAB_SIZE, CHAR_VOCAB_SIZE, CHAR_TO_IDX, _char_tokenizer,
            batch_word_char_alignment, compute_model_feat_len,
        )
        super().__init__()

        if vocab_size is None:
            vocab_size = VOCAB_SIZE

        self._VOCAB_SIZE = VOCAB_SIZE
        self._CHAR_VOCAB_SIZE = CHAR_VOCAB_SIZE
        self._CHAR_TO_IDX = CHAR_TO_IDX
        self._char_tokenizer = _char_tokenizer
        self._batch_word_char_alignment = batch_word_char_alignment
        self._compute_model_feat_len = compute_model_feat_len

        self.encoder = StreamingCharModel(
            hparams_file=hparams_file,
            checkpoint_path=checkpoint_path,
            output_hidden_states=True,
        )
        self.chunk_config = DynChunkTrainConfig(
            chunk_size=chunk_size, left_context_size=left_context
        )

        d_enc = 640      # proj_enc output dimension
        d_hidden = 512   # transformer hidden dim
        self.num_layers = 12
        self.concat_layers = self.num_layers // 2

        feat_dim = 256
        word_emb_dim = 1024

        self.acoustic_head = nn.Sequential(
            nn.Linear(d_hidden * self.concat_layers, d_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, 256),
        )

        self.agg_token = nn.Parameter(torch.randn(1, 1, feat_dim))

        self.word_agg_transformer = self._build_word_aggregation_transformer(
            d_model=feat_dim, nhead=8, num_layers=2,
            dim_feedforward=512, dropout=dropout,
        )

        self.char_agg_token = nn.Parameter(torch.randn(1, 1, word_emb_dim))

        self.char_agg_transformer = self._build_word_aggregation_transformer(
            d_model=word_emb_dim, nhead=8, num_layers=2,
            dim_feedforward=512, dropout=dropout,
        )

        self.register_buffer(
            "word_pos_encoding",
            self._get_sinusoidal_positional_encoding(2048, 256),
        )
        self.register_buffer(
            "char_pos_encoding",
            self._get_sinusoidal_positional_encoding(256, word_emb_dim),
        )

        self.residual_vq = ResidualVQ(
            dim=feat_dim,
            num_quantizers=3,
            codebook_size=codebook_size,
            kmeans_init=True,
            kmeans_iters=100,
            threshold_ema_dead_code=2,
            commitment_weight=0.25,
        )

        self.char_embedding = nn.Embedding(CHAR_VOCAB_SIZE, word_emb_dim)

        in_seg = feat_dim + word_emb_dim + spk_emb_dim
        self.spk_emb_dim = spk_emb_dim

        self.nhead = nhead
        self.unit_head_pre = nn.Linear(in_seg, d_hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_hidden, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=6)
        self.unit_head_post = nn.Linear(d_hidden, vocab_size)

        self.register_buffer(
            "positional_encoding",
            self._get_sinusoidal_positional_encoding(2048, d_hidden),
        )

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    def _build_word_aggregation_transformer(
        self, d_model, nhead, num_layers, dim_feedforward, dropout
    ):
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def _get_sinusoidal_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    # ------------------------------------------------------------------ #
    #  RNN-T greedy decode with frame-level alignment tracking
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _rnnt_greedy_decode_with_alignment(
        self, enc_out, hidden_state=None, max_symbols_per_step=5
    ):
        """Greedy RNN-T decode, tracking frame-level emission positions."""
        searcher = self.encoder.hparams["Greedysearcher"]
        B, T, _ = enc_out.shape
        device = enc_out.device

        predictions = [[] for _ in range(B)]
        frame_positions = [[] for _ in range(B)]

        input_PN = torch.ones((B, 1), device=device, dtype=torch.int32) * searcher.blank_id

        if hidden_state is None:
            out_PN, hidden = searcher._forward_PN(input_PN, searcher.decode_network_lst)
        else:
            out_PN, hidden = hidden_state

        for t_step in range(T):
            count = 0
            while count <= max_symbols_per_step:
                log_probs = searcher._joint_forward_step(
                    enc_out[:, t_step, :].unsqueeze(1).unsqueeze(1),
                    out_PN.unsqueeze(1),
                )
                _, positions = torch.max(log_probs.squeeze(1).squeeze(1), dim=1)

                have_update = []
                for i in range(B):
                    if positions[i].item() != searcher.blank_id:
                        predictions[i].append(positions[i].item())
                        frame_positions[i].append(t_step)
                        input_PN[i][0] = positions[i]
                        have_update.append(i)

                if have_update:
                    selected_input_PN, selected_hidden = searcher._get_sentence_to_update(
                        have_update, input_PN, hidden
                    )
                    selected_out_PN, selected_hidden = searcher._forward_PN(
                        selected_input_PN, searcher.decode_network_lst, selected_hidden,
                    )
                    out_PN[have_update] = selected_out_PN
                    hidden = searcher._update_hiddens(have_update, selected_hidden, hidden)
                else:
                    break
                count += 1

        return predictions, frame_positions, (out_PN, hidden)

    def _build_alignment_from_rnnt(
        self, predictions, frame_positions, T_speech,
        tokenizer, txt_normalizer, hubert_feat_lens=None,
    ):
        """Convert RNN-T decode output into alignment matrices and char_data."""
        B = len(predictions)
        device = next(self.parameters()).device

        texts = [self.encoder.tokenizer.decode(predictions[b]) for b in range(B)]

        char_batch_meta = self._batch_word_char_alignment(tokenizer, texts)

        char_data = []
        for text_norm, cd in zip(texts, char_batch_meta):
            word_token_ids = tokenizer.encode(text_norm, add_special_tokens=False)
            char_data.append({
                "char_sequence":    cd["char_sequence"],
                "char_indices":     cd["char_indices"],
                "char_to_word_map": cd["char_to_word_map"],
                "tokens":           cd["tokens"],
                "word_token_ids":   word_token_ids,
            })

        max_char = max((len(cd["char_indices"]) for cd in char_data), default=0)
        max_word = max(
            ((max(cd["char_to_word_map"]) + 1) if cd["char_to_word_map"] else 0)
            for cd in char_data
        ) if char_data else 0

        char_alignment = torch.zeros(B, max_char, T_speech, dtype=torch.bool, device=device)
        word_alignment = torch.zeros(B, max_word, T_speech, dtype=torch.bool, device=device)

        for b in range(B):
            fps = frame_positions[b]
            n_chars = len(char_data[b]["char_indices"])
            c2w = char_data[b]["char_to_word_map"]
            n_align = n_chars
            if n_align == 0:
                continue
            hubert_feat_len = hubert_feat_lens[b].item()
            for i in range(n_align):
                start = fps[i - 1] + 1 if i > 0 else 0
                end = fps[i] + 1 if i < n_align - 1 else hubert_feat_len
                if start >= end:
                    end = start + 1
                char_alignment[b, i, start:end] = True
                w = c2w[i]
                if 0 <= w < max_word:
                    word_alignment[b, w, start:end] = True

        gt_char_indices = torch.full(
            (B, max_char), self._CHAR_TO_IDX[" "], dtype=torch.long, device=device
        )
        for b in range(B):
            n_chars = len(char_data[b]["char_indices"])
            gt_char_indices[b, :n_chars] = torch.tensor(
                char_data[b]["char_indices"], device=device
            )

        return char_alignment, word_alignment, char_data, gt_char_indices

    # ------------------------------------------------------------------ #
    #  Forward passes
    # ------------------------------------------------------------------ #
    def forward(
        self,
        waveforms: torch.Tensor,
        wav_mask: torch.Tensor,
        spk_emb=None,
        detach: bool = True,
        tokenizer=None,
        txt_normalizer=None,
        epoch: int = 0,
        char_ids_list=None,
        constraints_list=None,
        vq_blend: float = 1.0,
    ):
        """Forward pass.

        When ``char_ids_list`` and ``constraints_list`` are provided (derived from
        chunk-level TextGrid files), the alignment is computed via the constrained
        Viterbi algorithm instead of unconstrained greedy decoding.  This gives
        much more accurate frame-to-character alignment during training.

        Parameters
        ----------
        char_ids_list : list[list[int] or None], optional
            Per-sample target character IDs from TextGrid.
        constraints_list : list[list[tuple] or None], optional
            Per-sample RNNT waypoint constraints ``(t_s, t_e, u_s, u_e)``.
        """
        input_len = (wav_mask == True).sum(dim=1)
        input_len_relative = input_len.float() / wav_mask.shape[1]
        enc_out, hidden = self.encoder(waveforms, input_len_relative, self.chunk_config, epoch=epoch)
        top_half_layers = hidden[1:(self.concat_layers + 1)]
        h_aco_raw = torch.cat(top_half_layers, dim=-1)
        if detach:
            enc_out = enc_out.detach()
            h_aco_raw = h_aco_raw.detach()

        hubert_feat_lens = torch.tensor(
            [self._compute_model_feat_len(int(L)) for L in input_len],
            dtype=input_len.dtype,
            device=input_len.device,
        )

        z_raw = self.acoustic_head(h_aco_raw)

        if spk_emb is None:
            spk_emb = waveforms.new_zeros(waveforms.size(0), self.spk_emb_dim)

        T_speech = enc_out.size(1)

        # Choose alignment strategy
        use_constrained = (
            char_ids_list is not None
            and constraints_list is not None
            and any(c is not None for c in char_ids_list)
        )
        if use_constrained:
            from utils.rnnt_align_loss import constrained_rnnt_align
            hparams = self.encoder.hparams
            blank_id = hparams["blank_index"]
            predictions, frame_positions = constrained_rnnt_align(
                enc_out.detach(),
                hubert_feat_lens,
                char_ids_list,
                constraints_list,
                hparams,
                blank_id,
            )
        else:
            raise Exception("Unconstrained RNN-T alignment is not supported in forward pass. Please provide char_ids_list and constraints_list for constrained alignment.")

        char_alignment, word_alignment, char_data, gt_char_indices = \
            self._build_alignment_from_rnnt(
                predictions, frame_positions, T_speech,
                tokenizer, txt_normalizer, hubert_feat_lens,
            )

        out = {
            "z_raw": z_raw,
            "char_alignment": char_alignment,
            "word_alignment": word_alignment,
        }

        if char_alignment.any() and word_alignment.any():
            segment_out = self._segment_pass(
                z_raw, char_alignment, word_alignment, char_data,
                gt_char_indices, spk_emb, vq_blend=vq_blend,
            )
            out.update(segment_out)

        return out

    def _get_audio_stream(self, streamer, frames_per_chunk, sample_rate=16000):
        """Yield audio chunks from torchaudio StreamReader."""
        stream_infos = [streamer.get_src_stream_info(i)
                        for i in range(streamer.num_src_streams)]
        audio_idx = [i for i, s in enumerate(stream_infos) if s.media_type == "audio"]
        if len(audio_idx) != 1:
            raise ValueError(f"Expected 1 audio stream, got {len(audio_idx)}")
        streamer.add_basic_audio_stream(
            frames_per_chunk=frames_per_chunk,
            stream_index=audio_idx[0],
            sample_rate=sample_rate,
            format="fltp",
            num_channels=1,
        )
        for (chunk,) in streamer.stream():
            yield chunk.squeeze(-1).unsqueeze(0)

    @staticmethod
    def _extract_segment_constraints(char_ids, constraints, frame_start, frame_end):
        """Return char_ids and constraints clipped to [frame_start, frame_end)."""
        seg_constraints = []
        u_min = u_max = None
        for t_s, t_e, u_s, u_e in constraints:
            if t_e <= frame_start or t_s >= frame_end:
                continue
            t_s_local = max(t_s, frame_start) - frame_start
            t_e_local = min(t_e, frame_end)   - frame_start
            seg_constraints.append((t_s_local, t_e_local, u_s, u_e))
            u_min = u_s if u_min is None else min(u_min, u_s)
            u_max = u_e if u_max is None else max(u_max, u_e)
        if not seg_constraints:
            return [], []
        seg_char_ids = char_ids[u_min:u_max]
        seg_constraints = [
            (t_s, t_e, u_s - u_min, u_e - u_min)
            for t_s, t_e, u_s, u_e in seg_constraints
        ]
        return seg_char_ids, seg_constraints

    @torch.no_grad()
    def forward_streaming(
        self,
        wav_path,
        spk_emb=None,
        txt_normalizer=None,
        tokenizer=None,
        word_asr_model=None,
        verbose=False,
        boundary_classifier=None,
    ):
        """Chunk-by-chunk streaming forward using constrained RNN-T alignment.

        For each chunk the word-model (``word_asr_model``) is queried for the
        committed text exactly as in ``generate_chunk_textgrids.py``.  That
        text is used to build per-chunk constraints for ``constrained_rnnt_align``,
        giving frame-accurate character alignment without a separate TextGrid.
        Word-boundary splitting and ``expanded_forward_streaming`` are unchanged.
        """
        from utils.rnnt_align_loss import constrained_rnnt_align

        device = next(self.parameters()).device
        chunk_size = self.encoder.get_chunk_size_frames(self.chunk_config)

        streamer = torchaudio.io.StreamReader(wav_path)
        chunks = self._get_audio_stream(streamer, chunk_size, sample_rate=16000)
        rel_length = torch.tensor([1.0])

        context_enc  = self.encoder.make_streaming_context(self.chunk_config)
        word_context = word_asr_model.make_streaming_context(self.chunk_config)

        hparams  = self.encoder.hparams
        blank_id = hparams["blank_index"]

        final_chunk_count = (
            self.encoder.hparams["fea_streaming_extractor"]
            .get_recommended_final_chunk_count(chunk_size)
        )
        final_chunks = [torch.zeros((1, chunk_size), device=device)] * final_chunk_count

        if spk_emb is None:
            spk_emb = torch.zeros(1, self.spk_emb_dim, device=device)

        acc_tokens, acc_fps, acc_z_raw = [], [], None
        acc_z_start = frame_offset = 0

        past_hidden = past_mask = past_word_ids = None
        word_offset = 0
        all_u_logits = []

        all_chunks = list(itertools.chain(chunks, final_chunks))
        total = len(all_chunks)

        enc_proj_buf: List[torch.Tensor] = []
        hook_handle = None
        if boundary_classifier is not None:
            hook_handle = word_asr_model.mods["proj_enc"].register_forward_hook(
            lambda _m, _i, out: enc_proj_buf.append(out.detach()))

        for idx, chunk in enumerate(all_chunks):
            chunk = chunk.to(device)
            is_last_chunk = (idx == total - 1)

            enc_out, enc_hidden = self.encoder.encode_chunk(context_enc, chunk, rel_length)
            T_c = enc_out.size(1)

            # Word-model text for this chunk (mirrors generate_chunk_textgrids.py)
            enc_proj_buf.clear()
            chunk_output = word_asr_model.transcribe_chunk(word_context, chunk)
            new_text   = chunk_output[0] if chunk_output else ""
            chunk_text = new_text.replace("\u2581", " ")   # ▁ → space

            top_half = enc_hidden[1:(self.concat_layers + 1)]
            h_aco = torch.cat(list(top_half), dim=-1).unsqueeze(0)
            z_raw_chunk = self.acoustic_head(h_aco)
            
            # Boundary classification (if enabled) to decide whether to trust the word-model text for alignment or not.
            is_boundary: Optional[bool] = None
            if boundary_classifier is not None and new_text and enc_proj_buf:
                enc_proj = enc_proj_buf[0]                              # [..., T_chunk, joint_dim]
                enc_feat = enc_proj.reshape(-1, enc_proj.shape[-1])[-1].unsqueeze(0)  # [1, joint_dim]
                pred_feat = compute_pred_feat(word_context, device)                           # [1, joint_dim]
                logits = boundary_classifier(enc_feat.to(device), pred_feat)          # [1, 2]
                is_boundary = logits.argmax(-1).item() == 1

            # Per-chunk constrained alignment using word-model text as constraint.
            # Build char_ids without stripping so spaces are preserved in acc_tokens
            # and word-boundary detection via decoded_text.rfind(" ") still works.
            char_ids_chunk = [
                self._CHAR_TO_IDX[ch] for ch in chunk_text.lower()
                if ch in self._CHAR_TO_IDX
            ]
            if char_ids_chunk:
                preds_chunk, fps_chunk = constrained_rnnt_align(
                    enc_out.detach(),
                    torch.tensor([T_c], device=device),
                    [char_ids_chunk],
                    [[(0, T_c, 0, len(char_ids_chunk))]],
                    hparams,
                    blank_id,
                )
                chunk_tokens = preds_chunk[0]
                global_fps   = [fp + frame_offset for fp in fps_chunk[0]]
            else:
                chunk_tokens = []
                global_fps   = []

            acc_tokens.extend(chunk_tokens)
            acc_fps.extend(global_fps)
            acc_z_raw = (
                torch.cat([acc_z_raw, z_raw_chunk], dim=1)
                if acc_z_raw is not None else z_raw_chunk
            )
            frame_offset += T_c

            if verbose and chunk_tokens:
                print(f"[{idx}] word_model='{chunk_text}' aligned='{self._char_tokenizer.decode(chunk_tokens)}'")

            decoded_text = self._char_tokenizer.decode(acc_tokens)
            if is_last_chunk or (not chunk_text.strip() and decoded_text.strip()) or (is_boundary is True):
                # Flush tokens/fps; split z_raw so the empty chunk's frames
                # are carried forward for the next word.
                proc_tokens = acc_tokens
                proc_fps    = acc_fps
                acc_tokens, acc_fps = [], []
                if is_last_chunk or (is_boundary is True):
                    seg_z_raw   = acc_z_raw
                    seg_start   = acc_z_start
                    T_seg       = seg_z_raw.size(1) if seg_z_raw is not None else 0
                    acc_z_raw   = None
                    acc_z_start = frame_offset
                    if verbose:
                        print(f"  final chunk reached.")
                        print(f"  emit:  '{self._char_tokenizer.decode(proc_tokens)}'")
                        print(f"  carry: ''")
                else:
                    # z_raw_chunk was already appended; split it off so it
                    # stays in the accumulator for the next segment.
                    T_before  = acc_z_raw.size(1) - z_raw_chunk.size(1) if acc_z_raw is not None else 0
                    seg_z_raw = acc_z_raw[:, :T_before, :] if T_before > 0 else None
                    seg_start = acc_z_start
                    T_seg     = T_before
                    acc_z_raw   = z_raw_chunk          # empty chunk's frames kept
                    acc_z_start = frame_offset - T_c   # global start of empty chunk
                    if verbose:
                        print(f"  emit:  '{self._char_tokenizer.decode(proc_tokens)}'")
                        print(f"  carry: ''")
            else:
                last_space = decoded_text.rfind(" ")
                if last_space <= 0:
                    continue

                split_tok     = last_space
                proc_tokens   = acc_tokens[:split_tok]
                proc_fps      = acc_fps[:split_tok]
                remain_tokens = acc_tokens[split_tok:]
                remain_fps    = acc_fps[split_tok:]

                last_proc_fp       = proc_fps[-1]
                split_frame_global = last_proc_fp + 1
                split_local = max(0, min(
                    split_frame_global - acc_z_start,
                    acc_z_raw.size(1),
                ))

                seg_z_raw = acc_z_raw[:, :split_local, :]
                seg_start = acc_z_start
                T_seg     = split_local

                acc_z_raw   = (
                    acc_z_raw[:, split_local:, :]
                    if split_local < acc_z_raw.size(1) else None
                )
                acc_z_start = split_frame_global
                acc_tokens  = remain_tokens
                acc_fps     = remain_fps

                if verbose:
                    print(f"  emit:  '{self._char_tokenizer.decode(proc_tokens)}'")
                    print(f"  carry: '{self._char_tokenizer.decode(remain_tokens)}'")

            if not proc_tokens or seg_z_raw is None or T_seg == 0:
                continue

            local_fps = [max(0, fp - seg_start) for fp in proc_fps]

            char_alignment, word_alignment, seg_char_data, gt_char_indices = \
                self._build_alignment_from_rnnt(
                    [proc_tokens], [local_fps], T_seg,
                    tokenizer, txt_normalizer,
                    hubert_feat_lens=torch.tensor([T_seg]),
                )

            if not char_alignment.any() or not word_alignment.any():
                continue

            B_s = 1
            T_words = word_alignment.size(1)
            T_chars = char_alignment.size(1)

            aggregated_segments, segment_info = self._pool_to_word_segments_transformer(
                seg_z_raw, word_alignment, char_alignment
            )
            feat_dim = aggregated_segments.size(1)
            seg_z = self._reconstruct_word_features_batch(
                aggregated_segments, segment_info, B_s, T_words, feat_dim, device
            )

            seg_char_emb = self._get_char_embeddings(gt_char_indices)

            word_to_char_alignment = torch.zeros(B_s, T_words, T_chars, device=device)
            for b in range(B_s):
                c2w = seg_char_data[b]["char_to_word_map"]
                c2w_t = torch.tensor(c2w, dtype=torch.long, device=device)
                char_one_hot = F.one_hot(c2w_t, num_classes=T_words)
                wc = char_one_hot.T.float()
                word_to_char_alignment[b, :, :wc.size(1)] = wc

            char_lens = torch.tensor(
                [len(d["char_to_word_map"]) for d in seg_char_data], device=device
            )
            c_idx = torch.arange(T_chars, device=device)
            valid_mask = c_idx.unsqueeze(0) < char_lens.unsqueeze(1)
            char_to_char_alignment = (
                valid_mask[:, None, :] & valid_mask[:, :, None]
            ).float()

            aggregated_char_embs, char_segment_info = \
                self._pool_character_segments_transformer(
                    seg_char_emb, word_to_char_alignment, char_to_char_alignment
                )
            feat_dim = aggregated_char_embs.size(1)
            seg_word_emb = self._reconstruct_word_features_batch(
                aggregated_char_embs, char_segment_info, B_s, T_words, feat_dim, device
            )

            seg_z_q, _, _ = self.residual_vq(seg_z)

            seg_feat = torch.cat([
                seg_word_emb,
                seg_z_q,
                spk_emb.unsqueeze(1).expand(-1, T_words, -1),
            ], dim=-1)
            exp_feat = self.expand_word_features(seg_feat, word_alignment)
            exp_mask = ~(char_alignment.any(1))

            seg_u_logits, past_hidden, past_mask, past_word_ids, word_offset = \
                self.expanded_forward_streaming(
                    exp_feat, exp_mask, word_alignment, word_offset,
                    past_hidden, past_mask, past_word_ids,
                )
            all_u_logits.append(seg_u_logits)

        if not all_u_logits:
            return None
        return torch.cat(all_u_logits, dim=2)

    def extract_rvq_indices_inference(
        self,
        waveforms: torch.Tensor,
        tokenizer,
        normalizer,
        wav_mask: torch.Tensor = None,
    ) -> dict:
        if wav_mask is None:
            wav_mask = torch.ones_like(waveforms, dtype=torch.bool)
        return self.extract_rvq_indices(
            waveforms=waveforms,
            wav_mask=wav_mask,
            tokenizer=tokenizer,
            txt_normalizer=normalizer,
            detach=True,
        )

    # ------------------------------------------------------------------ #
    #  Segment pass
    # ------------------------------------------------------------------ #
    def _segment_pass(
        self, z_raw, char_alignment, word_alignment, char_data, gt_char_indices, spk_emb,
        vq_blend: float = 1.0,
    ):
        aggregated_segments, segment_info = self._pool_to_word_segments_transformer(
            z_raw, word_alignment, char_alignment
        )

        B, T_words, T_chars = (
            word_alignment.size(0), word_alignment.size(1), char_alignment.size(1)
        )
        feat_dim = aggregated_segments.size(1)
        device = aggregated_segments.device

        seg_z = self._reconstruct_word_features_batch(
            aggregated_segments, segment_info, B, T_words, feat_dim, device
        )

        seg_char_emb = self._get_char_embeddings(gt_char_indices)

        word_to_char_alignment = torch.zeros(B, T_words, T_chars, device=device)
        for b in range(B):
            char_to_word = torch.tensor(
                char_data[b]["char_to_word_map"], dtype=torch.long, device=device
            )
            char_one_hot = F.one_hot(char_to_word, num_classes=T_words)
            word_char_tensor = char_one_hot.T.float()
            num_char = word_char_tensor.size(1)
            word_to_char_alignment[b, :, :num_char] = word_char_tensor

        char_lens = torch.tensor(
            [len(d["char_to_word_map"]) for d in char_data], device=device
        )
        idx = torch.arange(T_chars, device=device)
        valid_mask = idx.unsqueeze(0) < char_lens.unsqueeze(1)
        char_to_char_alignment = (
            valid_mask[:, None, :] & valid_mask[:, :, None]
        ).float()

        aggregated_char_embs, char_segment_info = self._pool_character_segments_transformer(
            seg_char_emb, word_to_char_alignment, char_to_char_alignment
        )

        feat_dim = aggregated_char_embs.size(1)
        seg_word_emb = self._reconstruct_word_features_batch(
            aggregated_char_embs, char_segment_info, B, T_words, feat_dim, device
        )
        # Only quantize non-padding word positions to avoid wasting
        # codebook entries on zeros.
        word_mask = word_alignment.any(dim=2)  # [B, T_words] — True for real words
        valid_z = seg_z[word_mask]             # [N_valid, feat_dim]
        if valid_z.size(0) > 0:
            valid_q, _, vq_loss = self.residual_vq(valid_z.unsqueeze(0))
            valid_q = valid_q.squeeze(0)
            seg_z_q = torch.zeros_like(seg_z)
            seg_z_q[word_mask] = valid_q
            recon_loss = F.mse_loss(valid_q, valid_z.detach())
        else:
            seg_z_q = torch.zeros_like(seg_z)
            vq_loss = torch.tensor(0.0, device=seg_z.device)
            recon_loss = torch.tensor(0.0, device=seg_z.device)
        # vq_blend controls how much quantized features are used:
        #   0.0 = pure seg_z (phase 1 / start of gradual warmup)
        #   1.0 = pure seg_z_q (full RVQ)
        #   0<x<1 = linear blend (gradual warmup)
        seg_z_used = (1.0 - vq_blend) * seg_z + vq_blend * seg_z_q
        B, T_words, _ = seg_z_used.shape
        seg_feat = torch.cat([
            seg_word_emb,
            seg_z_used,
            spk_emb.unsqueeze(1).expand(-1, T_words, -1),
        ], dim=-1)

        exp_feat = self.expand_word_features(seg_feat, word_alignment)
        exp_mask = ~(char_alignment.any(1))
        u_logits = self.expanded_forward(exp_feat, exp_mask, word_alignment)
        return {
            "vq_loss": vq_loss.mean() if vq_loss.dim() > 0 else vq_loss,
            "recon_loss": recon_loss,
            "u_logits": u_logits,
            "seg_char_emb": seg_word_emb,
            "seg_feat": seg_feat,
        }

    # ------------------------------------------------------------------ #
    #  Pooling helpers
    # ------------------------------------------------------------------ #
    def _pool_to_word_segments_transformer(
        self,
        features: torch.Tensor,
        word_alignment: torch.Tensor,
        char_alignment: torch.Tensor,
    ):
        B, T_word, T_speech = word_alignment.shape
        feat_dim = features.size(-1)
        device = features.device

        flat_mask = word_alignment.view(-1, T_speech)
        has_frames = flat_mask.any(dim=1)
        if not has_frames.any():
            return torch.zeros(B, T_word, feat_dim, device=device), []

        seg_idx = has_frames.nonzero(as_tuple=False).squeeze(1)
        n_seg = seg_idx.numel()
        seg_info = [(idx // T_word, idx % T_word) for idx in seg_idx.tolist()]

        word_segments = []
        for i in seg_idx:
            mask_i = flat_mask[i]
            seg_feat_i = features[i // T_word, mask_i]
            word_segments.append(seg_feat_i)

        seg_padded = pad_sequence(word_segments, batch_first=True)
        seg_lengths = torch.tensor(
            [x.size(0) for x in word_segments], device=device
        )

        max_len = seg_padded.size(1)
        agg_tok = self.agg_token.expand(n_seg, 1, feat_dim)
        padded = torch.cat([agg_tok, seg_padded], dim=1)

        steps = torch.arange(max_len + 1, device=device).unsqueeze(0)
        pad_mask = steps >= (seg_lengths + 1).unsqueeze(1)

        pos_enc = self.word_pos_encoding[:, :max_len + 1, :].to(device)
        fused = padded + pos_enc
        out = self.word_agg_transformer(fused, src_key_padding_mask=pad_mask)

        return out[:, 0], seg_info

    def _pool_character_segments_transformer(
        self,
        features: torch.Tensor,
        word_alignment: torch.Tensor,
        char_alignment: torch.Tensor,
    ):
        B, T_words, T_speech = word_alignment.shape
        feat_dim = features.size(-1)
        device = features.device

        word_segments, segment_lengths, segment_info, duration_frames = \
            self._extract_word_segments_batch(features, word_alignment, char_alignment)

        if not word_segments:
            return torch.zeros(B, T_words, feat_dim, device=device), []

        max_len = max(segment_lengths)
        n_seg = len(word_segments)

        padded = torch.zeros(n_seg, max_len + 1, feat_dim, device=device)
        pad_mask = torch.ones(n_seg, max_len + 1, device=device, dtype=torch.bool)

        padded[:, 0] = self.char_agg_token
        pad_mask[:, 0] = False

        for i, seg in enumerate(word_segments):
            L = seg.size(0)
            padded[i, 1:1 + L] = seg
            pad_mask[i, :1 + L] = False

        pos_enc = self.char_pos_encoding[:, :max_len + 1].to(device)
        padded = padded + pos_enc
        out = self.char_agg_transformer(padded, src_key_padding_mask=pad_mask)

        return out[:, 0], segment_info

    def _extract_word_segments_batch(
        self,
        features: torch.Tensor,
        word_alignment: torch.Tensor,
        char_alignment: torch.Tensor,
    ):
        B, T_words, T_speech = word_alignment.shape
        feat_dim = features.size(-1)

        segments, lengths, seg_info, dur_frames = [], [], [], []
        char_durations = char_alignment.sum(dim=2)

        for b in range(B):
            for w in range(T_words):
                frames = torch.where(word_alignment[b, w])[0]
                if frames.numel():
                    seg_feat = features[b, frames]
                    seg_durvec = torch.zeros(len(frames), device=features.device)
                    for j, f in enumerate(frames):
                        chars_here = torch.where(char_alignment[b, :, f])[0]
                        if chars_here.numel():
                            c = chars_here[0]
                            seg_durvec[j] = char_durations[b, c].float()
                    segments.append(seg_feat)
                    lengths.append(len(frames))
                    dur_frames.append(seg_durvec)
                else:
                    segments.append(torch.zeros(1, feat_dim, device=features.device))
                    lengths.append(1)
                    dur_frames.append(torch.zeros(1, device=features.device))
                seg_info.append((b, w))

        return segments, lengths, seg_info, dur_frames

    def _reconstruct_word_features_batch(
        self,
        aggregated_segments: torch.Tensor,
        segment_info: List[Tuple[int, int]],
        B: int,
        T_words: int,
        feat_dim: int,
        device: torch.device,
    ) -> torch.Tensor:
        word_features = torch.zeros(B, T_words, feat_dim, device=device)
        for vec, (b, w) in zip(aggregated_segments, segment_info):
            if b < B and w < T_words:
                word_features[b, w] = vec
        return word_features

    def _get_char_embeddings(self, gt_char_indices: torch.Tensor) -> torch.Tensor:
        valid_indices = torch.clamp(gt_char_indices, 0, self._CHAR_VOCAB_SIZE - 1)
        return self.char_embedding(valid_indices)

    def expand_word_features(
        self, seg_feat: torch.Tensor, word_alignment: torch.Tensor
    ) -> torch.Tensor:
        alignment_transposed = word_alignment.transpose(1, 2).float()
        return torch.bmm(alignment_transposed, seg_feat)

    def _word_causal_mask(self, frame_word_ids: torch.Tensor) -> torch.Tensor:
        B, T = frame_word_ids.shape
        ids_i = frame_word_ids.unsqueeze(2)
        ids_j = frame_word_ids.unsqueeze(1)
        allowed = ids_j <= ids_i
        mask = torch.where(allowed, 0.0, float("-inf"))
        mask = mask.unsqueeze(1).expand(-1, self.nhead, -1, -1)
        return mask.reshape(B * self.nhead, T, T)

    def expanded_forward(
        self,
        exp_feat: torch.Tensor,
        padding_mask: torch.Tensor,
        word_alignment: torch.Tensor,
    ):
        B, T, _ = exp_feat.shape
        frame_word_ids = word_alignment.float().argmax(dim=1)
        word_mask = self._word_causal_mask(frame_word_ids)

        transformer_input = self.unit_head_pre(exp_feat)
        pos_encoding = self.positional_encoding[:, :T, :]
        transformer_input = transformer_input + pos_encoding

        if padding_mask.dtype == torch.bool:
            padding_mask = torch.where(
                padding_mask, float("-inf"), 0.0
            ).to(transformer_input.dtype)

        transformer_output = self.transformer_encoder(
            transformer_input,
            mask=word_mask,
            src_key_padding_mask=padding_mask,
        )
        u_logits = self.unit_head_post(transformer_output)
        return u_logits.transpose(1, 2)

    def expanded_forward_streaming(
        self,
        new_exp_feat: torch.Tensor,
        new_padding_mask: torch.Tensor,
        new_word_alignment: torch.Tensor,
        word_offset: int,
        past_hidden: torch.Tensor = None,
        past_mask: torch.Tensor = None,
        past_word_ids: torch.Tensor = None,
    ):
        new_input = self.unit_head_pre(new_exp_feat)
        T_past = past_hidden.size(1) if past_hidden is not None else 0
        T_new = new_input.size(1)
        T_total = T_past + T_new

        pos_enc = self.positional_encoding[:, T_past:T_total, :]
        new_input = new_input + pos_enc

        new_word_ids = new_word_alignment.float().argmax(dim=1) + word_offset

        if new_padding_mask.dtype == torch.bool:
            new_padding_mask = torch.where(
                new_padding_mask, float("-inf"), 0.0
            ).to(new_input.dtype)

        if past_hidden is not None:
            full_input = torch.cat([past_hidden, new_input], dim=1)
            full_mask = torch.cat([past_mask, new_padding_mask], dim=1)
            full_word_ids = torch.cat([past_word_ids, new_word_ids], dim=1)
        else:
            full_input = new_input
            full_mask = new_padding_mask
            full_word_ids = new_word_ids

        word_mask = self._word_causal_mask(full_word_ids)

        full_output = self.transformer_encoder(
            full_input,
            mask=word_mask,
            src_key_padding_mask=full_mask,
        )

        new_output = full_output[:, T_past:, :]
        u_logits = self.unit_head_post(new_output).transpose(1, 2)
        new_word_offset = word_offset + new_word_alignment.size(1)

        return u_logits, full_input, full_mask, full_word_ids, new_word_offset