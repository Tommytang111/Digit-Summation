"""Length-generalization bonus: operand encoder + small autoregressive answer decoder.

The main-task classifier sum-pools the encoder outputs into a 91-way head. Here the pooling and
head are removed. A Transformer decoder instead writes the sum one character at a time, reading
the whole sequence of encoder outputs through cross-attention:

    images [B,L,1,28,28], token_ids [B,L], is_image [B,L], padding_mask [B,L] (True = padding)
      -> OperandEncoder.embed_operands        -> [B,L,D]   one vector per operand (CNN or token embedding)
      -> + operand positions 0..L-1 (variants with encoder positions; across the whole mixed sequence)
      -> OperandEncoder.encode (bidirectional) -> memory [B,L,D]

    decoder_input [B,T] (e.g. [BOS, "4", "3"]) -> answer embedding + positions 0..T-1 (0 = BOS)
      -> N x TransformerDecoderLayer:
           causal self-attention   (tgt_mask: position t sees positions <= t)
           cross-attention to memory (memory_key_padding_mask = padding_mask)
      -> LayerNorm [-> + projected sum summary: ablation only] -> Linear -> logits [B,T,13]

Masks (all Boolean, True = "not allowed / padding", the PyTorch convention):
    src_key_padding_mask    [B,L]  encoder self-attention ignores padded operands
    memory_key_padding_mask [B,L]  decoder cross-attention ignores padded operands
    tgt_mask                [T,T]  causal: True above the diagonal (no peeking at later characters)
    tgt_key_padding_mask    [B,T]  decoder self-attention ignores padded answer positions
Padded *target* positions are excluded from the loss by ignore_index=OUT_PAD_ID.

Variants (BonusConfig.architecture):
    items_only_sinusoidal        primary: sinusoidal operand positions + decoder reads encoder items only
    items_only_no_positions      position-free operands (the main-task encoder) + decoder reads items only
    sinusoidal_plus_sum_summary  primary + a masked sum of encoder items added before the vocabulary head
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from model import OperandEncoder, operand_encoder_kwargs
from tokenizer import BOS_ID, EOS_ID, OUT_PAD_ID, OUTPUT_VOCAB_SIZE

ARCHITECTURES = ("items_only_sinusoidal", "items_only_no_positions", "sinusoidal_plus_sum_summary")
DECODER_INPUT_KEYS = ("decoder_input", "decoder_padding_mask")


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sin/cos position vectors (Vaswani et al., 2017), computed for any length on demand.

        PE[pos, 2i]   = sin(pos / 10000^(2i / D))
        PE[pos, 2i+1] = cos(pos / 10000^(2i / D))

    Each dimension pair is a sinusoid with its own wavelength, from 2*pi up to 10000 * 2*pi
    positions. Every position gets a distinct, bounded ([-1, 1]) vector, including positions never
    seen in training, because nothing is looked up from a learned table. That makes positions
    *distinguishable* beyond the training range. It does not make the model count, add, or
    generalize; that still has to be learned.

    Computed in float32 from a non-persistent buffer (not saved in checkpoints), then cast to the
    caller's device and dtype, so MPS never sees float64. Values for positions 0..n-1 do not
    depend on how many positions are requested (prefix-consistent).
    """

    def __init__(self, d_model: int, base: float = 10_000.0):
        super().__init__()
        if d_model % 2:
            raise ValueError("Sinusoidal positions need an even d_model")
        inverse_frequency = 1.0 / (base ** (torch.arange(0, d_model, 2, dtype=torch.float32) / d_model))  # [D/2]
        self.register_buffer("inverse_frequency", inverse_frequency, persistent=False)

    def forward(self, length: int, device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Position vectors for positions 0..length-1: [length, D]."""
        positions = torch.arange(length, device=self.inverse_frequency.device, dtype=torch.float32)  # [length]
        angles = positions[:, None] * self.inverse_frequency[None, :]  # [length, D/2]
        encoding = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(start_dim=1)  # [length, D], sin/cos interleaved
        return encoding.to(device=device if device is not None else encoding.device, dtype=dtype)


def causal_mask(length: int, device: torch.device) -> torch.Tensor:
    """[T, T] Boolean mask, True above the diagonal: query position t may not attend to keys > t."""
    return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)


class SumAnswerModel(nn.Module):
    """Operand encoder + autoregressive decoder that writes the sum as digit characters."""

    def __init__(
        self,
        encoder_kwargs: dict,
        architecture: str = "items_only_sinusoidal",
        decoder_layers: int = 2,
        decoder_heads: int = 4,
        decoder_feedforward: int = 512,
        decoder_dropout: float = 0.1,
        summary_scale: float = 0.1,
    ):
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {ARCHITECTURES}, got {architecture!r}")
        d_model = encoder_kwargs["d_model"]
        self.architecture = architecture
        self.use_operand_positions = architecture != "items_only_no_positions"
        self.summary_scale = summary_scale

        self.operands = OperandEncoder(**encoder_kwargs)  # CNN + token/modality embeddings + bidirectional encoder
        # One parameter-free module serves two *separate* position sequences: operands (0..L-1)
        # and answer characters (0 = BOS, 1, 2, ...).
        self.positions = SinusoidalPositionalEncoding(d_model)
        self.answer_embedding = nn.Embedding(OUTPUT_VOCAB_SIZE, d_model, padding_idx=OUT_PAD_ID)
        # Independently constructed layers (each gets its own random init), as in the encoder.
        self.decoder_layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=decoder_heads,
                    dim_feedforward=decoder_feedforward,
                    dropout=decoder_dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(decoder_layers)
            ]
        )
        self.decoder_norm = nn.LayerNorm(d_model)
        self.vocab_head = nn.Linear(d_model, OUTPUT_VOCAB_SIZE)
        # Ablation only. It is built last, so under the same seed every shared parameter is
        # initialized exactly as in the other variants.
        self.summary_projection = nn.Linear(d_model, d_model) if architecture == "sinusoidal_plus_sum_summary" else None

    # ------------------------------------------------------------------ encoder side

    def encode(self, images, token_ids, is_image, padding_mask) -> torch.Tensor:
        """Operands -> memory [B, L, D].

        With operand positions, position p is added to the vector at slot p of the *mixed*
        sequence, whether that slot holds an image or text. Batches are right-padded, so real
        operands always sit at positions 0..len-1, and padded slots (which also get a position
        vector) stay masked. The encoder is bidirectional: there is no causal mask here.
        """
        features = self.operands.embed_operands(images, token_ids, is_image, padding_mask)  # [B, L, D]
        if self.use_operand_positions:
            features = features + self.positions(features.shape[1], features.device, features.dtype)[None]  # [1, L, D]
        return self.operands.encode(features, padding_mask)  # [B, L, D]

    def sum_summary(self, memory: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """Ablation: masked SUM of encoder outputs [B, L, D] -> [B, D], times a fixed scale.

        Unlike attention (a weighted *average*), a sum grows with the number of operands, so it
        carries count and magnitude information. It is never normalized afterward, which would
        erase that signal. The scale is the same constant at every length, for numerical range
        only. It sums learned features, not digit labels or the target, so this is an additive
        architectural bias, not a programmed arithmetic solver.
        """
        return memory.masked_fill(padding_mask.unsqueeze(-1), 0.0).sum(dim=1) * self.summary_scale

    # ------------------------------------------------------------------ decoder side

    def decode(self, memory, memory_padding_mask, decoder_input, decoder_padding_mask=None) -> torch.Tensor:
        """Decoder over a (prefix of an) answer: decoder_input [B, T] -> logits [B, T, 13].

        Logits at position t predict the token at position t+1 of [BOS, d1, d2, ..., EOS]. The same
        positions (0 = BOS) are used in teacher forcing and in free generation.
        """
        length = decoder_input.shape[1]
        hidden = self.answer_embedding(decoder_input)  # [B, T, D]
        hidden = hidden + self.positions(length, hidden.device, hidden.dtype)[None]  # [1, T, D]
        if decoder_padding_mask is None:
            decoder_padding_mask = torch.zeros_like(decoder_input, dtype=torch.bool)
        future = causal_mask(length, hidden.device)  # [T, T]
        for layer in self.decoder_layers:
            hidden = layer(
                hidden,
                memory,
                tgt_mask=future,
                tgt_key_padding_mask=decoder_padding_mask,
                memory_key_padding_mask=memory_padding_mask,
                tgt_is_causal=True,
            )
        hidden = self.decoder_norm(hidden)  # [B, T, D]
        if self.summary_projection is not None:
            hidden = hidden + self.summary_projection(self.sum_summary(memory, memory_padding_mask))[:, None, :]
        return self.vocab_head(hidden)  # [B, T, 13]

    def forward(self, images, token_ids, is_image, padding_mask, decoder_input, decoder_padding_mask=None) -> torch.Tensor:
        """Teacher-forced logits [B, T, 13]. Takes only model inputs (no targets, labels, or sums)."""
        memory = self.encode(images, token_ids, is_image, padding_mask)
        return self.decode(memory, padding_mask, decoder_input, decoder_padding_mask)

    @torch.no_grad()
    def generate(self, images, token_ids, is_image, padding_mask, max_new_tokens: int = 4) -> torch.Tensor:
        """Greedy free generation from BOS, using only the model's own previous predictions.

        Returns the generated tokens after BOS: [B, <= max_new_tokens].
        - The encoder runs once. Each step re-decodes the whole prefix, which is cheap at
          T <= 5 and gives exactly the teacher-forcing positions.
        - Completion is tracked per row. A finished row gets PAD after its EOS, which
          tokenizer.parse_generated ignores, and the loop ends when every row has emitted EOS
          or the cap is reached.
        - No special-token suppression: PAD/BOS can be emitted and then count as invalid output.
        - A row without EOS within the cap is a generation failure ("missing_eos").
        """
        memory = self.encode(images, token_ids, is_image, padding_mask)
        return self.generate_from_memory(memory, padding_mask, max_new_tokens)

    @torch.no_grad()
    def generate_from_memory(self, memory: torch.Tensor, padding_mask: torch.Tensor, max_new_tokens: int = 4) -> torch.Tensor:
        """generate() for an already-computed encoder memory [B, L, D] (see generate)."""
        batch = memory.shape[0]
        sequence = torch.full((batch, 1), BOS_ID, dtype=torch.long, device=memory.device)  # [B, 1]
        finished = torch.zeros(batch, dtype=torch.bool, device=memory.device)
        for _ in range(max_new_tokens):
            logits = self.decode(memory, padding_mask, sequence)  # [B, t, 13]
            next_token = logits[:, -1].argmax(dim=-1).masked_fill(finished, OUT_PAD_ID)  # [B]
            sequence = torch.cat([sequence, next_token[:, None]], dim=1)
            finished = finished | (next_token == EOS_ID)
            if bool(finished.all()):
                break
        return sequence[:, 1:]


def teacher_forced_loss(logits: torch.Tensor, decoder_target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy averaged over valid (non-pad) target tokens in the batch.

    A 2-digit answer contributes 3 tokens ("4", "3", EOS) and a 1-digit answer 2, so examples with
    longer answers weigh slightly more. Padded target positions (OUT_PAD_ID) are ignored.
    """
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), decoder_target.reshape(-1), ignore_index=OUT_PAD_ID)


def build_bonus_model(cfg) -> SumAnswerModel:
    b = cfg.bonus
    return SumAnswerModel(
        operand_encoder_kwargs(cfg.model),
        architecture=b.architecture,
        decoder_layers=b.decoder_layers,
        decoder_heads=b.decoder_heads,
        decoder_feedforward=b.decoder_feedforward,
        decoder_dropout=b.decoder_dropout,
        summary_scale=b.summary_scale,
    )


def bonus_model_inputs(batch: dict, device: torch.device, with_decoder: bool = True) -> dict[str, torch.Tensor]:
    """Model inputs only (operands + optionally the teacher-forcing decoder input), moved to `device`."""
    keys = ("images", "token_ids", "is_image", "padding_mask") + (DECODER_INPUT_KEYS if with_decoder else ())
    return {key: batch[key].to(device, non_blocking=True) for key in keys}
