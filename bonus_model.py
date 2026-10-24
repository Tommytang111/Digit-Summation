"""Length-generalization bonus: operand encoder + small autoregressive answer decoder.

The main-task classifier sum-pools the encoder outputs into a 91-way head. Here the pooling and
head are removed. A Transformer decoder instead writes the sum one character at a time, reading
the whole sequence of encoder outputs through cross-attention:

    images [B,L,1,28,28], token_ids [B,L], is_image [B,L], padding_mask [B,L] (True = padding)
      -> OperandEncoder.embed_operands        -> [B,L,D]   one vector per operand (CNN or token embedding)
      -> + sinusoidal operand positions 0..L-1 (additive-position variants only)
      -> OperandEncoder.encode (bidirectional) -> memory [B,L,D]

    decoder_input [B,T] (e.g. [BOS, "4", "3"]) -> answer embedding + positions 0..T-1 (0 = BOS)
      -> N x TransformerDecoderLayer:
           causal self-attention   (tgt_mask: position t sees positions <= t)
           cross-attention to memory (memory_key_padding_mask = padding_mask)
      -> LayerNorm [-> + projected sum summary: ablation only] -> Linear -> logits [B,T,13]

RoPE / ALiBi variants instead put positions inside encoder and decoder self-attention, without
adding sinusoidal vectors. Cross-attention has no positional transform: operands and answer
characters have separate coordinate systems. All variants use the same teacher-forced loss.

Public masks (Boolean, True = "not allowed / padding", the MultiheadAttention convention):
    src_key_padding_mask    [B,L]  encoder self-attention ignores padded operands
    memory_key_padding_mask [B,L]  decoder cross-attention ignores padded operands
    tgt_mask                [T,T]  causal: True above the diagonal (no peeking at later characters)
    tgt_key_padding_mask    [B,T]  decoder self-attention ignores padded answer positions
Padded *target* positions are excluded from the loss by ignore_index=OUT_PAD_ID.

Variants (BonusConfig.architecture):
    items_only_sinusoidal        primary: sinusoidal operand positions + decoder reads encoder items only
    items_only_no_positions      position-free operands (the main-task encoder) + decoder reads items only
    sinusoidal_plus_sum_summary  primary + a masked sum of encoder items added before the vocabulary head
    items_only_rope              rotary Q/K in encoder and decoder self-attention; values unchanged
    items_only_alibi             symmetric distance bias in encoder, causal distance bias in decoder
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from model import OperandEncoder, operand_encoder_kwargs
from tokenizer import BOS_ID, EOS_ID, OUT_PAD_ID, OUTPUT_VOCAB_SIZE

ARCHITECTURES = (
    "items_only_sinusoidal", "items_only_no_positions", "sinusoidal_plus_sum_summary",
    "items_only_rope", "items_only_alibi",
)
DECODER_INPUT_KEYS = ("decoder_input", "decoder_padding_mask")


def position_schemes(architecture: str) -> dict[str, str]:
    """Position metadata shared by the model, checkpoints, and diagnostic scripts."""
    if architecture not in ARCHITECTURES:
        raise ValueError(f"architecture must be one of {ARCHITECTURES}, got {architecture!r}")
    if architecture in ("items_only_rope", "items_only_alibi"):
        scheme = architecture.removeprefix("items_only_")
        return {"operands": scheme, "answer": scheme}
    return {"operands": "none" if architecture == "items_only_no_positions" else "sinusoidal", "answer": "sinusoidal"}


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


class RotaryPositionEncoding(nn.Module):
    """RoPE on adjacent feature pairs of Q/K [B,H,L,head_dim], with base 10,000.

    Frequencies use head_dim, not the model width. Positions are computed in float32 at any
    requested length, then cast to the projection dtype. V is never rotated. No learned table.
    Reference: https://arxiv.org/abs/2104.09864
    """

    def __init__(self, head_dim: int):
        super().__init__()
        if head_dim < 2 or head_dim % 2:
            raise ValueError("RoPE needs an even head dimension >= 2")
        frequency = 10_000.0 ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        self.register_buffer("inverse_frequency", frequency, persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        positions = torch.arange(offset, offset + x.shape[-2], device=x.device, dtype=torch.float32)
        angles = positions[:, None] * self.inverse_frequency[None]
        cosine, sine = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1).flatten(-2)


def alibi_slopes(num_heads: int) -> torch.Tensor:
    """Fixed ALiBi slopes, including the authors' extension to non-power-of-two head counts.

    Reference: https://github.com/ofirpress/attention_with_linear_biases/issues/5
    """
    if num_heads < 1:
        raise ValueError("ALiBi needs at least one attention head")

    def power_of_two(count):
        start = 2 ** (-2 ** -(math.log2(count) - 3))
        return [start ** (i + 1) for i in range(count)]

    closest = 2 ** int(math.log2(num_heads))
    values = power_of_two(closest)
    if closest != num_heads:
        values += power_of_two(2 * closest)[0::2][:num_heads - closest]
    return torch.tensor(values, dtype=torch.float32)


class RelativeSelfAttention(nn.Module):
    """Parameter-free positional attention using an existing layer's QKV/output projections.

    Keeps the same parameter names, initialization, and count as nn.MultiheadAttention. Uses
    explicit SDPA so a TransformerEncoderLayer inference fast path cannot bypass RoPE/ALiBi.
    ALiBi adds -slope[h] * abs(query_position - key_position) AFTER QK scaling; decoder future
    keys are additionally masked. Padding always means True = excluded at this interface.
    """

    def __init__(self, scheme: str, head_dim: int, num_heads: int):
        super().__init__()
        if scheme not in ("rope", "alibi"):
            raise ValueError(f"Unknown relative position scheme: {scheme!r}")
        self.rotary = RotaryPositionEncoding(head_dim) if scheme == "rope" else None
        self.register_buffer("slopes", alibi_slopes(num_heads) if scheme == "alibi" else None, persistent=False)

    def forward(self, attention: nn.MultiheadAttention, hidden: torch.Tensor,
                padding_mask: torch.Tensor, causal: bool = False) -> torch.Tensor:
        batch, length, width = hidden.shape
        heads, head_dim = attention.num_heads, attention.head_dim
        projected = F.linear(hidden, attention.in_proj_weight, attention.in_proj_bias)
        q, k, v = projected.reshape(batch, length, 3, heads, head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        if self.rotary is not None:
            q, k = self.rotary(q), self.rotary(k)
        # SDPA's BOOLEAN mask has the opposite meaning to the public padding mask. Float
        # 0 / -inf is unambiguous and combines with ALiBi without losing its distance bias.
        bias = torch.zeros(1, 1, length, length, device=q.device, dtype=q.dtype)
        if self.slopes is not None:
            positions = torch.arange(length, device=q.device, dtype=torch.float32)
            distance = (positions[:, None] - positions[None, :]).abs()
            bias = (-self.slopes[:, None, None] * distance).to(q.dtype)[None]
        bias = bias.masked_fill(padding_mask[:, None, None, :], float("-inf"))
        if causal:
            bias = bias.masked_fill(causal_mask(length, q.device), float("-inf"))
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, dropout_p=attention.dropout if self.training else 0.0,
        )
        combined = attended.transpose(1, 2).reshape(batch, length, width)
        return F.linear(combined, attention.out_proj.weight, attention.out_proj.bias)


def relative_encoder_layer(layer, hidden, padding_mask, attention):
    """The usual encoder residual/FFN blocks, replacing only self-attention; supports pre/post-LN."""
    def self_attention(x):
        return layer.dropout1(attention(layer.self_attn, x, padding_mask))

    def feedforward(x):
        return layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(x)))))

    if layer.norm_first:
        hidden = hidden + self_attention(layer.norm1(hidden))
        return hidden + feedforward(layer.norm2(hidden))
    hidden = layer.norm1(hidden + self_attention(hidden))
    return layer.norm2(hidden + feedforward(hidden))


def relative_decoder_layer(layer, hidden, memory, padding_mask, memory_padding_mask, attention):
    """Pre-LN decoder: relative causal self-attention, ordinary cross-attention, then FFN."""
    hidden = hidden + layer.dropout1(attention(layer.self_attn, layer.norm1(hidden), padding_mask, causal=True))
    query = layer.norm2(hidden)
    cross = layer.multihead_attn(query, memory, memory, key_padding_mask=memory_padding_mask, need_weights=False)[0]
    hidden = hidden + layer.dropout2(cross)
    return hidden + layer.dropout3(layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm3(hidden))))))


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
        schemes = position_schemes(architecture)
        d_model = encoder_kwargs["d_model"]
        self.architecture = architecture
        self.use_operand_positions = schemes["operands"] == "sinusoidal"  # additive vectors only
        self.summary_scale = summary_scale

        self.operands = OperandEncoder(**encoder_kwargs)  # CNN + token/modality embeddings + bidirectional encoder
        # The original variants share an additive encoding for two separate position sequences.
        # Relative variants have no additive encoding on either side.
        self.positions = SinusoidalPositionalEncoding(d_model) if schemes["answer"] == "sinusoidal" else None
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
        # Added after all learned modules; no random draws or persistent buffers, so old
        # checkpoint keys and seeded parameter initialization remain exactly the same.
        self.encoder_attention = self.decoder_attention = None
        if schemes["operands"] in ("rope", "alibi"):
            heads = self.operands.layers[0].self_attn.num_heads
            self.encoder_attention = RelativeSelfAttention(schemes["operands"], d_model // heads, heads)
            self.decoder_attention = RelativeSelfAttention(schemes["answer"], d_model // decoder_heads, decoder_heads)

    # ------------------------------------------------------------------ encoder side

    def encode(self, images, token_ids, is_image, padding_mask) -> torch.Tensor:
        """Operands -> memory [B, L, D].

        With operand positions, position p is added to the vector at slot p of the *mixed*
        sequence, whether that slot holds an image or text. Batches are right-padded, so real
        operands always sit at positions 0..len-1, and padded slots (which also get a position
        vector) stay masked. The encoder is bidirectional: there is no causal mask here.
        Relative variants instead transform self-attention Q/K or scores at each layer.
        """
        features = self.operands.embed_operands(images, token_ids, is_image, padding_mask)  # [B, L, D]
        if self.use_operand_positions:
            features = features + self.positions(features.shape[1], features.device, features.dtype)[None]  # [1, L, D]
        if self.encoder_attention is not None:
            for layer in self.operands.layers:
                features = relative_encoder_layer(layer, features, padding_mask, self.encoder_attention)
            return self.operands.final_norm(features)
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
        Relative variants replace additive answer positions with causal RoPE/ALiBi attention.
        """
        length = decoder_input.shape[1]
        hidden = self.answer_embedding(decoder_input)  # [B, T, D]
        if self.decoder_attention is None:
            hidden = hidden + self.positions(length, hidden.device, hidden.dtype)[None]  # [1, T, D]
        if decoder_padding_mask is None:
            decoder_padding_mask = torch.zeros_like(decoder_input, dtype=torch.bool)
        future = causal_mask(length, hidden.device)  # [T, T]
        for layer in self.decoder_layers:
            if self.decoder_attention is not None:
                hidden = relative_decoder_layer(
                    layer, hidden, memory, decoder_padding_mask, memory_padding_mask, self.decoder_attention,
                )
                continue
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
