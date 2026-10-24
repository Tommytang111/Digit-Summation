"""Independent attention references and integration checks for RoPE / ALiBi."""

import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from bonus_checks import text_batch, with_extra_padding
from bonus_model import (
    RelativeSelfAttention, RotaryPositionEncoding, alibi_slopes, build_bonus_model,
    causal_mask, relative_decoder_layer, relative_encoder_layer, teacher_forced_loss,
)
from config import ConfigError, ExperimentConfig, load_config
from tokenizer import BOS_ID, EOS_ID


def test_rope_preserves_norms_relative_offsets_and_prefixes():
    torch.manual_seed(42)
    rope = RotaryPositionEncoding(8)
    q, k = torch.randn(2, 3, 24, 8), torch.randn(2, 3, 24, 8)
    rotated = rope(q)
    assert torch.allclose(rotated.square().sum(-1), q.square().sum(-1), atol=2e-6)
    assert torch.equal(rope(q[:, :, :10]), rotated[:, :, :10])
    assert torch.equal(rotated[:, :, 0], q[:, :, 0])
    # Moving all positions by the same offset preserves Q.K, even beyond training lengths.
    dots = rotated @ rope(k).transpose(-1, -2)
    shifted = rope(q, offset=37) @ rope(k, offset=37).transpose(-1, -2)
    assert torch.allclose(dots, shifted, atol=1e-5)
    assert rope(q.bfloat16()).dtype == torch.bfloat16
    with pytest.raises(ValueError, match="even head"):
        RotaryPositionEncoding(3)


def test_alibi_slopes_match_paper_schedule_and_non_power_of_two_extension():
    assert torch.equal(alibi_slopes(4), torch.tensor([1 / 4, 1 / 16, 1 / 64, 1 / 256]))
    assert torch.equal(alibi_slopes(3), torch.tensor([1 / 16, 1 / 256, 1 / 4]))
    with pytest.raises(ValueError):
        alibi_slopes(0)


@pytest.mark.parametrize("scheme", ("rope", "alibi"))
@pytest.mark.parametrize("causal", (False, True))
def test_attention_matches_independent_matrix_reference(scheme, causal):
    """Check Q/K-only rotations, ALiBi sign/scale, key padding and causal masking together."""
    torch.manual_seed(7)
    hidden = torch.randn(2, 5, 8)
    pad = torch.tensor([[False, False, False, True, True], [False] * 5])
    mha = nn.MultiheadAttention(8, 2, batch_first=True, dropout=0.0).eval()
    attention = RelativeSelfAttention(scheme, 4, 2).eval()
    actual = attention(mha, hidden, pad, causal=causal)

    projections = [F.linear(hidden, w, b).view(2, 5, 2, 4).transpose(1, 2)
                   for w, b in zip(mha.in_proj_weight.chunk(3), mha.in_proj_bias.chunk(3))]
    q, k, v = projections
    if scheme == "rope":
        # Construct explicit 4x4 rotation matrices, independently of the vectorized helper.
        matrices = []
        for position in range(5):
            blocks = []
            for frequency in (1.0, 0.01):
                angle = position * frequency
                c, s = math.cos(angle), math.sin(angle)
                blocks.append(torch.tensor([[c, -s], [s, c]]))
            matrices.append(torch.block_diag(*blocks))
        rotation = torch.stack(matrices)
        q = torch.einsum("bhld,led->bhle", q, rotation)
        k = torch.einsum("bhld,led->bhle", k, rotation)
    scores = (q @ k.transpose(-1, -2)) / 2.0  # sqrt(head_dim) = 2
    if scheme == "alibi":
        for h, slope in enumerate((1 / 16, 1 / 256)):
            for i in range(5):
                for j in range(5):
                    scores[:, h, i, j] -= slope * abs(i - j)
    scores = scores.masked_fill(pad[:, None, None, :], -torch.inf)
    if causal:
        scores = scores.masked_fill(torch.triu(torch.ones(5, 5, dtype=torch.bool), 1), -torch.inf)
    expected = (scores.softmax(-1) @ v).transpose(1, 2).reshape(2, 5, 8)
    expected = F.linear(expected, mha.out_proj.weight, mha.out_proj.bias)
    assert torch.allclose(actual, expected, atol=1e-6)


@pytest.mark.parametrize("scheme", ("rope", "alibi"))
@pytest.mark.parametrize("norm_first", (False, True))
def test_relative_layer_blocks_match_pytorch_when_position_effect_is_zero(scheme, norm_first):
    """Ensure residuals, normalization, feedforward and cross-attention are unchanged."""
    torch.manual_seed(9)
    attention = RelativeSelfAttention(scheme, 4, 2).eval()
    if scheme == "rope":
        attention.rotary.inverse_frequency.zero_()
    else:
        attention.slopes.zero_()
    encoder = nn.TransformerEncoderLayer(8, 2, 16, dropout=0, batch_first=True, norm_first=norm_first).eval()
    decoder = nn.TransformerDecoderLayer(8, 2, 16, dropout=0, batch_first=True, norm_first=True).eval()
    memory, hidden = torch.randn(2, 5, 8), torch.randn(2, 3, 8)
    memory_pad = torch.tensor([[False, False, False, True, True], [False] * 5])
    answer_pad = torch.tensor([[False, False, True], [False] * 3])
    with torch.no_grad():
        expected = encoder(memory, src_key_padding_mask=memory_pad)
        actual = relative_encoder_layer(encoder, memory, memory_pad, attention)
        assert torch.allclose(actual, expected, atol=1e-6)
        expected = decoder(hidden, memory, tgt_mask=causal_mask(3, hidden.device),
                           tgt_key_padding_mask=answer_pad, memory_key_padding_mask=memory_pad)
        actual = relative_decoder_layer(decoder, hidden, memory, answer_pad, memory_pad, attention)
        assert torch.allclose(actual, expected, atol=1e-6)


@pytest.mark.parametrize("architecture", ("items_only_rope", "items_only_alibi"))
@pytest.mark.parametrize("device", ["cpu", pytest.param("mps", marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available"))])
def test_long_inputs_training_generation_and_padding(architecture, device):
    cfg = ExperimentConfig()
    cfg.model.d_model, cfg.model.nhead, cfg.model.num_layers, cfg.model.dim_feedforward = 24, 3, 1, 48
    cfg.bonus.decoder_layers, cfg.bonus.decoder_heads, cfg.bonus.decoder_feedforward = 1, 2, 48
    cfg.bonus.architecture = architecture
    torch.manual_seed(1)
    model = build_bonus_model(cfg).to(device)
    inputs = text_batch([[1], [2, 3] * 5, [4, 0, 5, 1] * 5, [i % 10 for i in range(64)]], device)
    prefix = torch.tensor([[BOS_ID, 7, 6]] * 4, device=device)
    targets = torch.tensor([[7, 6, EOS_ID]] * 4, device=device)
    loss = teacher_forced_loss(model(**inputs, decoder_input=prefix), targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.eval()
    with torch.no_grad():
        memory = model.encode(**inputs)
        full = model.decode(memory, inputs["padding_mask"], prefix)
        # Teacher forcing and generation use exactly the same positions for each prefix.
        for t in range(1, 4):
            partial = model.decode(memory, inputs["padding_mask"], prefix[:, :t])
            assert torch.allclose(full[:, t - 1], partial[:, -1], atol=2e-5)
        padded = model(**with_extra_padding(inputs, 5), decoder_input=prefix)
        assert torch.allclose(full, padded, atol=2e-5)
        generated = model.generate(**inputs)
        assert generated.shape[0] == 4 and 1 <= generated.shape[1] <= 4
        assert torch.equal(generated, model.generate(**inputs))  # no attention dropout in eval


@pytest.mark.parametrize("architecture", ("items_only_rope", "items_only_alibi"))
def test_relative_variants_cpu_bfloat16_autocast(architecture):
    cfg = ExperimentConfig()
    cfg.bonus.architecture = architecture
    model = build_bonus_model(cfg)
    inputs = text_batch([[1], [2, 3, 4]], "cpu")
    prefix = torch.tensor([[BOS_ID, 7]] * 2)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = model(**inputs, decoder_input=prefix)
        loss = teacher_forced_loss(logits, torch.tensor([[7, EOS_ID]] * 2))
    loss.backward()
    assert logits.dtype == torch.bfloat16 and torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize("architecture", ("items_only_rope", "items_only_alibi"))
def test_relative_variants_load_through_existing_configs(architecture):
    for path in ("configs/bonus_debug.yaml", "configs/bonus_local.yaml", "configs/bonus_h100.yaml"):
        cfg = load_config(path, [f"bonus.architecture={architecture}"])
        assert cfg.bonus.architecture == architecture


@pytest.mark.parametrize("overrides", [
    ["model.d_model=12", "model.nhead=4", "bonus.decoder_heads=2"],
    ["model.d_model=12", "model.nhead=2", "bonus.decoder_heads=4"],
])
def test_rope_rejects_odd_encoder_or_decoder_head_dimension(overrides):
    with pytest.raises(ConfigError, match="even encoder and decoder head"):
        load_config("configs/bonus_debug.yaml", ["bonus.architecture=items_only_rope", *overrides])
