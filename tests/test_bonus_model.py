import inspect

import pytest
import torch
import torch.nn.functional as F

from bonus_checks import text_batch, with_extra_padding
from bonus_model import ARCHITECTURES, SinusoidalPositionalEncoding, SumAnswerModel, build_bonus_model, causal_mask, teacher_forced_loss
from config import ExperimentConfig
from tokenizer import BOS_ID, EOS_ID, OUT_PAD_ID, OUTPUT_VOCAB

CPU = torch.device("cpu")


def make(architecture="items_only_sinusoidal", seed=0, dropout=None):
    cfg = ExperimentConfig()
    cfg.bonus.architecture = architecture
    if dropout is not None:
        cfg.model.dropout = dropout
        cfg.bonus.decoder_dropout = dropout
    torch.manual_seed(seed)
    return build_bonus_model(cfg)


# ----------------------------------------------------------------------------- positions


def test_positional_encoding_shape_dtype_bounds_and_distinct_rows():
    pe = SinusoidalPositionalEncoding(128)
    table = pe(20)
    assert table.shape == (20, 128) and table.dtype == torch.float32
    assert torch.isfinite(table).all() and table.abs().max() <= 1.0
    distances = torch.cdist(pe(64), pe(64))  # well beyond 20: still one distinct vector per position
    assert (distances + torch.eye(64) * 10).min() > 1e-3
    assert pe(5, dtype=torch.bfloat16).dtype == torch.bfloat16


def test_positional_encoding_is_prefix_consistent():
    pe = SinusoidalPositionalEncoding(128)
    assert torch.equal(pe(20)[:10], pe(10))
    assert torch.equal(pe(4)[:3], pe(3))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_positional_encoding_on_mps_matches_cpu():
    pe = SinusoidalPositionalEncoding(128)
    on_mps = pe.to("mps")(20, device=torch.device("mps"))
    assert on_mps.device.type == "mps" and on_mps.dtype == torch.float32
    assert torch.allclose(on_mps.cpu(), SinusoidalPositionalEncoding(128)(20), atol=1e-5)


def test_odd_model_width_rejected():
    with pytest.raises(ValueError):
        SinusoidalPositionalEncoding(127)


# ----------------------------------------------------------------------------- masks


def test_causal_mask_blocks_only_future_positions():
    assert causal_mask(3, CPU).tolist() == [[False, True, True], [False, False, True], [False, False, False]]


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_decoder_is_causal(architecture):
    model = make(architecture).eval()
    inputs = text_batch([[3, 4], [9, 9, 9]], CPU)
    decoder_input = torch.tensor([[BOS_ID, 6, 4], [BOS_ID, 5, 10]])
    changed = decoder_input.clone()
    changed[:, 2] = 12  # alter only the LAST decoder input
    with torch.no_grad():
        a, b = model(**inputs, decoder_input=decoder_input), model(**inputs, decoder_input=changed)
    assert torch.allclose(a[:, :2], b[:, :2], atol=1e-6) and not torch.allclose(a[:, 2], b[:, 2])


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_operand_padding_with_garbage_changes_nothing(architecture):
    model = make(architecture).eval()
    inputs = text_batch([[3], [4, 0, 5], [9, 9, 1, 0, 2, 7, 3]], CPU)
    decoder_input = torch.tensor([[BOS_ID, 6]] * 3)
    with torch.no_grad():
        plain = model(**inputs, decoder_input=decoder_input)
        padded = model(**with_extra_padding(inputs, extra=6), decoder_input=decoder_input)
    assert torch.allclose(plain, padded, atol=1e-6)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_decoder_padding_positions_do_not_affect_valid_positions(architecture):
    model = make(architecture).eval()
    inputs = text_batch([[7]], CPU)
    short = torch.tensor([[BOS_ID, OUTPUT_VOCAB["7"]]])
    padded = torch.tensor([[BOS_ID, OUTPUT_VOCAB["7"], OUTPUT_VOCAB["9"]]])  # junk in a padded position
    with torch.no_grad():
        a = model(**inputs, decoder_input=short)
        b = model(**inputs, decoder_input=padded, decoder_padding_mask=torch.tensor([[False, False, True]]))
    assert torch.allclose(a, b[:, :2], atol=1e-6)


def test_loss_ignores_padded_targets_and_averages_over_valid_tokens():
    logits = torch.randn(2, 3, 13)
    targets = torch.tensor([[OUTPUT_VOCAB["9"], EOS_ID, OUT_PAD_ID], [OUTPUT_VOCAB["4"], OUTPUT_VOCAB["3"], EOS_ID]])
    loss = teacher_forced_loss(logits, targets)
    per_token = [F.cross_entropy(logits[b, t][None], targets[b, t][None]) for b in range(2) for t in range(3) if targets[b, t] != OUT_PAD_ID]
    assert loss.item() == pytest.approx(torch.stack(per_token).mean().item(), abs=1e-6)  # 5 valid tokens
    noisy = logits.clone()
    noisy[0, 2] = 100.0  # garbage at the padded position
    assert teacher_forced_loss(noisy, targets).item() == pytest.approx(loss.item(), abs=1e-6)


# ----------------------------------------------------------------------------- gradients + signature


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_gradients_are_finite_through_encoder_cnn_and_decoder(architecture):
    model = make(architecture)
    inputs = text_batch([[3, 1], [0, 5, 2]], CPU)
    inputs["is_image"][0, 1] = True  # one image operand so the CNN is used
    inputs["images"][0, 1] = torch.randn(1, 28, 28)
    inputs["token_ids"][0, 1] = 0
    decoder_input = torch.tensor([[BOS_ID, 7], [BOS_ID, 10]])
    targets = torch.tensor([[7, EOS_ID], [10, EOS_ID]])
    teacher_forced_loss(model(**inputs, decoder_input=decoder_input), targets).backward()
    for module in (model.operands.image_encoder, model.operands.layers, model.decoder_layers, model.vocab_head, model.answer_embedding):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads) and sum(g.abs().sum() for g in grads) > 0
    if model.summary_projection is not None:
        assert model.summary_projection.weight.grad.abs().sum() > 0
    assert (model.answer_embedding.weight.grad[OUT_PAD_ID] == 0).all()


def test_forward_takes_only_model_inputs():
    assert list(inspect.signature(SumAnswerModel.forward).parameters)[1:] == [
        "images", "token_ids", "is_image", "padding_mask", "decoder_input", "decoder_padding_mask"
    ]


def test_variant_parameter_counts_and_shared_initialization():
    counts = {a: sum(p.numel() for p in make(a).parameters()) for a in ARCHITECTURES}
    assert counts["items_only_sinusoidal"] == counts["items_only_no_positions"]
    assert counts["sinusoidal_plus_sum_summary"] - counts["items_only_sinusoidal"] == 128 * 128 + 128
    base, ablation = make("items_only_sinusoidal").state_dict(), make("sinusoidal_plus_sum_summary").state_dict()
    assert all(torch.equal(base[k], ablation[k]) for k in base)  # summary layer is built last
    for architecture in ("items_only_rope", "items_only_alibi"):
        relative = make(architecture).state_dict()
        assert counts[architecture] == counts["items_only_sinusoidal"]
        assert relative.keys() == base.keys()
        assert all(torch.equal(base[k], relative[k]) for k in base)


# ----------------------------------------------------------------------------- generation


class ScriptedDecode:
    """Stand-in for model.decode: row r emits scripts[r][step] at each step."""

    def __init__(self, scripts):
        self.scripts = scripts
        self.calls = 0

    def __call__(self, memory, memory_padding_mask, decoder_input, decoder_padding_mask=None):
        step = decoder_input.shape[1] - 1
        logits = torch.full((decoder_input.shape[0], decoder_input.shape[1], 13), -10.0)
        for row, script in enumerate(self.scripts):
            logits[row, -1, script[min(step, len(script) - 1)]] = 10.0
        self.calls += 1
        return logits


def test_generation_allows_three_digits_stops_at_eos_and_flags_missing_eos():
    from tokenizer import parse_generated

    model = make().eval()
    scripts = [[OUTPUT_VOCAB["1"], OUTPUT_VOCAB["0"], OUTPUT_VOCAB["0"], EOS_ID], [OUTPUT_VOCAB["7"], EOS_ID], [OUTPUT_VOCAB["5"]] * 4]
    model.decode = ScriptedDecode(scripts)
    generated = model.generate(**text_batch([[1], [2], [3]], CPU), max_new_tokens=4)
    assert generated.shape == (3, 4) and model.decode.calls == 4
    assert [parse_generated(row) for row in generated.tolist()] == [("100", "valid"), ("7", "valid"), (None, "missing_eos")]
    assert generated[1, 2:].tolist() == [OUT_PAD_ID, OUT_PAD_ID]  # finished row padded after its EOS


def test_generation_stops_early_when_every_row_finished():
    model = make().eval()
    model.decode = ScriptedDecode([[OUTPUT_VOCAB["7"], EOS_ID], [EOS_ID]])
    model.generate(**text_batch([[1], [2]], CPU), max_new_tokens=4)
    assert model.decode.calls == 2


# ----------------------------------------------------------------------------- repeated operands + summary


@torch.no_grad()
def next_token(model, rows, prefix=(BOS_ID,)):
    inputs = text_batch(rows, CPU)
    return model(**inputs, decoder_input=torch.tensor([list(prefix)] * len(rows)))[:, -1].softmax(-1)


@pytest.mark.parametrize("architecture", ("items_only_no_positions", "items_only_rope", "items_only_alibi"))
def test_position_free_model_is_count_blind_for_identical_operands(architecture):
    """Softmax attention averages identical items, so [3], [3,3], ..., [3]*20 are indistinguishable."""
    model = make(architecture).eval()
    reference = next_token(model, [[3]])
    for k in (2, 3, 10, 20):
        assert torch.allclose(next_token(model, [[3] * k]), reference, atol=1e-5)
        assert torch.allclose(next_token(model, [[3] * k], prefix=(BOS_ID, OUTPUT_VOCAB["3"])), next_token(model, [[3]], prefix=(BOS_ID, OUTPUT_VOCAB["3"])), atol=1e-5)


def test_sinusoidal_model_represents_identical_operands_differently_by_position():
    model = make("items_only_sinusoidal").eval()
    with torch.no_grad():
        memory = model.encode(**text_batch([[3] * 5], CPU))[0]  # [5, D]
    assert (memory[1:] - memory[0]).abs().amax(dim=1).min() > 1e-3
    # Positions make outputs differ across k. Nothing is asserted about correctness: untrained.
    assert not torch.allclose(next_token(model, [[3]]), next_token(model, [[3, 3]]), atol=1e-4)


def test_sum_summary_is_count_sensitive_and_ignores_padding():
    model = make("sinusoidal_plus_sum_summary")
    memory = torch.randn(1, 3, 128)
    pad = torch.zeros(1, 3, dtype=torch.bool)
    single = model.sum_summary(memory, pad)
    doubled = model.sum_summary(torch.cat([memory, memory], dim=1), torch.zeros(1, 6, dtype=torch.bool))
    assert torch.allclose(doubled, 2 * single, atol=1e-5)  # a sum, not an average: copies add up
    garbage = torch.cat([memory, torch.randn(1, 2, 128) * 100], dim=1)
    assert torch.allclose(model.sum_summary(garbage, torch.tensor([[False, False, False, True, True]])), single, atol=1e-5)
    assert torch.allclose(single, memory.sum(dim=1) * model.summary_scale)
