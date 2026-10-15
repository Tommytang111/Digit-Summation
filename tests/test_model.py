import inspect
import math

import pytest
import torch
import torch.nn.functional as F

from config import ExperimentConfig
from model import build_model
from sanity_checks import synthetic_batch
from tokenizer import PAD_ID

INPUTS = ("images", "token_ids", "is_image", "padding_mask")


def make_model(seed=0, randomize_head=False, **model_overrides):
    torch.manual_seed(seed)
    cfg = ExperimentConfig()
    for key, value in model_overrides.items():
        setattr(cfg.model, key, value)
    model = build_model(cfg)
    if randomize_head:
        # The head is zero-initialized, which blocks backbone gradients on the very first
        # backward pass. Gradient-flow tests randomize it so one backward pass reaches everything.
        torch.nn.init.normal_(model.head.weight, std=0.02)
    return model


def inputs_of(batch):
    return {k: batch[k] for k in INPUTS}


@pytest.mark.parametrize("kind", ["mixed", "all_text", "all_image"])
def test_output_shape_and_finiteness(kind):
    model = make_model().eval()
    logits = model(**inputs_of(synthetic_batch(batch_size=3, kind=kind)))
    assert logits.shape == (3, 91) and logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_forward_accepts_only_model_inputs():
    assert list(inspect.signature(build_model(ExperimentConfig()).forward).parameters) == list(INPUTS)


def test_initial_predictions_are_uniform():
    batch = synthetic_batch(batch_size=8)
    loss = F.cross_entropy(make_model()(**inputs_of(batch)), batch["targets"])
    assert loss.item() == pytest.approx(math.log(91), abs=1e-5)


def test_padding_is_ignored_even_when_padded_slots_hold_garbage():
    model = make_model(randomize_head=True).eval()
    batch = synthetic_batch(batch_size=4, seed=1)
    padded = {k: v.clone() for k, v in batch.items()}
    extra = 3
    padded["images"] = torch.cat([batch["images"], torch.randn(4, extra, 1, 28, 28) * 50], dim=1)
    padded["token_ids"] = torch.cat([batch["token_ids"], torch.randint(1, 11, (4, extra))], dim=1)
    padded["is_image"] = torch.cat([batch["is_image"], torch.ones(4, extra, dtype=torch.bool)], dim=1)
    padded["padding_mask"] = torch.cat([batch["padding_mask"], torch.ones(4, extra, dtype=torch.bool)], dim=1)
    with torch.no_grad():
        assert torch.allclose(model(**inputs_of(batch)), model(**inputs_of(padded)), atol=1e-5)


def test_permutation_invariance_in_eval_mode():
    model = make_model(randomize_head=True).eval()
    batch = synthetic_batch(batch_size=4, seed=2)
    order = torch.randperm(10, generator=torch.Generator().manual_seed(0))
    permuted = {k: (v[:, order] if v.dim() >= 2 else v) for k, v in batch.items()}
    with torch.no_grad():
        assert torch.allclose(model(**inputs_of(batch)), model(**inputs_of(permuted)), atol=1e-5)


def test_encoder_is_bidirectional_no_causal_mask():
    model = make_model().eval()
    batch = synthetic_batch(batch_size=1, kind="all_text", seed=3)
    changed = {k: v.clone() for k, v in batch.items()}
    changed["token_ids"][0, -1] = batch["token_ids"][0, -1] % 10 + 1  # alter only the LAST operand
    with torch.no_grad():
        first = model.encode(model.embed_operands(**inputs_of(batch)), batch["padding_mask"])[0, 0]
        second = model.encode(model.embed_operands(**inputs_of(changed)), changed["padding_mask"])[0, 0]
    assert not torch.allclose(first, second)  # position 0 sees position 9


def _grad_norm(module):
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    return None if not grads else float(torch.sqrt(sum((g**2).sum() for g in grads)))


@pytest.mark.parametrize("kind", ["mixed", "all_text", "all_image"])
def test_gradients_flow_only_through_active_branches(kind):
    model = make_model(randomize_head=True)
    batch = synthetic_batch(batch_size=4, kind=kind, seed=4)
    F.cross_entropy(model(**inputs_of(batch)), batch["targets"]).backward()
    for module in (model.layers, model.head, model.final_norm):
        norm = _grad_norm(module)
        assert norm is not None and math.isfinite(norm) and norm > 0
    cnn_norm = _grad_norm(model.image_encoder)
    text_rows = model.token_embedding.weight.grad[1:]
    if kind == "all_text":
        assert cnn_norm is None  # CNN never ran
    else:
        assert cnn_norm is not None and math.isfinite(cnn_norm) and cnn_norm > 0
    if kind == "all_image":
        assert (text_rows == 0).all()  # no text operands, so no digit-token gradient
    else:
        assert text_rows.abs().sum() > 0
    assert (model.token_embedding.weight.grad[PAD_ID] == 0).all()


def test_encoder_layers_are_independently_initialized():
    model = make_model()
    first, second = model.layers[0], model.layers[1]
    assert first is not second
    assert not torch.equal(first.linear1.weight, second.linear1.weight)
    assert not torch.equal(first.self_attn.in_proj_weight, second.self_attn.in_proj_weight)


def test_modality_embedding_is_optional():
    model = make_model(use_modality_embedding=False).eval()
    assert model.modality_embedding is None
    assert model(**inputs_of(synthetic_batch())).shape == (4, 91)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_cpu_and_mps_agree():
    model = make_model(randomize_head=True).eval()
    batch = synthetic_batch(batch_size=4, seed=5)
    with torch.no_grad():
        cpu_logits = model(**inputs_of(batch))
        mps_logits = model.to("mps")(**{k: v.to("mps") for k, v in inputs_of(batch).items()}).cpu()
    assert torch.allclose(cpu_logits, mps_logits, atol=1e-4)
