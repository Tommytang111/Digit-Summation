import numpy as np
import pytest

from tokenizer import PAD_ID, VOCAB, VOCAB_SIZE, class_id_to_text, decode_tokens, encode_operand, number_to_string, tokenize


def test_fixed_vocabulary():
    assert PAD_ID == 0 and VOCAB["<pad>"] == 0
    assert VOCAB_SIZE == 11
    assert [VOCAB[str(d)] for d in range(10)] == list(range(1, 11))


def test_integer_to_string_to_token_chain():
    assert number_to_string(7) == "7"
    assert tokenize("7") == [8]
    assert encode_operand(7) == 8
    assert encode_operand(np.int64(3)) == 4  # numpy integers from manifests are accepted


def test_zero_is_not_padding():
    assert encode_operand(0) == 1
    assert encode_operand(0) != PAD_ID


def test_multi_character_strings_tokenize_per_character_but_operands_must_be_single_digit():
    assert tokenize("42") == [5, 3]
    with pytest.raises(ValueError):
        encode_operand(10)


@pytest.mark.parametrize("bad", [-1, 10, 123])
def test_out_of_range_operands_rejected(bad):
    with pytest.raises(ValueError):
        encode_operand(bad)


@pytest.mark.parametrize("bad", [True, 3.0, "3", None])
def test_non_integer_operands_rejected(bad):
    with pytest.raises(TypeError):
        number_to_string(bad)


@pytest.mark.parametrize("bad", ["", "a", "4a", " 4"])
def test_invalid_strings_rejected(bad):
    with pytest.raises(ValueError):
        tokenize(bad)


def test_decoding_and_answer_text():
    assert decode_tokens([8, PAD_ID, 1]) == "70"
    assert class_id_to_text(47) == "47"
    assert class_id_to_text(np.int64(0)) == "0"
