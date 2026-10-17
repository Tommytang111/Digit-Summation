import pytest

from tokenizer import (
    BOS_ID,
    EOS_ID,
    OUT_PAD_ID,
    OUTPUT_VOCAB,
    OUTPUT_VOCAB_SIZE,
    PAD_ID,
    VOCAB,
    answer_ids,
    answer_text,
    encode_operand,
    parse_generated,
    teacher_forcing_pair,
)


def test_output_vocabulary_is_separate_and_fixed():
    assert (OUT_PAD_ID, BOS_ID, EOS_ID) == (0, 1, 2)
    assert OUTPUT_VOCAB_SIZE == 13
    assert [OUTPUT_VOCAB[str(d)] for d in range(10)] == list(range(3, 13))
    assert PAD_ID == 0 and VOCAB["0"] == 1 and encode_operand(0) == 1  # input vocabulary unchanged


@pytest.mark.parametrize("total, text", [(0, "0"), (7, "7"), (43, "43"), (90, "90"), (100, "100"), (180, "180")])
def test_canonical_answers(total, text):
    assert answer_text(total) == text


def test_answer_ids_and_teacher_forcing_shift():
    ids = answer_ids("43")
    assert ids == [BOS_ID, OUTPUT_VOCAB["4"], OUTPUT_VOCAB["3"], EOS_ID]
    decoder_input, decoder_target = teacher_forcing_pair(ids)
    assert decoder_input == [BOS_ID, OUTPUT_VOCAB["4"], OUTPUT_VOCAB["3"]]  # [BOS, "4", "3"]
    assert decoder_target == [OUTPUT_VOCAB["4"], OUTPUT_VOCAB["3"], EOS_ID]  # ["4", "3", EOS]


@pytest.mark.parametrize("bad", ["", "043", "4a", "-1"])
def test_answer_ids_reject_non_canonical(bad):
    with pytest.raises(ValueError):
        answer_ids(bad)


def test_teacher_forcing_pair_requires_bos_and_eos():
    with pytest.raises(ValueError):
        teacher_forcing_pair([OUTPUT_VOCAB["4"], EOS_ID])


def d(text: str) -> list[int]:
    return [OUTPUT_VOCAB[c] for c in text]


@pytest.mark.parametrize(
    "generated, expected",
    [
        (d("43") + [EOS_ID], ("43", "valid")),
        (d("0") + [EOS_ID, OUT_PAD_ID, OUT_PAD_ID], ("0", "valid")),  # tokens after EOS are ignored
        (d("100") + [EOS_ID], ("100", "valid")),  # three digits are allowed
        (d("1000"), (None, "missing_eos")),  # cap reached without EOS
        ([EOS_ID], (None, "empty")),
        (d("4") + [OUT_PAD_ID, EOS_ID], (None, "special_token")),
        ([BOS_ID] + d("4") + [EOS_ID], (None, "special_token")),
        (d("05") + [EOS_ID], (None, "leading_zero")),
    ],
)
def test_parse_generated(generated, expected):
    assert parse_generated(generated) == expected
