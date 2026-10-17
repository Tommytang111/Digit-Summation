"""Explicit digit-string vocabulary for numeric operands.

A numeric operand follows: integer -> decimal string -> token IDs -> learned embedding.

    7  ->  "7"  ->  [8]  ->  nn.Embedding row 8

Token IDs are *categorical indices*, not arithmetic values: the model never sees "8" as the
number eight; it learns a 128-d vector for the token whose string is "7".

Vocabulary (fixed, 11 entries):
    PAD_ID = 0          "<pad>"  (unused token slots: image operands and sequence padding)
    "0".."9" -> 1..10

Note that the digit 0 is token ID 1, *not* PAD_ID. Padding is never inferred from token IDs; the
batch carries a separate `padding_mask`.
"""

from __future__ import annotations

import numpy as np

PAD_TOKEN = "<pad>"
PAD_ID = 0
DIGIT_STRINGS = tuple(str(digit) for digit in range(10))
VOCAB: dict[str, int] = {PAD_TOKEN: PAD_ID, **{text: index + 1 for index, text in enumerate(DIGIT_STRINGS)}}
ID_TO_TOKEN: dict[int, str] = {token_id: token for token, token_id in VOCAB.items()}
VOCAB_SIZE = len(VOCAB)  # 11


def number_to_string(value: int) -> str:
    """Convert a non-negative integer operand to its decimal string (7 -> "7")."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"Operands must be integers, got {type(value).__name__}: {value!r}")
    if value < 0:
        raise ValueError(f"Operands must be non-negative, got {value}")
    return str(int(value))


def tokenize(text: str) -> list[int]:
    """Map a digit string to token IDs, one per character ("7" -> [8], "42" -> [5, 3])."""
    if not isinstance(text, str) or len(text) == 0:
        raise ValueError(f"Expected a non-empty digit string, got {text!r}")
    token_ids = []
    for character in text:
        if character not in DIGIT_STRINGS:
            raise ValueError(f"Character {character!r} in {text!r} is not in the digit vocabulary")
        token_ids.append(VOCAB[character])
    return token_ids


def encode_operand(value: int) -> int:
    """Main-task operand (0-9) -> exactly one token ID, via its string: 7 -> "7" -> 8.

    Multi-digit operands would tokenize to several IDs; supporting them is part of the
    (unimplemented) bonus design, so they are rejected here rather than silently truncated.
    """
    token_ids = tokenize(number_to_string(value))
    if len(token_ids) != 1:
        raise ValueError(f"Main-task operands are single digits 0-9; got {value}")
    return token_ids[0]


def decode_tokens(token_ids) -> str:
    """Token IDs -> string, skipping PAD ([8] -> "7")."""
    return "".join(ID_TO_TOKEN[int(token_id)] for token_id in token_ids if int(token_id) != PAD_ID)


def class_id_to_text(class_id: int) -> str:
    """Predicted class index -> answer text. Class k means "the sum is k" (47 -> "47")."""
    return number_to_string(int(class_id))


# ----------------------------------------------------------------------------- answer (output) vocabulary
#
# Used by the length-generalization bonus, whose decoder writes the sum character by character.
# It is a separate vocabulary from the input one above, and both are saved in bonus checkpoints.
#
#   OUT_PAD = 0  "<pad>"  target padding (ignored by the loss; never a valid generated token)
#   BOS     = 1  "<bos>"  first decoder input; never a target
#   EOS     = 2  "<eos>"  ends an answer
#   "0".."9" -> 3..12
#
# Canonical answers are plain decimal strings without leading zeros (except "0" itself):
#   43 -> "43" -> [BOS, "4", "3", EOS] = [1, 7, 6, 2]
# Teacher forcing shifts that sequence by one position:
#   decoder input  = [BOS, "4", "3"]   (what the decoder has seen so far)
#   decoder target = ["4", "3", EOS]   (the next character at each position)

OUT_PAD_TOKEN, BOS_TOKEN, EOS_TOKEN = "<pad>", "<bos>", "<eos>"
OUT_PAD_ID, BOS_ID, EOS_ID = 0, 1, 2
OUTPUT_VOCAB: dict[str, int] = {
    OUT_PAD_TOKEN: OUT_PAD_ID,
    BOS_TOKEN: BOS_ID,
    EOS_TOKEN: EOS_ID,
    **{text: index + 3 for index, text in enumerate(DIGIT_STRINGS)},
}
OUTPUT_ID_TO_TOKEN: dict[int, str] = {token_id: token for token, token_id in OUTPUT_VOCAB.items()}
OUTPUT_VOCAB_SIZE = len(OUTPUT_VOCAB)  # 13


def answer_text(total: int) -> str:
    """Canonical decimal answer for a non-negative sum (no leading zeros): 43 -> "43", 0 -> "0"."""
    return number_to_string(total)


def answer_ids(text: str) -> list[int]:
    """Answer string -> [BOS, digit IDs..., EOS] in the output vocabulary ("43" -> [1, 7, 6, 2])."""
    if not text or any(character not in DIGIT_STRINGS for character in text):
        raise ValueError(f"Answers are non-empty digit strings, got {text!r}")
    if len(text) > 1 and text[0] == "0":
        raise ValueError(f"Answers are canonical (no leading zeros), got {text!r}")
    return [BOS_ID, *(OUTPUT_VOCAB[character] for character in text), EOS_ID]


def teacher_forcing_pair(ids: list[int]) -> tuple[list[int], list[int]]:
    """[BOS, a, b, EOS] -> (decoder input [BOS, a, b], decoder target [a, b, EOS])."""
    if len(ids) < 2 or ids[0] != BOS_ID or ids[-1] != EOS_ID:
        raise ValueError(f"Expected [BOS, ..., EOS], got {ids}")
    return ids[:-1], ids[1:]


ANSWER_STATUSES = ("valid", "missing_eos", "empty", "special_token", "leading_zero")


def parse_generated(generated: list[int]) -> tuple[str | None, str]:
    """Generated token IDs (after BOS, up to the cap) -> (answer text or None, status).

    Everything after the first EOS is ignored. Status:
      valid          digits followed by EOS, canonical ("0" or no leading zero)
      missing_eos    the cap was reached without EOS (explicit generation failure)
      empty          EOS was the first token
      special_token  PAD or BOS appeared before EOS (no special-token suppression is applied)
      leading_zero   digits with a leading zero, e.g. "05" (not canonical, counted as invalid)
    """
    characters = []
    for token_id in generated:
        token_id = int(token_id)
        if token_id == EOS_ID:
            if not characters:
                return None, "empty"
            text = "".join(characters)
            if len(text) > 1 and text[0] == "0":
                return None, "leading_zero"
            return text, "valid"
        if token_id in (OUT_PAD_ID, BOS_ID):
            return None, "special_token"
        characters.append(OUTPUT_ID_TO_TOKEN[token_id])
    return None, "missing_eos"
