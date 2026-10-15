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
