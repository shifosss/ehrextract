"""Tests for constrained decoding (lm-format-enforcer stable-core integration).

Token-level correctness is tested with a real TokenEnforcer over a tiny fake
vocabulary -- no torch or GPU required.
"""

import sys

import pytest

lmformatenforcer = pytest.importorskip("lmformatenforcer")

from ehrextract import constrained
from ehrextract.constrained import (
    _build_regular_tokens_list,
    build_prefix_allowed_tokens_fn,
    get_tokenizer_data,
)

# Token ids for TinyTokenizer (see _VOCAB order below).
EOS, BRACE_OPEN, BRACE_CLOSE, QUOTE, LETTER_A, COLON, SPACE, Y, N, ZERO, X, HELLO = range(12)


class TinyTokenizer:
    """SentencePiece-style fake: word-start tokens carry a leading space that
    decode() strips at sequence start (metaspace behavior)."""

    _VOCAB: list[tuple[str, bool]] = [
        ("</s>", False),  # 0: special, eos
        ("{", False),
        ("}", False),
        ('"', False),
        ("a", False),
        (":", False),
        (" ", False),
        ("Y", False),
        ("N", False),
        ("0", False),
        ("x", False),
        ("hello", True),  # word-start: " hello" mid-sequence
    ]
    eos_token_id = EOS
    all_special_ids = [EOS]

    def __len__(self):
        return len(self._VOCAB)

    def encode(self, text):
        assert text == "0"
        return [ZERO]

    def decode(self, ids, skip_special_tokens=True):
        out = ""
        for tid in ids:
            if tid in self.all_special_ids:
                continue
            text, word_start = self._VOCAB[tid]
            out += " " + text if (word_start and out) else text
        return out


class _FakeSent:
    """Stand-in for the torch tensor generate() hands to prefix_allowed_tokens_fn."""

    def __init__(self, ids):
        self._ids = list(ids)

    def tolist(self):
        return self._ids


ENUM_SCHEMA = {
    "type": "object",
    "properties": {"a": {"type": "string", "enum": ["Y", "N"]}},
    "required": ["a"],
    "additionalProperties": False,
}


def _walk(fn, prompt, steps):
    """Feed tokens one at a time (as generate() does), asserting each step is allowed."""
    seq = list(prompt)
    allowed = fn(0, _FakeSent(seq))
    for t in steps:
        assert t in allowed, f"token {t} not allowed after {seq}"
        seq.append(t)
        allowed = fn(0, _FakeSent(seq))
    return allowed


def test_regular_tokens_word_start_detection():
    tok = TinyTokenizer()
    by_id = {tid: (text, ws) for tid, text, ws in _build_regular_tokens_list(tok, len(tok))}
    assert EOS not in by_id  # special tokens excluded
    assert by_id[LETTER_A] == ("a", False)
    assert by_id[HELLO] == (" hello", True)  # leading space recovered


def test_decode_fn_strips_trailing_replacement_char():
    class TruncatedUtfTokenizer(TinyTokenizer):
        def decode(self, ids, skip_special_tokens=True):
            return super().decode(ids) + "�"

    data = get_tokenizer_data(TruncatedUtfTokenizer())
    assert data.decoder([Y]) == "Y"


def test_tokenizer_data_built_from_fake_vocab():
    data = get_tokenizer_data(TinyTokenizer())
    assert data.vocab_size == 12
    assert data.eos_token_id == EOS
    assert len(data.regular_tokens) == 11  # 12 minus the special token


def test_first_allowed_token_is_open_brace():
    data = get_tokenizer_data(TinyTokenizer())
    fn = build_prefix_allowed_tokens_fn(data, ENUM_SCHEMA)
    allowed = fn(0, _FakeSent([ZERO]))  # first call roots at the prompt
    assert BRACE_OPEN in allowed
    assert X not in allowed
    assert BRACE_CLOSE not in allowed


def test_enum_values_enforced_at_value_position():
    data = get_tokenizer_data(TinyTokenizer())
    fn = build_prefix_allowed_tokens_fn(data, ENUM_SCHEMA)
    # Walk the prefix for: {"a": "
    allowed = _walk(fn, [ZERO], [BRACE_OPEN, QUOTE, LETTER_A, QUOTE, COLON, SPACE, QUOTE])
    assert set(allowed) == {Y, N}


def test_each_build_returns_independent_enforcer():
    data = get_tokenizer_data(TinyTokenizer())
    fn1 = build_prefix_allowed_tokens_fn(data, ENUM_SCHEMA)
    fn2 = build_prefix_allowed_tokens_fn(data, ENUM_SCHEMA)
    # Different prompts root independently; both start a fresh JSON object.
    assert BRACE_OPEN in fn1(0, _FakeSent([ZERO]))
    assert BRACE_OPEN in fn2(0, _FakeSent([ZERO, ZERO]))


def test_missing_lmformatenforcer_raises_actionable_importerror(monkeypatch):
    monkeypatch.setitem(sys.modules, "lmformatenforcer", None)
    with pytest.raises(ImportError, match=r"ehrextract\[hf\]"):
        constrained.get_tokenizer_data(TinyTokenizer())
