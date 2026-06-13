"""Constrained JSON decoding for the local HuggingFace provider.

Builds a HuggingFace ``prefix_allowed_tokens_fn`` (via lm-format-enforcer)
that restricts ``model.generate`` to emit only JSON conforming to the task's
JSON Schema: exactly the expected keys, enum fields limited to their allowed
values. Inference-time only; enabled per task or via ``--constrained``.

Uses lm-format-enforcer's stable core API (``JsonSchemaParser``,
``TokenEnforcer``, ``TokenEnforcerTokenizerData``) directly. The packaged
``lmformatenforcer.integrations.transformers`` convenience module is not used:
it imports ``PreTrainedTokenizerBase`` from a module path that was removed in
transformers 5.x. Building the tokenizer data manually is the documented
manual-integration path and works on transformers 4.x and 5.x alike.

A fresh ``TokenEnforcer`` is built per ``generate()`` call: the enforcer
memoizes one state per generated token, keyed by the full token sequence, and
never evicts -- one enforcer reused across a long run leaks memory without
bound. Only the expensive per-tokenizer vocab scan
(``TokenEnforcerTokenizerData``) is reusable; the provider caches it, one per
tokenizer.
"""

from typing import Any, Callable

__all__ = ["build_prefix_allowed_tokens_fn", "get_tokenizer_data"]


def _require_lmformatenforcer() -> None:
    try:
        import lmformatenforcer  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "constrained decoding requires the 'lm-format-enforcer' package. "
            "Install with: pip install 'ehrextract[hf]' (or: pip install "
            "lm-format-enforcer), or disable constrained decoding with "
            "--no-constrained / generation={'constrained': False}."
        ) from e


def _build_regular_tokens_list(tokenizer, vocab_size: int) -> list[tuple[int, str, bool]]:
    """Decode every non-special token, flagging word-start tokens (leading space).

    Mirrors lm-format-enforcer's own ``_build_regular_tokens_list``: prepend a
    "0" token and drop its first character to recover the leading space of
    word-start tokens, which single-token ``decode`` strips.
    """
    token_0 = tokenizer.encode("0")[-1]
    special_ids = set(tokenizer.all_special_ids)
    regular_tokens: list[tuple[int, str, bool]] = []
    for token_idx in range(vocab_size):
        if token_idx in special_ids:
            continue
        decoded_after_0 = tokenizer.decode([token_0, token_idx])[1:]
        decoded_regular = tokenizer.decode([token_idx])
        is_word_start = len(decoded_after_0) > len(decoded_regular)
        regular_tokens.append((token_idx, decoded_after_0, is_word_start))
    return regular_tokens


def get_tokenizer_data(tokenizer) -> Any:
    """Build lm-format-enforcer ``TokenEnforcerTokenizerData`` from a HF tokenizer.

    The vocab scan is expensive (~150k decodes for Qwen-family vocabularies);
    the result is task-independent and safe to reuse across all enforcers
    built for the same tokenizer.
    """
    _require_lmformatenforcer()
    from lmformatenforcer.tokenenforcer import TokenEnforcerTokenizerData

    vocab_size = len(tokenizer)
    regular_tokens = _build_regular_tokens_list(tokenizer, vocab_size)

    def decode_fn(tokens: list[int]) -> str:
        return tokenizer.decode(tokens).rstrip("�")

    return TokenEnforcerTokenizerData(
        regular_tokens, decode_fn, tokenizer.eos_token_id, False, vocab_size
    )


def build_prefix_allowed_tokens_fn(
    tokenizer_data: Any, json_schema: dict
) -> Callable[[int, Any], list[int]]:
    """Return a fresh ``prefix_allowed_tokens_fn`` for one ``generate()`` call.

    Builds a new ``TokenEnforcer`` per call (see module docstring). Pass the
    returned callable as ``model.generate(prefix_allowed_tokens_fn=...)``.
    """
    _require_lmformatenforcer()
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.tokenenforcer import TokenEnforcer

    enforcer = TokenEnforcer(tokenizer_data, JsonSchemaParser(json_schema))

    def prefix_allowed_tokens_fn(batch_id: int, sent) -> list[int]:
        return enforcer.get_allowed_tokens(sent.tolist()).allowed_tokens

    return prefix_allowed_tokens_fn
