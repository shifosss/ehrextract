# Contributing to ehrextract

Thanks for your interest. This is a small, focused library; we're happy to accept high-quality patches.

## Development setup

```bash
git clone https://github.com/shifosss/ehrextract
cd ehrextract
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,openai,anthropic]"
```

For HuggingFace local-inference development, also install the `hf` extra (pulls torch, transformers, peft, accelerate -- heavy):

```bash
pip install -e ".[dev,openai,anthropic,hf]"
```

## Running tests

```bash
pytest tests/ehrextract -q
```

GPU-gated HuggingFace integration tests are skipped by default. To run them on a machine with CUDA:

```bash
pytest tests/ehrextract -m gpu -v
```

## Pull request guidelines

1. **One feature per PR.** Bug fixes can be bundled if they're related.
2. **Tests required.** New code paths need at least one test. Bug fixes need a regression test.
3. **Keep the public API stable.** Breaking changes to `ehrextract.{Extractor, providers, schema}` need a major-version bump and a CHANGELOG entry.
4. **No PHI in fixtures.** All test fixtures must be synthetic.
5. **Match the existing style.** Type hints throughout, docstrings on public functions, no `print` statements in library code.

## Documentation

User-facing docs live in [`docs/ehrextract/`](docs/ehrextract). The [`README.md`](README.md) is the entry point. Keep examples runnable.

## Reporting bugs

Open a GitHub issue with: minimal reproduction, expected vs actual behavior, the provider/model you're using, and `python --version` + `pip show ehrextract`.

For security issues, see [SECURITY.md](SECURITY.md) -- please do not report them in public issues.
