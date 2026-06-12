"""CLI entrypoint for ehrextract."""

import argparse
import logging
import os
import sys
from pathlib import Path

from ehrextract import __version__
from ehrextract.pipeline import extract
from ehrextract.providers import PROVIDER_NAMES
from ehrextract.schema import SchemaError

logger = logging.getLogger(__name__)

_LOG_HANDLER: logging.Handler | None = None


def _setup_logging(verbosity: int) -> None:
    global _LOG_HANDLER
    level = logging.WARNING
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    root = logging.getLogger("ehrextract")
    root.setLevel(level)
    if _LOG_HANDLER is None:  # idempotent: repeated main() calls add one handler
        _LOG_HANDLER = logging.StreamHandler(sys.stderr)
        _LOG_HANDLER.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        root.addHandler(_LOG_HANDLER)


def _resolve_prompt(prompt_arg: str) -> str:
    p = Path(prompt_arg)
    if p.exists():
        return p.read_text(encoding="utf-8")
    looks_like_path = (
        os.sep in prompt_arg
        or (os.altsep is not None and os.altsep in prompt_arg)
        or prompt_arg.lower().endswith((".txt", ".md"))
    )
    if looks_like_path:
        raise FileNotFoundError(f"prompt file not found: {prompt_arg}")
    return prompt_arg


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ehrextract", description="Structured extraction from clinical notes"
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("--task", required=True, help="Built-in task name or task YAML path")
    parser.add_argument("--prompt", help="System prompt override -- file path or inline text")
    parser.add_argument("--provider", default="huggingface", choices=PROVIDER_NAMES)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", help="HF only -- optional PEFT adapter dir")
    parser.add_argument("--base-url", help="OpenAI-compat base URL override")
    parser.add_argument("--api-key-env", help="Env var name for API key")
    parser.add_argument("--input", required=True, help="Input file or '-' for stdin")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--id-column", default="note_id")
    parser.add_argument("--text-column", default="note_text")
    parser.add_argument("--max-concurrency", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument(
        "--trust-remote-code", action="store_true",
        help="HF only -- pass trust_remote_code=True to transformers loaders",
    )
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16",
        help="HF only -- model load dtype",
    )
    parser.add_argument("--ack-egress", action="store_true", help="Suppress the data egress notice")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _setup_logging(0 if args.quiet else args.verbose + 1)

    # Only explicit flags override task / GenerationConfig defaults.
    overrides = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
    }
    generation = {k: v for k, v in overrides.items() if v is not None} or None

    try:
        api_key = None
        if args.api_key_env is not None:
            api_key = os.environ.get(args.api_key_env)
            if api_key is None:
                raise ValueError(
                    f"--api-key-env: environment variable {args.api_key_env!r} is not set"
                )
        prompt = _resolve_prompt(args.prompt) if args.prompt is not None else None
        notes = sys.stdin.read() if args.input == "-" else Path(args.input)
        df = extract(
            notes,
            args.task,
            provider=args.provider,
            model=args.model,
            adapter=args.adapter,
            base_url=args.base_url,
            api_key=api_key,
            prompt=prompt,
            generation=generation,
            output=args.output,
            max_concurrency=args.max_concurrency,
            id_column=args.id_column,
            text_column=args.text_column,
            on_egress="silent" if args.ack_egress else "warn",
            trust_remote_code=args.trust_remote_code,
            dtype=args.dtype,
        )
    except (SchemaError, FileNotFoundError, ValueError) as e:
        print(f"ehrextract: error: {e}", file=sys.stderr)
        return 2

    rows = len(df)
    parsed = int(df["parse_success"].sum()) if rows else 0
    provider_errors = int((df["finish_reason"] == "error").sum()) if rows else 0
    logger.info(
        "done: %d rows, %d parsed, %d provider errors", rows, parsed, provider_errors
    )
    return 1 if rows and provider_errors == rows else 0
