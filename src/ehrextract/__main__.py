"""Allow `python -m ehrextract` invocation."""

from ehrextract.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
