"""Input loaders + output writers with format detection by file extension."""

import json
from pathlib import Path

import pandas as pd

OUTPUT_EXTENSIONS = {".csv", ".jsonl", ".json", ".xlsx", ".parquet"}


def load_notes(path: str | Path, *, id_column: str, text_column: str) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        df = pd.DataFrame(rows)
    elif suffix == ".json":
        rows = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"{p}: .json input must be a JSON array of objects")
        df = pd.DataFrame(rows)
    elif suffix == ".csv":
        df = pd.read_csv(p, encoding="utf-8")
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(p)
    elif suffix == ".txt":
        lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        df = pd.DataFrame({id_column: range(len(lines)), text_column: lines})
    else:
        raise ValueError(f"unsupported input extension {suffix!r} (.jsonl/.json/.csv/.xlsx/.txt)")

    if text_column not in df.columns:
        raise ValueError(f"input is missing required column {text_column!r} (have: {list(df.columns)})")
    if id_column not in df.columns:
        df[id_column] = range(len(df))
    return df


def validate_output_path(path: str | Path) -> Path:
    """Reject unsupported output extensions up front (fail-fast, before any model work)."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix not in OUTPUT_EXTENSIONS:
        raise ValueError(
            f"unsupported output extension {suffix!r} (.csv/.jsonl/.json/.xlsx/.parquet)"
        )
    return p


def write_results(df: pd.DataFrame, path: str | Path) -> None:
    p = validate_output_path(path)
    suffix = p.suffix.lower()
    p.parent.mkdir(parents=True, exist_ok=True)

    if suffix == ".csv":
        df.to_csv(p, index=False)
    elif suffix == ".jsonl":
        # Real JSON types (booleans/ints, NaN -> null), not stringified cells.
        df.to_json(p, orient="records", lines=True, force_ascii=False)
    elif suffix == ".json":
        df.to_json(p, orient="records", indent=2, force_ascii=False)
    elif suffix == ".xlsx":
        df.to_excel(p, index=False)
    else:  # ".parquet" -- validate_output_path guarantees the extension set
        df.to_parquet(p, index=False)
