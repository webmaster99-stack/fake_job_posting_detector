"""Small IO helpers so every stage reads/writes artifacts the same way."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from fjd.utils.config import PROJECT_ROOT

INTERIM = PROJECT_ROOT / "data" / "interim"
PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS = PROJECT_ROOT / "reports"


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def read_split(name: str) -> pd.DataFrame:
    """`train`, `val` or `test` from the `split` stage."""
    return read_parquet(PROCESSED / f"{name}.parquet")


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Force LF so the file (and its hash in dvc.lock) is identical on Windows and Linux.
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
