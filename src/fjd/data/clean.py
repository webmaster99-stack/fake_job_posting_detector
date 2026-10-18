"""Stage `clean_dedupe`: normalize whitespace and drop duplicate postings.

Duplicates must go before the split: ~15% of EMSCAD rows share their text with
another row (same posting re-listed in a different location), which would leak
between train and test.
"""

import pandas as pd

from fjd.data.ingest import OUT as INGESTED
from fjd.data.io import INTERIM, REPORTS, read_parquet, write_json, write_parquet
from fjd.data.schema import CATEGORICAL_COLUMNS, TARGET, TEXT_COLUMNS
from fjd.utils.config import load_params

OUT = INTERIM / "clean.parquet"
REPORT = REPORTS / "clean.json"


def normalize_whitespace(s: pd.Series) -> pd.Series:
    out = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return out.mask(out == "")  # empty strings become missing


def dedupe_key(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    joined = df[columns].fillna("").agg(" ".join, axis=1)
    return joined.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


def clean(df: pd.DataFrame, dedupe_on: list[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    df = df.copy()
    for col in TEXT_COLUMNS + CATEGORICAL_COLUMNS:
        df[col] = normalize_whitespace(df[col])

    key = dedupe_key(df, dedupe_on)
    conflicting = int((df.groupby(key)[TARGET].nunique() > 1).sum())
    # Rows are sorted by (source, job_id), so `keep="first"` keeps the lowest id.
    keep = ~key.duplicated(keep="first")
    out = df[keep].reset_index(drop=True)
    stats = {
        "rows_in": len(df),
        "rows_out": len(out),
        "duplicates_removed": int((~keep).sum()),
        "conflicting_label_groups": conflicting,
    }
    return out, stats


def main() -> None:
    params = load_params()
    out, stats = clean(read_parquet(INGESTED), params["clean"]["dedupe_on"])
    write_parquet(out, OUT)
    write_json(stats, REPORT)


if __name__ == "__main__":
    main()
