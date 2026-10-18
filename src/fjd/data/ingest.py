"""Stage `ingest`: merge every dataset in params.yaml into the canonical layout."""

import pandas as pd

from fjd.data.io import INTERIM, write_parquet
from fjd.data.schema import COLUMNS, TARGET
from fjd.utils.config import PROJECT_ROOT, load_params

OUT = INTERIM / "ingested.parquet"


def load_dataset(name: str, path: str, target: str) -> pd.DataFrame:
    df = pd.read_csv(PROJECT_ROOT / path)
    df = df.rename(columns={target: TARGET})
    df.insert(0, "source", name)
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"dataset {name!r} is missing columns: {sorted(missing)}")
    return df[COLUMNS]


def ingest(datasets: list[dict[str, str]]) -> pd.DataFrame:
    frames = [load_dataset(d["name"], d["path"], d["target"]) for d in datasets]
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["source", "job_id"], kind="stable", ignore_index=True)


def main() -> None:
    params = load_params()
    write_parquet(ingest(params["datasets"]), OUT)


if __name__ == "__main__":
    main()
