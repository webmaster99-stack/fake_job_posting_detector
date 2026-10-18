"""Stage `split`: stratified, group-aware train/val/test split.

Rows sharing a description template or company profile (see `groups.py`) always land in
the same split, so test metrics measure performance on unseen campaigns/companies.
The `group_id` column is kept so cross-validation during tuning can respect it too.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from fjd.data.clean import OUT as CLEAN
from fjd.data.groups import template_groups
from fjd.data.io import PROCESSED, REPORTS, read_parquet, write_json, write_parquet
from fjd.data.schema import TARGET
from fjd.utils.config import load_params
from fjd.utils.seed import set_seed

SPLITS = ("train", "val", "test")
REPORT = REPORTS / "split.json"
GROUP = "group_id"


def _hold_out(df: pd.DataFrame, fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out one fold of a StratifiedGroupKFold with ~`fraction` of the rows."""
    n_splits = max(2, round(1 / fraction))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest_idx, held_idx = next(cv.split(df, df[TARGET], groups=df[GROUP]))
    return df.iloc[rest_idx], df.iloc[held_idx]


def split(
    df: pd.DataFrame, test_size: float, val_size: float, seed: int
) -> dict[str, pd.DataFrame]:
    """`test_size` and `val_size` are approximate fractions of the full dataset.

    Requires a `group_id` column. Fold sizes are approximate because whole groups move together.
    """
    rest, test = _hold_out(df, test_size, seed)
    train, val = _hold_out(rest, val_size / (1 - test_size), seed)
    return {"train": train, "val": val, "test": test}


def main() -> None:
    params = load_params()
    set_seed(params["seed"])
    cfg = params["split"]
    df = read_parquet(CLEAN)
    df[GROUP] = template_groups(df, cfg["template_chars"], cfg["min_key_chars"])
    parts = split(df, cfg["test_size"], cfg["val_size"], params["seed"])
    for name, part in parts.items():
        write_parquet(part.reset_index(drop=True), PROCESSED / f"{name}.parquet")

    group_sizes = df[GROUP].value_counts()
    write_json(
        {
            "groups": int(len(group_sizes)),
            "largest_group_rows": int(group_sizes.iloc[0]),
            **{
                name: {
                    "rows": len(part),
                    "share": round(len(part) / len(df), 4),
                    "fraud_rows": int(part[TARGET].sum()),
                    "fraud_rate": round(float(part[TARGET].mean()), 6),
                    "groups": int(part[GROUP].nunique()),
                }
                for name, part in parts.items()
            },
        },
        REPORT,
    )
    shared = set(parts["train"][GROUP]) & (set(parts["val"][GROUP]) | set(parts["test"][GROUP]))
    assert not shared and np.isin(parts["val"][GROUP], parts["test"][GROUP]).sum() == 0


if __name__ == "__main__":
    main()
