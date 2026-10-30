"""The columns drift monitoring compares between training data and live predictions.

They are the featurizer's own tabular and categorical inputs (`fjd.features.build`): missing-field
flags, the binary flags, link/email/phone flags, log word counts per text field, and the one-hot
categoricals (with `country` derived from `location`). TF-IDF terms are left out: tens of
thousands of sparse columns would drown the report, and the word counts and flags already show
when postings start to look different.
"""

import pandas as pd

from fjd.data.schema import BINARY_COLUMNS
from fjd.features.build import (
    INPUT_COLUMNS,
    ONE_HOT_COLUMNS,
    TABULAR_FEATURES,
    categorical_frame,
    tabular_frame,
)

NUMERICAL = TABULAR_FEATURES
CATEGORICAL = ONE_HOT_COLUMNS


def monitoring_frame(postings: pd.DataFrame) -> pd.DataFrame:
    """Drift columns (`NUMERICAL + CATEGORICAL`) from postings in canonical columns."""
    df = postings.reindex(columns=INPUT_COLUMNS)
    df[BINARY_COLUMNS] = df[BINARY_COLUMNS].fillna(0).astype(int)
    out = pd.concat([tabular_frame(df), categorical_frame(df)], axis=1)
    return out[NUMERICAL + CATEGORICAL].reset_index(drop=True)
