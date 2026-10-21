"""The feature pipeline: TF-IDF over all text plus tabular features, from raw postings.

The fitted featurizer is the first step of every saved classical model, so models take raw
postings (canonical columns) as input. The functions below are pickled by reference: renaming
or moving them (or this module) breaks every saved model. They live here, not in the
`featurize` stage module, because functions defined in a `python -m` entry point are pickled
as `__main__.<name>` and can't be loaded anywhere else.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler, OneHotEncoder

from fjd.data.schema import BINARY_COLUMNS, CATEGORICAL_COLUMNS, TEXT_COLUMNS

# EMSCAD anonymized links/contacts as e.g. `#URL_<hash>#`; keep the kind, drop the hash.
PLACEHOLDER = r"#(URL|EMAIL|PHONE)_[^#]*#"
PLACEHOLDER_KINDS = ["url", "email", "phone"]
INPUT_COLUMNS = TEXT_COLUMNS + CATEGORICAL_COLUMNS + BINARY_COLUMNS  # what a model needs
ONE_HOT_COLUMNS = [
    "employment_type",
    "required_experience",
    "required_education",
    "function",
    "industry",
    "country",  # derived from `location`
]
TABULAR_FEATURES = (
    [f"missing_{c}" for c in TEXT_COLUMNS + CATEGORICAL_COLUMNS]
    + BINARY_COLUMNS
    + [f"has_{k}" for k in PLACEHOLDER_KINDS]
    + [f"log_words_{c}" for c in TEXT_COLUMNS]
)


def _joined_text(df: pd.DataFrame) -> pd.Series:
    return df[TEXT_COLUMNS].fillna("").agg(" ".join, axis=1)


def combine_text(df: pd.DataFrame) -> pd.Series:
    """All text fields as one document, placeholders replaced by `xxurl`/`xxemail`/`xxphone`."""
    return _joined_text(df).str.replace(PLACEHOLDER, r" xx\1 ", regex=True)


def tabular_frame(df: pd.DataFrame) -> pd.DataFrame:
    text = _joined_text(df)
    out = {f"missing_{c}": df[c].isna() for c in TEXT_COLUMNS + CATEGORICAL_COLUMNS}
    out |= {c: df[c] for c in BINARY_COLUMNS}
    out |= {f"has_{k}": text.str.contains(f"#{k.upper()}_", regex=False) for k in PLACEHOLDER_KINDS}
    out |= {f"log_words_{c}": np.log1p(df[c].fillna("").str.count(r"\S+")) for c in TEXT_COLUMNS}
    return pd.DataFrame(out)[TABULAR_FEATURES].astype(float)


def categorical_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df[ONE_HOT_COLUMNS[:-1]].copy()
    country = df["location"].str.split(",").str[0].str.strip()
    out["country"] = country.mask(country == "")
    return out.fillna("(missing)")


def _text_names(transformer: Any, input_features: Any) -> np.ndarray:
    return np.array(["text"])


def _tabular_names(transformer: Any, input_features: Any) -> np.ndarray:
    return np.array(TABULAR_FEATURES)


def _categorical_names(transformer: Any, input_features: Any) -> np.ndarray:
    return np.array(ONE_HOT_COLUMNS)


def build_featurizer(cfg: dict[str, Any]) -> ColumnTransformer:
    """Unfitted featurizer from `params.yaml: features`. Output is a sparse, non-negative matrix
    (non-negative so Naive Bayes can use it too)."""
    tfidf = {**cfg["tfidf"], "ngram_range": tuple(cfg["tfidf"]["ngram_range"])}
    return ColumnTransformer(
        [
            (
                "text",
                make_pipeline(
                    FunctionTransformer(combine_text, feature_names_out=_text_names),
                    TfidfVectorizer(**tfidf),
                ),
                TEXT_COLUMNS,
            ),
            (
                "tabular",
                make_pipeline(
                    FunctionTransformer(tabular_frame, feature_names_out=_tabular_names),
                    MaxAbsScaler(),
                ),
                INPUT_COLUMNS,
            ),
            (
                "categorical",
                make_pipeline(
                    FunctionTransformer(categorical_frame, feature_names_out=_categorical_names),
                    OneHotEncoder(
                        handle_unknown="infrequent_if_exist",
                        min_frequency=cfg["categorical_min_frequency"],
                    ),
                ),
                ONE_HOT_COLUMNS[:-1] + ["location"],
            ),
        ],
        sparse_threshold=1.0,
    )
