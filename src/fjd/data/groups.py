"""Template groups: postings linked by a shared description opening or company profile.

Two rows are in the same group if they share *any* grouping key (connected components),
so a whole scam campaign or company ends up on one side of every split.
"""

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def _letters(s: pd.Series) -> pd.Series:
    return s.fillna("").str.lower().str.replace(r"[^a-z]", "", regex=True)


def grouping_keys(df: pd.DataFrame, template_chars: int) -> list[pd.Series]:
    return [_letters(df["description"]).str[:template_chars], _letters(df["company_profile"])]


def template_groups(df: pd.DataFrame, template_chars: int, min_key_chars: int) -> np.ndarray:
    """Return a group id per row. Keys shorter than `min_key_chars` link nothing."""
    n = len(df)
    idx = np.arange(n)
    rows, cols = [], []
    for key in grouping_keys(df, template_chars):
        codes = pd.factorize(key.where(key.str.len() >= min_key_chars))[0]
        linked = codes >= 0
        first = pd.Series(idx[linked]).groupby(codes[linked]).transform("min").to_numpy()
        rows.append(idx[linked])
        cols.append(first)
    r, c = np.concatenate(rows), np.concatenate(cols)
    graph = coo_matrix((np.ones(len(r), dtype=np.int8), (r, c)), shape=(n, n))
    return connected_components(graph, directed=False)[1]
