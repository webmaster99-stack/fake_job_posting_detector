"""How sure is a model ranking? Grouped bootstrap of PR-AUC, for notebooks/05_model_comparison.

Val and test each hold ~100 fraud postings, and postings sharing a template or company
(`group_id`) are not independent. Resampling whole groups with replacement gives an honest spread
of each model's PR-AUC, and paired differences: every model is scored on the same resamples.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def group_bootstrap_pr_auc(
    y: Any, scores: dict[str, np.ndarray], groups: Any, n_resamples: int, seed: int
) -> pd.DataFrame:
    """One row per resample, one column per model: PR-AUC on that resample of the groups."""
    y = np.asarray(y)
    codes, uniques = pd.factorize(np.asarray(groups))
    members = [np.flatnonzero(codes == g) for g in range(len(uniques))]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    while len(rows) < n_resamples:
        idx = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
        if y[idx].min() == y[idx].max():  # PR-AUC needs both classes
            continue
        rows.append({name: average_precision_score(y[idx], s[idx]) for name, s in scores.items()})
    return pd.DataFrame(rows)


def paired_summary(boot: pd.DataFrame, reference: str, level: float = 0.95) -> pd.DataFrame:
    """Each model's PR-AUC difference from `reference` across resamples: mean, CI and the share
    of resamples in which the model beats the reference."""
    diff = boot.sub(boot[reference], axis=0)
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    return pd.DataFrame(
        {
            "pr_auc_mean": boot.mean(),
            "diff_mean": diff.mean(),
            f"diff_ci_{lo:.3f}": diff.quantile(lo),
            f"diff_ci_{hi:.3f}": diff.quantile(hi),
            "p_beats_reference": (diff > 0).mean(),
        }
    ).sort_values("pr_auc_mean", ascending=False)
