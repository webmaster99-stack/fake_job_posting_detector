"""Group-aware cross-validation on the train split.

Folds come from StratifiedGroupKFold on `group_id`, so a description template or company never
sits on both sides of a fold (see `fjd.data.groups`). The featurizer is refit inside each fold,
so TF-IDF vocabulary and IDF weights never see the fold's held-out rows.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold

from fjd.data.schema import GROUP, TARGET
from fjd.features.build import build_featurizer
from fjd.models.classical import build_model
from fjd.models.metrics import fraud_scores, pr_auc


@dataclass
class Fold:
    X_train: Any
    y_train: np.ndarray
    X_val: Any
    y_val: np.ndarray
    feature_names: np.ndarray  # columns of X_*, prefixed by block: text__, tabular__, ...


def group_folds(
    df: pd.DataFrame, features_cfg: dict[str, Any], n_splits: int, seed: int
) -> list[Fold]:
    """Featurized folds, built once and reused across models and Optuna trials."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for train_idx, val_idx in cv.split(df, df[TARGET], groups=df[GROUP]):
        train, val = df.iloc[train_idx], df.iloc[val_idx]
        featurizer = build_featurizer(features_cfg).fit(train)
        folds.append(
            Fold(
                featurizer.transform(train),
                train[TARGET].to_numpy(),
                featurizer.transform(val),
                val[TARGET].to_numpy(),
                featurizer.get_feature_names_out(),
            )
        )
    return folds


def fold_pr_auc(name: str, hyperparams: dict[str, Any], fold: Fold, seed: int) -> float:
    model = build_model(name, hyperparams, seed).fit(fold.X_train, fold.y_train)
    return pr_auc(fold.y_val, fraud_scores(model, fold.X_val))


def cross_val_pr_auc(
    name: str, hyperparams: dict[str, Any], folds: list[Fold], seed: int
) -> list[float]:
    """Folds run in parallel threads (the model libraries release the GIL). Each fold's result
    is independent of scheduling, so this stays deterministic."""
    jobs = (delayed(fold_pr_auc)(name, hyperparams, fold, seed) for fold in folds)
    return list(Parallel(n_jobs=len(folds), prefer="threads")(jobs))
