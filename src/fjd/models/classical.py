"""Classical model families: one per key of `params.yaml: classical`.

Tunable hyperparameters come from params.yaml. The fixed settings here handle class imbalance
(~4.5% fraud) and reproducibility: seeds, and a fixed thread count for the boosters so results
don't depend on the machine's core count.

Tree models first keep the `k_features` columns with the highest chi-squared score: on all ~50k
TF-IDF columns they are both slow (minutes per fit) and worse than on a selected subset.

Cross-platform reproducibility (Windows vs Linux, checked in Docker) constrains two settings:
XGBoost's row/column sampling draws differently per C++ standard library, so it is fixed off;
liblinear's default tolerance stops a weakly regularised logreg at OS-dependent points.
"""

from collections.abc import Callable
from typing import Any

from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

BOOSTER_THREADS = 2  # tuning runs 5 folds at once; more threads oversubscribe the CPU
DEFAULT_K_FEATURES = 5000


def _select_then(model: BaseEstimator, k_features: int) -> Pipeline:
    # chi2 needs non-negative features, which the featurizer guarantees.
    return Pipeline([("select", SelectKBest(chi2, k=k_features)), ("clf", model)])


def _logreg(seed: int, **hp: Any) -> BaseEstimator:
    return LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        tol=1e-6,  # the default 1e-4 differs across OSes in the 6th decimal of PR-AUC
        max_iter=5000,
        random_state=seed,
        **hp,
    )


def _linear_svm(seed: int, **hp: Any) -> BaseEstimator:
    # No predict_proba: scores are margins, which is fine for PR-AUC and a tuned threshold.
    return LinearSVC(class_weight="balanced", max_iter=20000, random_state=seed, **hp)


def _naive_bayes(seed: int, **hp: Any) -> BaseEstimator:
    # ComplementNB is the Naive Bayes variant suited to imbalanced text classes.
    return ComplementNB(**hp)


def _random_forest(seed: int, k_features: int = DEFAULT_K_FEATURES, **hp: Any) -> Pipeline:
    model = RandomForestClassifier(
        class_weight="balanced_subsample", n_jobs=-1, random_state=seed, **hp
    )
    return _select_then(model, k_features)


def _xgboost(seed: int, k_features: int = DEFAULT_K_FEATURES, **hp: Any) -> Pipeline:
    model = XGBClassifier(
        tree_method="hist",
        eval_metric="aucpr",
        # Any sampling (subsample, colsample_*) gives different models on Windows and Linux.
        # Passing these in params.yaml raises a duplicate-keyword TypeError, which is intended.
        subsample=1.0,
        colsample_bytree=1.0,
        colsample_bylevel=1.0,
        colsample_bynode=1.0,
        n_jobs=BOOSTER_THREADS,
        random_state=seed,
        **hp,
    )
    return _select_then(model, k_features)


def _lightgbm(seed: int, k_features: int = DEFAULT_K_FEATURES, **hp: Any) -> Pipeline:
    model = LGBMClassifier(
        deterministic=True,
        force_col_wise=True,
        subsample_freq=1,  # otherwise `subsample` is ignored
        n_jobs=BOOSTER_THREADS,
        random_state=seed,
        verbose=-1,
        **hp,
    )
    return _select_then(model, k_features)


MODELS: dict[str, Callable[..., BaseEstimator]] = {
    "logreg": _logreg,
    "linear_svm": _linear_svm,
    "naive_bayes": _naive_bayes,
    "random_forest": _random_forest,
    "xgboost": _xgboost,
    "lightgbm": _lightgbm,
}


def build_model(name: str, hyperparams: dict[str, Any], seed: int) -> BaseEstimator:
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; expected one of {sorted(MODELS)}")
    return MODELS[name](seed, **hyperparams)
