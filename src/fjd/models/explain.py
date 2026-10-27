"""Why a posting got its score: each feature's contribution to the model's raw score.

Linear models (logistic regression, linear SVM, Complement Naive Bayes) are explained exactly: a
feature contributes its weight times its value. Boosted trees (XGBoost, LightGBM) use TreeSHAP,
from the boosters' own `pred_contrib`, so `shap` isn't needed. Contributions are in raw-score
units (log-odds or SVM margin) and positive ones push toward fraud.

The random forest and the transformer have no explainer, so `explain` returns nothing for them
(token occlusion for the transformer is planned only if it becomes the champion).

Classical models are `Pipeline(features, model)` (see `fjd.models.train`); tree models are
`Pipeline(select, clf)` inside the `model` step (see `fjd.models.classical`).
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from fjd.features.build import ONE_HOT_COLUMNS

LINEAR, TREE_SHAP = "linear", "tree_shap"
Kind = Literal["text", "tabular", "categorical"]  # the featurizer's blocks


@dataclass(frozen=True)
class Contribution:
    kind: Kind  # "text" is a TF-IDF term
    feature: str  # the term, the tabular feature's name, or "<column>=<value>"
    value: float  # contribution to the raw score; positive pushes toward fraud


def _steps(classifier: Any) -> tuple[Any, Any] | None:
    if isinstance(classifier, Pipeline) and list(classifier.named_steps) == ["features", "model"]:
        return classifier.named_steps["features"], classifier.named_steps["model"]
    return None


def _booster(model: Any) -> Any:
    if isinstance(model, Pipeline) and list(model.named_steps) == ["select", "clf"]:
        clf = model.named_steps["clf"]
        if type(clf).__name__ in {"XGBClassifier", "LGBMClassifier"}:
            return clf
    return None


def method(classifier: Any) -> str | None:
    """How `classifier` is explained: `LINEAR`, `TREE_SHAP`, or None if it can't be."""
    steps = _steps(classifier)
    if steps is None:
        return None
    model = steps[1]
    if isinstance(model, LogisticRegression | LinearSVC | ComplementNB):
        return LINEAR
    return TREE_SHAP if _booster(model) is not None else None


def _linear_weights(model: Any) -> np.ndarray:
    if isinstance(model, ComplementNB):
        # The fraud-vs-real log-likelihood difference is linear in the features.
        return np.asarray(model.feature_log_prob_[1] - model.feature_log_prob_[0])
    return np.asarray(model.coef_[0])


@lru_cache(maxsize=8)
def _feature_names(classifier: Any) -> np.ndarray:
    """Names of the features the final estimator sees. Cached: building the ~50k TF-IDF names
    takes longer than scoring a posting."""
    features, model = classifier.named_steps["features"], classifier.named_steps["model"]
    names = features.get_feature_names_out()
    if method(classifier) == TREE_SHAP:
        names = model.named_steps["select"].get_feature_names_out(names)
    return np.asarray(names)


def contributions(classifier: Any, postings: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature contributions (one row per posting) and the feature names. Rows sum to the
    raw score minus the model's intercept/base value."""
    steps = _steps(classifier)
    kind = method(classifier)
    if steps is None or kind is None:
        raise ValueError(f"no explainer for {type(classifier).__name__}")
    features, model = steps
    X = sparse.csr_matrix(features.transform(postings))
    names = _feature_names(classifier)
    if kind == LINEAR:
        return np.asarray(X.multiply(_linear_weights(model)).toarray()), names

    select, clf = model.named_steps["select"], model.named_steps["clf"]
    X = select.transform(X)
    if type(clf).__name__ == "XGBClassifier":
        from xgboost import DMatrix

        contrib = clf.get_booster().predict(DMatrix(X), pred_contribs=True)
    else:
        contrib = clf.predict(X, pred_contrib=True)
    contrib = contrib.toarray() if sparse.issparse(contrib) else np.asarray(contrib)
    return contrib[:, :-1], names  # the last column is the base value


def _describe(name: str) -> tuple[Kind, str]:
    """("text", "work from") for "text__work from", ("categorical", "country=US") for
    "categorical__country_US", and so on."""
    kind, feature = name.split("__", 1)
    if kind == "categorical":
        column = next(c for c in ONE_HOT_COLUMNS if feature.startswith(f"{c}_"))
        value = feature.removeprefix(f"{column}_")
        feature = f"{column}={'(infrequent)' if value == 'infrequent_sklearn' else value}"
    return cast(Kind, kind), feature


def explain(classifier: Any, posting: pd.DataFrame, top_k: int = 10) -> list[Contribution]:
    """The `top_k` largest contributions (by absolute value) for a one-posting frame, largest
    first. Empty if the model has no explainer."""
    if method(classifier) is None:
        return []
    values, names = contributions(classifier, posting)
    row = values[0]
    order = np.argsort(-np.abs(row), kind="stable")[:top_k]
    return [Contribution(*_describe(str(names[i])), float(row[i])) for i in order if row[i] != 0]
