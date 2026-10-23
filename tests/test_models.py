from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import pytest
from optuna.trial import TrialState
from sklearn.pipeline import Pipeline

from fjd.data.schema import TARGET
from fjd.features.build import TABULAR_FEATURES, build_featurizer, combine_text
from fjd.models.classical import MODELS, build_model
from fjd.models.cv import cross_val_pr_auc, group_folds
from fjd.models.metrics import classification_metrics, fraud_scores, tune_threshold
from fjd.models.tune import summarize, tune
from tests.test_data import make_df

FEATURES = {
    "tfidf": {"ngram_range": [1, 2], "min_df": 1, "sublinear_tf": True},
    "categorical_min_frequency": 2,
}


def make_postings(n: int = 120) -> pd.DataFrame:
    """Fraud postings mention a phone placeholder and lack a logo, so models can learn them."""
    df = make_df(n)
    fraud = df[TARGET] == 1
    df.loc[fraud, "description"] = "Earn money fast from home call #PHONE_abc123# now"
    df.loc[~fraud, "description"] = "Engineer role building data pipelines in a team"
    df.loc[fraud, "has_company_logo"] = 0
    df["location"] = np.where(fraud, "US, TX, Houston", "GB, LND, London")
    df["employment_type"] = np.where(fraud, "Part-time", "Full-time")
    df["group_id"] = df["job_id"] // 2
    return df


def test_combine_text_replaces_placeholders() -> None:
    df = make_df(1)
    df.loc[0, "description"] = "mail #EMAIL_0a1b# or visit #URL_ff00#"
    assert combine_text(df).iloc[0].split() == [
        "Job",
        "0",
        "mail",
        "xxEMAIL",
        "or",
        "visit",
        "xxURL",
    ]


def test_featurizer_output_is_sparse_nonnegative_and_named() -> None:
    df = make_postings()
    featurizer = build_featurizer(FEATURES).fit(df)
    X = featurizer.transform(df)
    names = featurizer.get_feature_names_out()
    assert X.shape == (len(df), len(names)) and hasattr(X, "tocsr")
    assert X.min() >= 0  # Naive Bayes needs non-negative features
    assert {f"tabular__{f}" for f in TABULAR_FEATURES} <= set(names)
    assert "categorical__country_US" in names and "text__xxphone" in names


def test_featurizer_handles_unseen_categories() -> None:
    df = make_postings()
    featurizer = build_featurizer(FEATURES).fit(df)
    new = df.head(2).copy()
    new["location"] = "ZZ, Nowhere"
    assert featurizer.transform(new).shape[0] == 2


@pytest.mark.parametrize("name", sorted(MODELS))
def test_every_model_learns_and_is_deterministic(name: str) -> None:
    df = make_postings()
    featurizer = build_featurizer(FEATURES).fit(df)
    X, y = featurizer.transform(df), df[TARGET]
    trees = {"random_forest", "xgboost", "lightgbm"}
    hp = {"n_estimators": 20, "k_features": 10} if name in trees else {}
    a = fraud_scores(build_model(name, hp, seed=0).fit(X, y), X)
    b = fraud_scores(build_model(name, hp, seed=0).fit(X, y), X)
    np.testing.assert_array_equal(a, b)
    assert classification_metrics(y, a, tune_threshold(y, a))["pr_auc"] > 0.9


def test_pipeline_predicts_from_raw_postings() -> None:
    df = make_postings()
    featurizer = build_featurizer(FEATURES).fit(df)
    model = build_model("logreg", {}, seed=0).fit(featurizer.transform(df), df[TARGET])
    pipeline = Pipeline([("features", featurizer), ("model", model)])
    assert fraud_scores(pipeline, df.drop(columns=[TARGET, "group_id"])).shape == (len(df),)


def test_build_model_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        build_model("svm_rbf", {}, seed=0)


@pytest.mark.parametrize("param", ["subsample", "colsample_bytree"])
def test_xgboost_rejects_sampling(param: str) -> None:
    # XGBoost's sampling is not reproducible across OSes (see fjd.models.classical).
    with pytest.raises(TypeError, match=param):
        build_model("xgboost", {param: 0.8}, seed=0)


def test_tune_threshold_maximizes_f1() -> None:
    y = np.array([0, 0, 0, 1, 1, 0, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.65, 0.9])
    # >= 0.6 gives P=3/4, R=1 (F1 0.857); >= 0.7 gives P=1, R=2/3 (F1 0.8).
    assert tune_threshold(y, scores) == 0.6
    m = classification_metrics(y, scores, 0.6)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (3, 1, 0, 3)
    assert m["f1"] == round(6 / 7, 6)


def test_group_folds_never_split_a_group() -> None:
    df = make_postings()
    folds = group_folds(df, FEATURES, n_splits=3, seed=0)
    assert len(folds) == 3
    assert sum(f.X_val.shape[0] for f in folds) == len(df)
    scores = cross_val_pr_auc("naive_bayes", {}, folds, seed=0)
    assert len(scores) == 3 and all(0 <= s <= 1 for s in scores)


def test_tune_is_deterministic() -> None:
    folds = group_folds(make_postings(), FEATURES, n_splits=3, seed=0)
    a, b = (summarize(tune("logreg", folds, n_trials=3, seed=0)) for _ in range(2))
    assert a == b and set(a["best_params"]) == {"C"}


def test_resumed_study_matches_uninterrupted(tmp_path: Path) -> None:
    folds = group_folds(make_postings(), FEATURES, n_splits=3, seed=0)
    storage = f"sqlite:///{(tmp_path / 'optuna.db').as_posix()}"
    uninterrupted = tune("naive_bayes", folds, n_trials=4, seed=0)

    tune("naive_bayes", folds, n_trials=2, seed=0, storage=storage, study_name="s")
    optuna.load_study(study_name="s", storage=storage).ask()  # a trial a killed run left behind
    resumed = tune("naive_bayes", folds, n_trials=4, seed=0, storage=storage, study_name="s")

    def completed(study: optuna.Study) -> list[tuple[dict[str, Any], float | None]]:
        return [(t.params, t.value) for t in study.trials if t.state == TrialState.COMPLETE]

    assert completed(resumed) == completed(uninterrupted)
    assert summarize(resumed) == summarize(uninterrupted)
