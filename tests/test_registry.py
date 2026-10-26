from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient
from sklearn.pipeline import Pipeline

from fjd.data.schema import BINARY_COLUMNS, TARGET
from fjd.features.build import build_featurizer
from fjd.models import select
from fjd.models.classical import build_model
from fjd.models.metrics import fraud_scores
from fjd.models.promote import Candidate, candidate, refusals
from fjd.models.registry import (
    CHALLENGER,
    CHAMPION,
    CONFIG_ARTIFACT,
    MODEL_ARTIFACT,
    Calibration,
    Detector,
    DetectorConfig,
    FraudDetectorModel,
    load_classifier,
)
from tests.test_models import FEATURES, make_postings

LIMITS = {"metric": "pr_auc", "max_model_size_mb": 300, "max_p95_latency_ms": 300}


def fit_pipeline(name: str, df: pd.DataFrame, **hp: Any) -> Pipeline:
    featurizer = build_featurizer(FEATURES).fit(df)
    model = build_model(name, hp, seed=0).fit(featurizer.transform(df), df[TARGET])
    return Pipeline([("features", featurizer), ("model", model)])


@pytest.fixture(scope="module")
def svm_detector(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """A LinearSVC (margins, not probabilities) wrapped as the registry serves it."""
    df = make_postings()
    path = tmp_path_factory.mktemp("models") / "linear_svm.joblib"
    joblib.dump(fit_pipeline("linear_svm", df), path)
    pipeline = load_classifier("linear_svm", path)
    scores = fraud_scores(pipeline, df)
    config = DetectorConfig(
        "linear_svm", float(np.median(scores)), Calibration.fit(df[TARGET], scores)
    )
    return {"df": df, "path": path, "scores": scores, "detector": Detector(config, pipeline)}


def test_calibration_keeps_the_ranking_and_the_labels(svm_detector: dict[str, Any]) -> None:
    df, scores, detector = svm_detector["df"], svm_detector["scores"], svm_detector["detector"]
    out = detector.predict(df)
    assert list(out.columns) == ["probability", "fraud", "threshold"]
    assert out["probability"].between(0, 1).all()
    order = np.argsort(scores, kind="stable")
    assert np.all(np.diff(out["probability"].to_numpy()[order]) >= 0)  # same ranking
    np.testing.assert_array_equal(out["fraud"], scores >= detector.config.threshold)
    # The probability threshold separates the same labels as the score threshold.
    np.testing.assert_array_equal(out["fraud"], out["probability"] >= out["threshold"])


def test_detector_scores_a_lone_posting_without_optional_fields(
    svm_detector: dict[str, Any],
) -> None:
    """As the API may send it: one posting, nullable fields left out."""
    one = svm_detector["df"].head(1)[["title", "description", *BINARY_COLUMNS]]
    out = svm_detector["detector"].predict(one)
    assert len(out) == 1 and 0 <= out["probability"].iloc[0] <= 1


def test_calibration_fits_the_fraud_rate() -> None:
    rng = np.random.default_rng(0)
    y = rng.random(5000) < 0.05
    scores = np.where(y, 1.0, -1.0) + rng.normal(0, 1, len(y))  # margins, any scale
    p = Calibration.fit(y, scores)(scores)
    assert abs(p.mean() - y.mean()) < 0.005  # logistic regression matches the base rate


def test_config_round_trips_through_json() -> None:
    config = DetectorConfig("logreg", 0.25, Calibration(2.0, -3.0))
    assert DetectorConfig.from_json(config.to_json()) == config
    assert config.threshold_probability == pytest.approx(1 / (1 + np.exp(2.5)))


def test_load_classifier_caps_threads(tmp_path: Path) -> None:
    path = tmp_path / "random_forest.joblib"
    joblib.dump(fit_pipeline("random_forest", make_postings(), n_estimators=5, k_features=10), path)
    assert load_classifier("random_forest", path).get_params()["model__clf__n_jobs"] == 1
    assert (
        load_classifier("random_forest", path, threads=None).get_params()["model__clf__n_jobs"]
        == -1
    )


def test_pyfunc_predicts_like_the_detector(svm_detector: dict[str, Any], tmp_path: Path) -> None:
    detector = svm_detector["detector"]
    config_path = tmp_path / "detector.json"
    config_path.write_text(detector.config.to_json(), encoding="utf-8")
    mlflow.pyfunc.save_model(
        tmp_path / "pyfunc",
        python_model=FraudDetectorModel(),
        artifacts={CONFIG_ARTIFACT: str(config_path), MODEL_ARTIFACT: str(svm_detector["path"])},
    )
    loaded = mlflow.pyfunc.load_model(str(tmp_path / "pyfunc"))
    df = svm_detector["df"]
    pd.testing.assert_frame_equal(loaded.predict(df), detector.predict(df))
    assert loaded.unwrap_python_model().detector.config == detector.config


def test_ranking_is_by_val_metric_then_name() -> None:
    metrics = {n: {"val": {"pr_auc": v}} for n, v in [("b", 0.7), ("a", 0.7), ("c", 0.8)]}
    assert select.ranking(metrics, "pr_auc") == ["c", "a", "b"]


def test_select_skips_models_outside_the_limits() -> None:
    deployable = {"big": False, "ok": True, "never_checked": True}
    name, checked = select.select(
        ["big", "ok", "never_checked"], lambda n: {"deployable": deployable[n]}
    )
    assert name == "ok" and list(checked) == ["big", "ok"]
    with pytest.raises(RuntimeError, match="no model"):
        select.select(["big"], lambda n: {"deployable": False})


def make_candidate(score: float, size: float = 10, p95: float = 50) -> Candidate:
    return Candidate("2", "logreg", score, size, p95, "md5")


def test_refusals() -> None:
    assert refusals(make_candidate(0.5), None, LIMITS) == []  # the first champion
    assert refusals(make_candidate(0.71), make_candidate(0.70), LIMITS) == []
    assert len(refusals(make_candidate(0.70), make_candidate(0.70), LIMITS)) == 1  # must beat it
    assert len(refusals(make_candidate(0.9, size=301, p95=301), None, LIMITS)) == 2


def test_register_sets_challenger(
    svm_detector: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(select, "model_path", lambda name: svm_detector["path"])
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("test")
    try:
        metrics = {"val_pr_auc": 0.6, "size_mb": 1.0, "p95_latency_ms": 5.0}
        df = svm_detector["df"]
        version = select.register(
            svm_detector["detector"],
            df.head(3),
            "fjd-test",
            {"model_name": "linear_svm"},
            metrics,
            {},
        )
        client = MlflowClient()
        challenger = candidate(client, "fjd-test", CHALLENGER, "pr_auc")
        assert challenger == Candidate(version, "linear_svm", 0.6, 1.0, 5.0, challenger.val_md5)
        assert candidate(client, "fjd-test", CHAMPION, "pr_auc") is None
        loaded = mlflow.pyfunc.load_model(f"models:/fjd-test@{CHALLENGER}")
        pd.testing.assert_frame_equal(loaded.predict(df), svm_detector["detector"].predict(df))
    finally:
        mlflow.set_tracking_uri(None)
