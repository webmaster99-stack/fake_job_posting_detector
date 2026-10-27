from typing import Any

import joblib
import numpy as np
import pytest

from fjd.data.schema import TARGET
from fjd.models.metrics import fraud_scores
from fjd.models.registry import Calibration, Detector, DetectorConfig, load_classifier
from tests.test_models import make_postings
from tests.test_registry import fit_pipeline


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
