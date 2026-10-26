"""The registered model `fake-job-detector`: one MLflow pyfunc for any selected model.

It wraps the model (a classical sklearn pipeline or the ONNX transformer, both of which take raw
canonical columns), its val-tuned decision threshold and a Platt calibration fitted on val.
Raw scores are inflated probabilities (balanced class weights) or margins (LinearSVC), so the
calibration maps them to P(fraud) at the val fraud rate. The sigmoid is increasing, so PR-AUC and
the labels at the threshold are exactly those reported by `evaluate`.

`select_and_register` logs it; the API loads `models:/fake-job-detector@champion` and gets the
underlying `Detector` (for explanations) with `unwrap_python_model().detector`.
"""

import json
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from mlflow.pyfunc import PythonModel
from sklearn.linear_model import LogisticRegression

from fjd.features.build import INPUT_COLUMNS
from fjd.models.metrics import fraud_scores
from fjd.models.transformer import TransformerClassifier

TRANSFORMER = "transformer"  # selection name of the transformer; every other name is classical
CONFIG_ARTIFACT = "detector"
MODEL_ARTIFACT = "model"
CHALLENGER, CHAMPION = "challenger", "champion"
SERVING_THREADS = 1  # Render's free tier has a fraction of one CPU
REQUIREMENTS = [
    "mlflow",
    "scikit-learn",
    "pandas",
    "numpy",
    "scipy",
    "xgboost",
    "lightgbm",
    "onnxruntime",
    "tokenizers",
]


@dataclass(frozen=True)
class Calibration:
    """Platt scaling: P(fraud) = sigmoid(slope * score + intercept)."""

    slope: float
    intercept: float

    @classmethod
    def fit(cls, y: Any, scores: np.ndarray) -> "Calibration":
        lr = LogisticRegression(C=np.inf).fit(np.asarray(scores).reshape(-1, 1), y)
        return cls(float(lr.coef_[0, 0]), float(lr.intercept_[0]))

    def __call__(self, scores: np.ndarray) -> np.ndarray:
        z = self.slope * np.asarray(scores, dtype=np.float64) + self.intercept
        return np.asarray(1 / (1 + np.exp(-z)))


@dataclass(frozen=True)
class DetectorConfig:
    model_name: str
    threshold: float  # on the raw score; fraud when score >= threshold
    calibration: Calibration

    @property
    def threshold_probability(self) -> float:
        return float(self.calibration(np.array([self.threshold]))[0])

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, text: str) -> "DetectorConfig":
        data = json.loads(text)
        return cls(data["model_name"], data["threshold"], Calibration(**data["calibration"]))


def load_classifier(model_name: str, path: Path, threads: int | None = SERVING_THREADS) -> Any:
    """The model saved by `train_classical@<model_name>` (a joblib file) or `train_transformer`
    (a directory), with its thread count capped at `threads`."""
    if model_name == TRANSFORMER:
        return TransformerClassifier(path, threads=threads)
    pipeline = joblib.load(path)
    if threads is not None:
        n_jobs = [k for k in pipeline.get_params() if k == "n_jobs" or k.endswith("__n_jobs")]
        pipeline.set_params(**dict.fromkeys(n_jobs, threads))
    return pipeline


class Detector:
    """A loaded model plus its threshold and calibration. Predicts from raw canonical columns."""

    def __init__(self, config: DetectorConfig, classifier: Any) -> None:
        self.config = config
        self.classifier = classifier

    def scores(self, postings: pd.DataFrame) -> np.ndarray:
        # The signature marks nullable fields optional, so a request may leave them out.
        return fraud_scores(self.classifier, postings.reindex(columns=INPUT_COLUMNS))

    def predict(self, postings: pd.DataFrame) -> pd.DataFrame:
        """One row per posting: `probability` (calibrated P(fraud)), `fraud` (0/1, the label at
        the tuned threshold) and `threshold` (the same threshold, as a probability)."""
        scores = self.scores(postings)
        return pd.DataFrame(
            {
                "probability": self.config.calibration(scores),
                "fraud": (scores >= self.config.threshold).astype(np.int64),
                "threshold": np.full(len(scores), self.config.threshold_probability),
            },
            index=postings.index,
        )


class FraudDetectorModel(PythonModel):
    """MLflow wrapper around `Detector`. Artifacts: the config JSON and the model file/dir."""

    def load_context(self, context: Any) -> None:
        config_text = Path(context.artifacts[CONFIG_ARTIFACT]).read_text(encoding="utf-8")
        config = DetectorConfig.from_json(config_text)
        model_path = Path(context.artifacts[MODEL_ARTIFACT])
        self.detector = Detector(config, load_classifier(config.model_name, model_path))

    def predict(self, context: Any, model_input: pd.DataFrame, params: Any = None) -> pd.DataFrame:
        return self.detector.predict(model_input)


def requirements() -> list[str]:
    return [f"{pkg}=={version(pkg)}" for pkg in REQUIREMENTS]
