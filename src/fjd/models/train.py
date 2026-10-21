"""Stage `train_classical@<name>`: fit one classical model on the train split.

Saves the full pipeline (the fitted featurizer from `featurize` + the model), so it predicts from
raw postings, plus the decision threshold that maximizes F1 on the val split, and reports the
val metrics at that threshold.

The threshold is a model artifact, not a `dvc metrics` value: its last digits differ between
Windows and Linux (float rounding in liblinear), while the rounded metrics are identical.
"""

import sys
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

from fjd.data.io import REPORTS, read_split, write_json
from fjd.data.schema import TARGET
from fjd.models.classical import build_model
from fjd.models.metrics import classification_metrics, fraud_scores, tune_threshold
from fjd.utils.config import PROJECT_ROOT, load_params
from fjd.utils.seed import set_seed

FEATURIZER = PROJECT_ROOT / "models" / "featurizer.joblib"
MODELS_DIR = PROJECT_ROOT / "models" / "classical"
REPORTS_DIR = REPORTS / "train"


def train(
    name: str, hyperparams: dict[str, Any], seed: int
) -> tuple[Pipeline, float, dict[str, Any]]:
    """Return the fitted pipeline, its val-tuned threshold and its val metrics."""
    featurizer = joblib.load(FEATURIZER)
    train_df, val_df = read_split("train"), read_split("val")
    model = build_model(name, hyperparams, seed).fit(
        featurizer.transform(train_df), train_df[TARGET]
    )
    # Steps are already fitted; the pipeline only chains them.
    pipeline = Pipeline([("features", featurizer), ("model", model)])
    scores = fraud_scores(pipeline, val_df)
    threshold = tune_threshold(val_df[TARGET], scores)
    return pipeline, threshold, classification_metrics(val_df[TARGET], scores, threshold)


def threshold_path(name: str) -> Path:
    return MODELS_DIR / f"{name}.threshold.json"


def main() -> None:
    name = sys.argv[1]
    params = load_params()
    set_seed(params["seed"])
    pipeline, threshold, val_metrics = train(name, params["classical"][name], params["seed"])
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODELS_DIR / f"{name}.joblib", compress=3)
    write_json({"threshold": threshold}, threshold_path(name))
    write_json({"val": val_metrics}, REPORTS_DIR / f"{name}.json")


if __name__ == "__main__":
    main()
