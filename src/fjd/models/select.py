"""Stage `select_and_register`: pick the model to serve and register it as `@challenger`.

Candidates are ranked by their val `selection.metric` in `evaluate`'s reports/metrics.json. The
best one within the deployment limits (`params.yaml: selection`) wins. Deployability is measured
here, on this machine, the way the API scores: one posting at a time on `SERVING_THREADS`.

The winner is wrapped with its threshold and a Platt calibration fitted on val
(`fjd.models.registry`), logged to MLflow, registered as a new version of
`mlflow.registered_model` and given the alias `@challenger`. Promotion to `@champion` is a
separate, manual step (`fjd.models.promote`).

reports/selection.json records the choice, which reproduces exactly. The calibration and
latencies don't (last float digits, machine speed), so they are only logged to MLflow.
"""

import json
from collections.abc import Callable
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.metrics import brier_score_loss, log_loss

from fjd.data.io import REPORTS, read_split, write_json
from fjd.data.schema import TARGET
from fjd.features.build import INPUT_COLUMNS
from fjd.models import deployability
from fjd.models.metrics import fraud_scores
from fjd.models.registry import (
    CHALLENGER,
    CONFIG_ARTIFACT,
    MODEL_ARTIFACT,
    TRANSFORMER,
    Calibration,
    Detector,
    DetectorConfig,
    FraudDetectorModel,
    load_classifier,
    requirements,
)
from fjd.models.train import MODELS_DIR, threshold_path
from fjd.models.transformer import MODEL_DIR as TRANSFORMER_DIR
from fjd.models.transformer import load_threshold
from fjd.utils.config import PROJECT_ROOT, load_params
from fjd.utils.seed import set_seed
from fjd.utils.tracking import flatten, run_tags, setup_mlflow

METRICS = REPORTS / "metrics.json"
OUT = REPORTS / "selection.json"
PACKAGE_DIR = PROJECT_ROOT / "src" / "fjd"


def model_path(name: str) -> Path:
    return TRANSFORMER_DIR if name == TRANSFORMER else MODELS_DIR / f"{name}.joblib"


def threshold(name: str) -> float:
    if name == TRANSFORMER:
        return load_threshold()
    return float(json.loads(threshold_path(name).read_text(encoding="utf-8"))["threshold"])


def ranking(metrics: dict[str, Any], metric: str) -> list[str]:
    """Model names, best val `metric` first (ties broken by name, so the order is stable)."""
    return sorted(metrics, key=lambda name: (-metrics[name]["val"][metric], name))


def select(
    ranked: list[str], check: Callable[[str], dict[str, Any]]
) -> tuple[str, dict[str, dict[str, Any]]]:
    """The first deployable model in `ranked`, and the deployability of every model checked."""
    checked = {}
    for name in ranked:
        checked[name] = check(name)
        if checked[name]["deployable"]:
            return name, checked
    raise RuntimeError(f"no model is within the deployment limits: {checked}")


def check_deployability(
    name: str, postings: pd.DataFrame, limits: dict[str, Any]
) -> dict[str, Any]:
    classifier = load_classifier(name, model_path(name))  # as served: SERVING_THREADS
    return deployability.check(
        model_path(name), partial(fraud_scores, classifier), postings[INPUT_COLUMNS], limits
    )


def calibration_metrics(y: Any, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "brier": float(brier_score_loss(y, probabilities)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
    }


def register(
    detector: Detector,
    postings: pd.DataFrame,
    registered_model: str,
    run_params: dict[str, Any],
    run_metrics: dict[str, float],
    report: dict[str, Any],
) -> str:
    """Log `detector` as a pyfunc, register it and alias it `@challenger`. Returns the version.

    The input schema is inferred from all of `postings`, so that columns which are sometimes
    missing get their real type; its first rows are the input example.
    """
    config = detector.config
    example = postings.head(3)
    with (
        TemporaryDirectory() as tmp,
        mlflow.start_run(
            run_name="select_and_register",
            tags=run_tags("pipeline") | {"model": config.model_name},
        ),
    ):
        config_path = Path(tmp) / "detector.json"
        config_path.write_text(config.to_json(), encoding="utf-8", newline="\n")
        mlflow.log_params(run_params)
        mlflow.log_metrics(run_metrics)
        mlflow.log_dict(report, "selection.json")
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=FraudDetectorModel(),
            artifacts={
                CONFIG_ARTIFACT: str(config_path),
                MODEL_ARTIFACT: str(model_path(config.model_name)),
            },
            code_paths=[str(PACKAGE_DIR)],
            signature=infer_signature(postings, detector.predict(example)),
            input_example=example,
            pip_requirements=requirements(),
            registered_model_name=registered_model,
        )
    version = str(info.registered_model_version)
    client = MlflowClient()
    client.set_registered_model_alias(registered_model, CHALLENGER, version)
    client.set_model_version_tag(registered_model, version, "model_name", config.model_name)
    return version


def main() -> None:
    params = load_params()
    cfg = params["selection"]
    set_seed(params["seed"])
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    val, test = read_split("val"), read_split("test")

    sample = val.sample(min(cfg["latency_sample"], len(val)), random_state=params["seed"])
    ranked = ranking(metrics, cfg["metric"])
    name, checked = select(ranked, partial(check_deployability, postings=sample, limits=cfg))
    write_json({"model": name, "ranking": ranked} | metrics[name], OUT)

    # All threads for scoring the splits; the registered model serves on SERVING_THREADS.
    scorer = load_classifier(name, model_path(name), threads=None)
    splits = {"val": val, "test": test}
    scores = {split: fraud_scores(scorer, df[INPUT_COLUMNS]) for split, df in splits.items()}
    calibration = Calibration.fit(val[TARGET], scores["val"])
    config = DetectorConfig(name, threshold(name), calibration)
    run_metrics = {
        "threshold": config.threshold,
        "threshold_probability": config.threshold_probability,
        "calibration_slope": calibration.slope,
        "calibration_intercept": calibration.intercept,
        **{f"{s}_{k}": v for s in splits for k, v in metrics[name][s].items()},
        **{
            f"{s}_{k}": v
            for s, df in splits.items()
            for k, v in calibration_metrics(df[TARGET], calibration(scores[s])).items()
        },
        **{k: float(v) for k, v in checked[name].items()},
    }
    run_params = {"model_name": name, "seed": params["seed"]} | flatten({"selection": cfg})
    setup_mlflow()
    version = register(
        Detector(config, load_classifier(name, model_path(name))),
        val[INPUT_COLUMNS],
        params["mlflow"]["registered_model"],
        run_params,
        run_metrics,
        {"ranking": ranked, "deployability": checked},
    )
    print(f"registered {name} as {params['mlflow']['registered_model']} v{version} @{CHALLENGER}")


if __name__ == "__main__":
    main()
