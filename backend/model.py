"""Load the served model: by default `models:/fake-job-detector@champion` from the registry.

The alias is resolved to a version once, at startup, and that exact version is loaded, so
`/model-info` always describes the model that answers `/predict`. Moving `@champion` takes
effect on the next restart (a Render deploy).
"""

import os
import re
from dataclasses import dataclass, field

import mlflow
from mlflow import MlflowClient

from fjd.models import explain
from fjd.models.registry import Detector
from fjd.utils.tracking import LOCAL_DB

ALIAS_URI = re.compile(r"^models:/(?P<name>[^/@]+)@(?P<alias>[^/@]+)$")
# Metrics of the `select_and_register` run reported by /model-info.
REPORTED_METRICS = ["pr_auc", "f1", "precision", "recall"]


@dataclass(frozen=True)
class ServedModel:
    detector: Detector
    uri: str  # what was loaded; an alias is resolved to `models:/<name>/<version>`
    registered_model: str | None = None
    version: str | None = None
    alias: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.detector.config.model_name

    @property
    def explanation_method(self) -> str | None:
        return explain.method(self.detector.classifier)


def _detector(uri: str) -> Detector:
    model = mlflow.pyfunc.load_model(uri).unwrap_python_model()
    detector: Detector = model.detector  # type: ignore[attr-defined]
    return detector


def load(uri: str) -> ServedModel:
    """Load `uri`: a registry alias (`models:/<name>@<alias>`) or any URI MLflow can load."""
    # Same default as the pipeline: the local registry when no tracking server is configured.
    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{LOCAL_DB.as_posix()}"
    )
    match = ALIAS_URI.match(uri)
    if match is None:
        return ServedModel(_detector(uri), uri)
    name, alias = match["name"], match["alias"]
    client = MlflowClient()
    version = client.get_model_version_by_alias(name, alias)
    resolved = f"models:/{name}/{version.version}"
    metrics = client.get_run(version.run_id).data.metrics if version.run_id else {}
    reported = {
        f"{split}_{m}": metrics[f"{split}_{m}"]
        for split in ("val", "test")
        for m in REPORTED_METRICS
        if f"{split}_{m}" in metrics
    }
    return ServedModel(_detector(resolved), resolved, name, str(version.version), alias, reported)
