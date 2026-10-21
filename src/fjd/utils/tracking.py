"""MLflow setup shared by pipeline stages and notebooks.

Runs go to DagsHub when MLFLOW_TRACKING_URI is set (maintainer and CI, via `.env`). Otherwise
they go to a local store in the project root (`mlflow.db` + `mlartifacts/`, both git-ignored),
so anyone can run `dvc repro` without write access to the shared tracking server.
"""

import os
import subprocess
from typing import Any

import mlflow
import yaml
from dotenv import load_dotenv

from fjd.utils.config import PROJECT_ROOT, load_params

LOCAL_DB = PROJECT_ROOT / "mlflow.db"
LOCAL_ARTIFACTS = PROJECT_ROOT / "mlartifacts"
DVC_LOCK = PROJECT_ROOT / "dvc.lock"


def setup_mlflow(experiment: str | None = None) -> str:
    """Point MLflow at the right server and experiment. Returns the tracking URI."""
    load_dotenv(PROJECT_ROOT / ".env")
    remote = os.environ.get("MLFLOW_TRACKING_URI")
    uri = remote or f"sqlite:///{LOCAL_DB.as_posix()}"
    mlflow.set_tracking_uri(uri)
    name = experiment or load_params()["mlflow"]["experiment"]
    if mlflow.get_experiment_by_name(name) is None:
        location = None if remote else LOCAL_ARTIFACTS.as_uri()
        mlflow.create_experiment(name, artifact_location=location)
    mlflow.set_experiment(name)
    return uri


def _git(*args: str) -> str:
    """Output of a git command, or "" if git or the repo is unavailable."""
    try:
        result = subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        return ""
    return result.stdout.strip()


def data_hashes() -> dict[str, str]:
    """md5 of each processed split, as recorded in dvc.lock by the `split` stage."""
    lock = yaml.safe_load(DVC_LOCK.read_text(encoding="utf-8"))
    outs = lock["stages"]["split"]["outs"]
    return {out["path"].rsplit("/", 1)[-1].split(".")[0]: out["md5"] for out in outs}


def run_tags(source: str) -> dict[str, str]:
    """Tags every run carries: where it came from, the code version and the data version."""
    return {
        "source": source,  # "pipeline" or a notebook name
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": str(bool(_git("status", "--porcelain"))).lower(),
        **{f"data_md5.{split}": md5 for split, md5 in data_hashes().items()},
    }


def flatten(params: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """{"a": {"b": 1}} -> {"a.b": 1}, for mlflow.log_params."""
    out: dict[str, Any] = {}
    for key, value in params.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out |= flatten(value, f"{name}.")
        else:
            out[name] = value
    return out
