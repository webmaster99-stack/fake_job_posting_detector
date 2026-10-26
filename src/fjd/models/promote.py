"""Promote the registered model's `@challenger` to `@champion`, after a manual approval.

`select_and_register` (run by `dvc repro`) registers each new challenger; the API serves
`@champion`. Run this by hand to promote: `uv run poe promote` (`--yes` skips the prompt, for a
workflow that has its own approval step). The challenger must be within the deployment limits
(`params.yaml: selection`) and beat the champion on val `selection.metric`, as logged by their
`select_and_register` runs. The first challenger only has to be within the limits.

Both are compared on their own val split. If the data changed between them (a different val
md5), the comparison is not like for like, so the script warns before asking.
"""

import argparse
import sys
from dataclasses import dataclass
from typing import Any

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from fjd.models.registry import CHALLENGER, CHAMPION
from fjd.utils.config import load_params
from fjd.utils.tracking import setup_mlflow


@dataclass(frozen=True)
class Candidate:
    version: str
    model_name: str
    score: float  # val `selection.metric`
    size_mb: float
    p95_latency_ms: float
    val_md5: str


def candidate(
    client: MlflowClient, registered_model: str, alias: str, metric: str
) -> Candidate | None:
    """The version behind `alias`, with its `select_and_register` run's results; None if unset."""
    try:
        mv = client.get_model_version_by_alias(registered_model, alias)
    except MlflowException:
        return None
    if mv.run_id is None:
        raise RuntimeError(f"v{mv.version} was not registered by `select_and_register`")
    run = client.get_run(mv.run_id)
    m = run.data.metrics
    return Candidate(
        version=str(mv.version),
        model_name=run.data.params["model_name"],
        score=m[f"val_{metric}"],
        size_mb=m["size_mb"],
        p95_latency_ms=m["p95_latency_ms"],
        val_md5=run.data.tags.get("data_md5.val", ""),
    )


def refusals(
    challenger: Candidate, champion: Candidate | None, limits: dict[str, Any]
) -> list[str]:
    """Why the challenger can't be promoted; empty if it can."""
    out = []
    if challenger.size_mb > limits["max_model_size_mb"]:
        out.append(f"size {challenger.size_mb} MB > {limits['max_model_size_mb']} MB")
    if challenger.p95_latency_ms > limits["max_p95_latency_ms"]:
        limit = limits["max_p95_latency_ms"]
        out.append(f"p95 latency {challenger.p95_latency_ms} ms > {limit} ms")
    if champion is not None and challenger.score <= champion.score:
        out.append(
            f"val {limits['metric']} {challenger.score:.4f} does not beat the champion's "
            f"{champion.score:.4f}"
        )
    return out


def describe(label: str, c: Candidate | None, metric: str) -> str:
    if c is None:
        return f"{label:<11} none"
    return (
        f"{label:<11} v{c.version} {c.model_name}: val {metric} {c.score:.4f}, "
        f"{c.size_mb} MB, p95 {c.p95_latency_ms} ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--yes", action="store_true", help="promote without asking")
    args = parser.parse_args()

    params = load_params()
    limits, registered = params["selection"], params["mlflow"]["registered_model"]
    metric = limits["metric"]
    print("MLflow:", setup_mlflow())
    client = MlflowClient()
    challenger = candidate(client, registered, CHALLENGER, metric)
    if challenger is None:
        sys.exit(f"{registered} has no @{CHALLENGER}; run `dvc repro` first")
    champion = candidate(client, registered, CHAMPION, metric)
    print(describe(f"@{CHALLENGER}", challenger, metric))
    print(describe(f"@{CHAMPION}", champion, metric))

    if champion is not None and champion.version == challenger.version:
        print(f"v{challenger.version} is already @{CHAMPION}")
        return
    problems = refusals(challenger, champion, limits)
    if problems:
        sys.exit("not promoting: " + "; ".join(problems))
    if champion is not None and champion.val_md5 != challenger.val_md5:
        print("warning: the val split changed since the champion was registered")
    answer = "y" if args.yes else input(f"Promote v{challenger.version} to @{CHAMPION}? [y/N] ")
    if answer.strip().lower() != "y":
        sys.exit("not promoted")
    client.set_registered_model_alias(registered, CHAMPION, challenger.version)
    print(f"{registered} v{challenger.version} is now @{CHAMPION}")


if __name__ == "__main__":
    main()
