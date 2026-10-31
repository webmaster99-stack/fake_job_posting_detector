"""Drift report: compare recent live predictions with the training data (`drift_reference`).

`uv run --group monitoring python -m fjd.monitoring.drift [--days N]` reads the predictions of
the last `monitoring.window_days` from `DATABASE_URL` (default: the API's `backend/local.db`),
runs Evidently's data drift preset on the columns in `fjd.monitoring.features`, and writes
`drift_report/` (git-ignored): `report.html`, `summary.json` and `issue.md`. Every run is logged to
MLflow (`monitoring.experiment`). `.github/workflows/drift.yml` runs it weekly and opens an issue
when `summary.json` says `"drift": true`, i.e. when at least `monitoring.drift_share` of the
columns drifted. With fewer than `monitoring.min_rows` predictions there is no report (status
`skipped`): a handful of postings can't show drift.

Needs the `monitoring` dependency group and the reference: `uv run dvc pull drift_reference`.
"""

import argparse
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from dotenv import load_dotenv

from fjd.data.io import read_parquet, write_json
from fjd.monitoring.export import load_predictions
from fjd.monitoring.features import CATEGORICAL, NUMERICAL, monitoring_frame
from fjd.monitoring.reference import OUT as REFERENCE
from fjd.utils.config import PROJECT_ROOT, load_params
from fjd.utils.tracking import run_tags, setup_mlflow

OUT_DIR = PROJECT_ROOT / "drift_report"
LOCAL_DB = PROJECT_ROOT / "backend" / "local.db"  # the API's default without DATABASE_URL


def column_drift(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-column drift from an Evidently snapshot: the test, its score and whether it drifted.
    p-value tests drift below their threshold; distances (Wasserstein, Jensen-Shannon) at or
    above it."""
    columns = []
    for metric in snapshot["metrics"]:
        config = metric["config"]
        if not config["type"].endswith(":ValueDrift"):
            continue
        score, threshold = float(metric["value"]), float(config["threshold"])
        p_value = "p_value" in config["method"]
        drifted = score < threshold if p_value else score >= threshold
        columns.append(
            {"column": config["column"], "method": config["method"], "score": score}
            | {"threshold": threshold, "drifted": drifted}
        )
    return columns


def drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, drift_share: float, html: Path
) -> dict[str, Any]:
    """Run Evidently, save its HTML report and return the drift summary."""
    # Imported here: evidently is only in the `monitoring` group, and is slow to import.
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    definition = DataDefinition(numerical_columns=NUMERICAL, categorical_columns=CATEGORICAL)
    report = Report([DataDriftPreset(drift_share=drift_share)])
    snapshot = report.run(
        Dataset.from_pandas(current, data_definition=definition),
        Dataset.from_pandas(reference, data_definition=definition),
    )
    html.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(html))
    columns = column_drift(snapshot.dict())
    drifted = [c["column"] for c in columns if c["drifted"]]
    share = len(drifted) / len(columns)
    return {"drift": share >= drift_share, "drifted_share": round(share, 4)} | {
        "drifted_columns": drifted,
        "columns": columns,
    }


def issue_body(summary: dict[str, Any]) -> str:
    rows = [
        f"| `{c['column']}` | {c['method']} | {c['score']:.4g} | {c['threshold']:g} |"
        for c in summary["columns"]
        if c["drifted"]
    ]
    return "\n".join(
        [
            f"{len(summary['drifted_columns'])} of {len(summary['columns'])} monitored columns "
            f"drifted ({summary['drifted_share']:.0%}, threshold {summary['drift_share']:.0%}) "
            f"in {summary['rows']} predictions from {summary['since']} to {summary['until']}.",
            "",
            f"Mean fraud probability {summary['mean_probability']:.3f}, flagged as fraud "
            f"{summary['fraud_rate']:.1%}. Model versions: {summary['model_versions']}.",
            "",
            "| Column | Test | Score | Threshold |",
            "| --- | --- | --- | --- |",
            *rows,
            "",
            "The full Evidently report is attached to the workflow run as an artifact.",
        ]
    )


def run(days: int | None = None, out_dir: Path = OUT_DIR) -> dict[str, Any]:
    cfg = load_params()["monitoring"]
    load_dotenv(PROJECT_ROOT / ".env")
    url = os.environ.get("DATABASE_URL") or f"sqlite:///{LOCAL_DB.as_posix()}"
    until = datetime.now(UTC)
    since = until - timedelta(days=days or cfg["window_days"])
    current = load_predictions(url, since, until, exclude=cfg["exclude_predictions"])

    summary: dict[str, Any] = {
        "since": since.isoformat(timespec="seconds"),
        "until": until.isoformat(timespec="seconds"),
        "rows": len(current),
        "drift_share": cfg["drift_share"],
        "model_versions": {
            str(k): int(v) for k, v in current["model_version"].fillna("?").value_counts().items()
        },
        "mean_probability": float(current["probability"].mean()) if len(current) else None,
        "fraud_rate": float(current["fraud"].mean()) if len(current) else None,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("report.html", "issue.md"):  # from an earlier run
        (out_dir / stale).unlink(missing_ok=True)
    if len(current) < cfg["min_rows"]:
        summary |= {"status": "skipped", "drift": False}
    else:
        reference = read_parquet(REFERENCE)
        result = drift_report(
            reference, monitoring_frame(current), cfg["drift_share"], out_dir / "report.html"
        )
        summary |= {"status": "drift" if result["drift"] else "ok"} | result
        (out_dir / "issue.md").write_text(issue_body(summary) + "\n", encoding="utf-8")
    write_json(summary, out_dir / "summary.json")

    setup_mlflow(cfg["experiment"])
    with mlflow.start_run(run_name="drift", tags=run_tags("drift") | {"status": summary["status"]}):
        mlflow.log_params({"since": summary["since"], "until": summary["until"]})
        mlflow.log_metric("rows", summary["rows"])
        for key in ("mean_probability", "fraud_rate", "drifted_share"):
            if summary.get(key) is not None:
                mlflow.log_metric(key, summary[key])
        mlflow.log_artifacts(str(out_dir), "drift_report")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--days", type=int, help="window length (default: monitoring.window_days)")
    args = parser.parse_args()
    summary = run(args.days)
    detail = {k: summary[k] for k in ("status", "rows", "since", "until")}
    if "drifted_share" in summary:
        detail["drifted_share"] = summary["drifted_share"]
    print(json.dumps(detail))


if __name__ == "__main__":
    main()
