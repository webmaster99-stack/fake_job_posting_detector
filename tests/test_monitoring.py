"""Drift monitoring: features, reading logged predictions, and the Evidently report.

Report tests need the `monitoring` dependency group (`uv sync --group monitoring`) and are
skipped without it.
"""

import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import mlflow
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend import db
from fjd.features.build import INPUT_COLUMNS
from fjd.monitoring import drift
from fjd.monitoring.export import load_predictions
from fjd.monitoring.features import CATEGORICAL, NUMERICAL, monitoring_frame
from fjd.serving import prepare
from fjd.utils.config import load_params
from tests.test_models import make_postings

NOW = datetime.now(UTC)


def store(url: str, postings: pd.DataFrame, ages: list[timedelta]) -> list[str]:
    """Log `postings` as the API does (prepared, as JSON) at NOW - age; returns their ids."""
    engine = create_engine(url)
    records = json.loads(prepare(postings).to_json(orient="records"))
    rows = [
        db.Prediction(
            id=uuid.uuid4(),
            created_at=NOW - age,
            model_name="fake-job-detector",
            model_version="1",
            posting=posting,
            probability=0.1,
            fraud=False,
            threshold=0.5,
            latency_ms=1.0,
        )
        for posting, age in zip(records, ages, strict=True)
    ]
    with Session(engine) as session:
        session.add_all(rows)
        session.commit()
        ids = [str(r.id) for r in rows]
    engine.dispose()
    return ids


@pytest.fixture
def db_url(tmp_path: Path) -> str:
    """An empty API database (the tables from backend/db.py)."""
    url = f"sqlite:///{(tmp_path / 'api.db').as_posix()}"
    engine = create_engine(url)
    db.Base.metadata.create_all(engine)
    engine.dispose()
    return url


def test_monitoring_frame_has_every_drift_column() -> None:
    frame = monitoring_frame(make_postings())
    assert list(frame.columns) == NUMERICAL + CATEGORICAL
    assert not frame.isna().any().any()
    assert all(pd.api.types.is_float_dtype(t) for t in frame[NUMERICAL].dtypes)


def test_monitoring_frame_takes_a_posting_with_only_the_required_fields() -> None:
    posting = pd.DataFrame([{"title": "Clerk", "telecommuting": 1}])
    frame = monitoring_frame(posting)
    assert frame.loc[0, "missing_description"] == 1
    assert frame.loc[0, "has_company_logo"] == 0
    assert frame.loc[0, "country"] == "(missing)"


def test_load_predictions_reads_the_window_without_excluded_ids(db_url: str) -> None:
    postings = make_postings(4)
    ids = store(db_url, postings, [timedelta(days=d) for d in (1, 2, 3, 10)])
    df = load_predictions(db_url, NOW - timedelta(days=7), NOW, exclude=[ids[1]])
    assert df["id"].tolist() == [ids[2], ids[0]]  # oldest first; 10 days old is outside
    assert list(df.columns[-len(INPUT_COLUMNS) :]) == INPUT_COLUMNS
    assert df.loc[0, "title"] == postings.loc[2, "title"]


def test_column_drift_reads_p_values_and_distances() -> None:
    def metric(column: str, method: str, value: float, threshold: float) -> dict[str, object]:
        config = {"type": "evidently:metric_v2:ValueDrift", "column": column}
        return {"config": config | {"method": method, "threshold": threshold}, "value": value}

    snapshot = {
        "metrics": [
            {"config": {"type": "evidently:metric_v2:DriftedColumnsCount"}, "value": {}},
            metric("a", "K-S p_value", 0.01, 0.05),
            metric("b", "chi-square p_value", 0.5, 0.05),
            metric("c", "Wasserstein distance (normed)", 0.2, 0.1),
            metric("d", "Jensen-Shannon distance", 0.05, 0.1),
        ]
    }
    drifted = {c["column"]: c["drifted"] for c in drift.column_drift(snapshot)}
    assert drifted == {"a": True, "b": False, "c": True, "d": False}


@pytest.mark.filterwarnings("ignore::SyntaxWarning", "ignore::DeprecationWarning")
def test_drift_report_flags_postings_unlike_the_reference(tmp_path: Path) -> None:
    pytest.importorskip("evidently")
    reference = monitoring_frame(pd.concat([make_postings()] * 5, ignore_index=True))
    same = drift.drift_report(reference, reference.sample(200, random_state=0), 0.5, tmp_path / "a")
    assert not same["drift"] and same["drifted_columns"] == []

    bare = pd.DataFrame([{"title": "Easy money", "telecommuting": 1}] * 60)
    shifted = drift.drift_report(reference, monitoring_frame(bare), 0.3, tmp_path / "b.html")
    assert shifted["drift"] and "missing_description" in shifted["drifted_columns"]
    assert (tmp_path / "b.html").stat().st_size > 0


@pytest.mark.filterwarnings("ignore::SyntaxWarning", "ignore::DeprecationWarning")
def test_run_writes_a_report_and_logs_it(
    db_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("evidently")
    cfg = load_params()["monitoring"]
    reference = tmp_path / "reference.parquet"
    monitoring_frame(pd.concat([make_postings()] * 5, ignore_index=True)).to_parquet(reference)
    monkeypatch.setattr(drift, "REFERENCE", reference)
    monkeypatch.setenv("DATABASE_URL", db_url)
    tracking = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setattr(drift, "setup_mlflow", lambda name: mlflow.set_experiment(name))
    mlflow.set_tracking_uri(tracking)
    monkeypatch.chdir(tmp_path)  # MLflow's default artifact root is relative
    out = tmp_path / "out"
    try:
        too_few = drift.run(out_dir=out)
        assert too_few["status"] == "skipped" and not (out / "report.html").exists()

        n = cfg["min_rows"]
        store(db_url, make_postings(n), [timedelta(hours=1)] * n)
        summary = drift.run(out_dir=out)
        assert summary["status"] in {"ok", "drift"} and summary["rows"] == n
        assert json.loads((out / "summary.json").read_text("utf-8")) == summary
        assert {p.name for p in out.iterdir()} == {"report.html", "summary.json", "issue.md"}
        runs = mlflow.search_runs(experiment_names=[cfg["experiment"]])
        assert len(runs) == 2 and runs["metrics.rows"].tolist() == [n, 0]
    finally:
        mlflow.set_tracking_uri(None)
