from collections.abc import Iterator
from pathlib import Path
from typing import Any

import mlflow
import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from backend import db
from backend.main import create_app
from backend.schemas import Posting
from backend.settings import Settings, database_url
from fjd.features.build import INPUT_COLUMNS
from fjd.models.registry import CONFIG_ARTIFACT, MODEL_ARTIFACT, FraudDetectorModel
from fjd.utils.config import PROJECT_ROOT

ORIGIN = "https://fjd.example.com"
POSTING = {
    "title": "Data entry clerk",
    "description": "Earn money fast from home, call +1 555 123 4567 now",
    "location": "US, TX, Houston",
    "telecommuting": True,
    "has_company_logo": False,
    "has_questions": False,
}


def migrate(url: str) -> None:
    config = Config(PROJECT_ROOT / "backend" / "alembic.ini")
    config.attributes.update(url=url, configure_logger=False)
    command.upgrade(config, "head")


@pytest.fixture(scope="module")
def model_uri(svm_detector: dict[str, Any], tmp_path_factory: pytest.TempPathFactory) -> str:
    """The test LinearSVC detector saved as the pyfunc that `select_and_register` registers."""
    tmp = tmp_path_factory.mktemp("pyfunc")
    config_path = tmp / "detector.json"
    config_path.write_text(svm_detector["detector"].config.to_json(), encoding="utf-8")
    mlflow.pyfunc.save_model(
        tmp / "model",
        python_model=FraudDetectorModel(),
        artifacts={CONFIG_ARTIFACT: str(config_path), MODEL_ARTIFACT: str(svm_detector["path"])},
    )
    return str(tmp / "model")


@pytest.fixture
def database(tmp_path: Path) -> str:
    url = f"sqlite:///{(tmp_path / 'api.db').as_posix()}"
    migrate(url)
    return url


def make_client(model_uri: str, database: str, **settings: Any) -> TestClient:
    s = Settings(model_uri=model_uri, database_url=database, allowed_origins=[ORIGIN], **settings)
    return TestClient(create_app(s))


@pytest.fixture
def client(model_uri: str, database: str) -> Iterator[TestClient]:
    with make_client(model_uri, database) as c:
        yield c


def rows(database: str, table: type[db.Base]) -> list[Any]:
    with Session(create_engine(database)) as session:
        return list(session.scalars(select(table)))


def test_migrations_match_the_models(database: str) -> None:
    with create_engine(database).connect() as connection:
        diff = compare_metadata(MigrationContext.configure(connection), db.Base.metadata)
    assert diff == []


def test_posting_fields_are_the_model_inputs() -> None:
    assert set(Posting.model_fields) == set(INPUT_COLUMNS)


def test_database_url_uses_psycopg() -> None:
    assert database_url("postgres://u:p@h/db?sslmode=require") == (
        "postgresql+psycopg://u:p@h/db?sslmode=require"
    )
    assert database_url("postgresql://h/db") == "postgresql+psycopg://h/db"
    assert database_url("sqlite:///x.db") == "sqlite:///x.db"


def test_health_and_model_info(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}
    info = client.get("/model-info").json()
    assert info["model_name"] == "linear_svm" and info["explanation_method"] == "linear"
    assert info["version"] is None  # loaded from a path, not the registry
    assert 0 < info["threshold"] < 1


def test_predict_explains_and_logs(client: TestClient, database: str) -> None:
    response = client.post("/predict", json=POSTING)
    assert response.status_code == 200
    out = response.json()
    assert 0 <= out["probability"] <= 1
    assert out["fraud"] == (out["probability"] >= out["threshold"])
    assert out["explanation"] and {s["kind"] for s in out["explanation"]} <= {
        "text",
        "tabular",
        "categorical",
    }
    [row] = rows(database, db.Prediction)
    assert str(row.id) == out["prediction_id"] and row.model_name == "linear_svm"
    # Stored as the model saw it: the phone number is a placeholder, missing fields are null.
    assert "#PHONE_" in row.posting["description"] and "555" not in row.posting["description"]
    assert row.posting["benefits"] is None and row.posting["telecommuting"] == 1


def test_predict_validates_the_posting(client: TestClient) -> None:
    flags = {k: POSTING[k] for k in ("telecommuting", "has_company_logo", "has_questions")}
    assert client.post("/predict", json={"title": "  "} | flags).status_code == 422
    assert client.post("/predict", json={"title": "x"}).status_code == 422  # flags required
    assert client.post("/predict", json=POSTING | {"salary": "1"}).status_code == 422
    assert client.post("/predict", json=POSTING | {"benefits": "x" * 20_001}).status_code == 422


def test_predict_without_a_database_still_answers(model_uri: str, tmp_path: Path) -> None:
    # No tables were created, so the insert fails.
    with make_client(model_uri, f"sqlite:///{(tmp_path / 'empty.db').as_posix()}") as c:
        response = c.post("/predict", json=POSTING)
    assert response.status_code == 200 and response.json()["prediction_id"] is None


def test_feedback_is_stored_once_per_prediction(client: TestClient, database: str) -> None:
    prediction_id = client.post("/predict", json=POSTING).json()["prediction_id"]
    first = client.post("/feedback", json={"prediction_id": prediction_id, "correct": True})
    assert first.status_code == 200
    again = {"prediction_id": prediction_id, "correct": False, "comment": "real company"}
    assert client.post("/feedback", json=again).json()["correct"] is False
    [row] = rows(database, db.Feedback)
    assert str(row.prediction_id) == prediction_id
    assert row.correct is False and row.comment == "real company"


def test_feedback_for_an_unknown_prediction(client: TestClient) -> None:
    body = {"prediction_id": "00000000-0000-0000-0000-000000000000", "correct": True}
    assert client.post("/feedback", json=body).status_code == 404


def test_cors_allows_only_the_configured_origin(client: TestClient) -> None:
    def preflight(origin: str) -> Any:
        headers = {"Origin": origin, "Access-Control-Request-Method": "POST"}
        return client.options("/predict", headers=headers).headers

    assert preflight(ORIGIN)["access-control-allow-origin"] == ORIGIN
    assert "access-control-allow-origin" not in preflight("https://evil.example.com")


def test_rate_limit(model_uri: str, database: str) -> None:
    with make_client(model_uri, database, rate_limit_per_minute=2) as c:
        codes = [c.post("/predict", json=POSTING).status_code for _ in range(3)]
        assert codes == [200, 200, 429]
        assert c.get("/health").status_code == 200  # not limited
