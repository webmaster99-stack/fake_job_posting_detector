"""API configuration, from environment variables (and the project's `.env` when run locally).

On Render every value comes from the service's environment (see render.yaml). Locally, with no
`DATABASE_URL`, predictions go to a SQLite file, and with no `MLFLOW_TRACKING_URI` the model is
loaded from the local registry (`mlflow.db`) that `dvc repro` writes.
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

from fjd.utils.config import PROJECT_ROOT, load_params

LOCAL_DB = PROJECT_ROOT / "backend" / "local.db"


def _list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def database_url(url: str) -> str:
    """SQLAlchemy URL for the psycopg 3 driver. Neon gives `postgresql://` (or `postgres://`)."""
    for scheme in ("postgresql://", "postgres://"):
        if url.startswith(scheme):
            return "postgresql+psycopg://" + url.removeprefix(scheme)
    return url


@dataclass(frozen=True)
class Settings:
    model_uri: str
    database_url: str
    allowed_origins: list[str] = field(default_factory=list)
    rate_limit_per_minute: int = 30  # per client IP, on /predict and /feedback
    explanation_terms: int = 10

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(PROJECT_ROOT / ".env")  # never overrides variables that are already set
        registered = load_params()["mlflow"]["registered_model"]
        return cls(
            model_uri=os.environ.get("MODEL_URI") or f"models:/{registered}@champion",
            database_url=database_url(
                os.environ.get("DATABASE_URL") or f"sqlite:///{LOCAL_DB.as_posix()}"
            ),
            allowed_origins=_list(os.environ.get("ALLOWED_ORIGINS", "")),
            rate_limit_per_minute=int(os.environ.get("RATE_LIMIT_PER_MINUTE") or 30),
        )
