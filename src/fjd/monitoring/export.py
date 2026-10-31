"""Read logged predictions from the API's database (Neon in production) for drift monitoring.

Only reads: the query names the columns it needs, so it works with a read-only role. The tables
are defined in `backend/db.py`; this module doesn't import `backend`, which depends on `fjd`.
"""

import uuid
from collections.abc import Iterable
from datetime import datetime

import pandas as pd
import sqlalchemy as sa

from fjd.features.build import INPUT_COLUMNS

PREDICTIONS = sa.table(
    "predictions",
    sa.column("id", sa.Uuid()),
    sa.column("created_at", sa.DateTime(timezone=True)),
    sa.column("model_version", sa.String()),
    sa.column("posting", sa.JSON()),
    sa.column("probability", sa.Float()),
    sa.column("fraud", sa.Boolean()),
)
META_COLUMNS = ["id", "created_at", "model_version", "probability", "fraud"]


def sqlalchemy_url(url: str) -> str:
    """The psycopg 3 driver for Postgres URLs, as in `backend.settings.database_url`."""
    for scheme in ("postgresql://", "postgres://"):
        if url.startswith(scheme):
            return "postgresql+psycopg://" + url.removeprefix(scheme)
    return url


def load_predictions(
    url: str, since: datetime, until: datetime, exclude: Iterable[str] = ()
) -> pd.DataFrame:
    """Predictions made in [since, until): `META_COLUMNS` plus the posting as the model saw it
    (`INPUT_COLUMNS`). `exclude` lists prediction ids to leave out, e.g. smoke tests."""
    query = (
        sa.select(PREDICTIONS)
        .where(PREDICTIONS.c.created_at >= since, PREDICTIONS.c.created_at < until)
        .order_by(PREDICTIONS.c.created_at)
    )
    engine = sa.create_engine(sqlalchemy_url(url))
    try:
        with engine.connect() as conn:
            rows = conn.execute(query).mappings().all()
    finally:
        engine.dispose()
    skip = {uuid.UUID(str(i)) for i in exclude}
    rows = [r for r in rows if r["id"] not in skip]
    meta = pd.DataFrame([{c: r[c] for c in META_COLUMNS} for r in rows], columns=META_COLUMNS)
    meta["id"] = meta["id"].astype(str)
    postings = pd.DataFrame([r["posting"] for r in rows]).reindex(columns=INPUT_COLUMNS)
    return pd.concat([meta, postings], axis=1)
