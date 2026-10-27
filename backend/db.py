"""Tables for prediction logging and user feedback (Neon Postgres in production).

The schema is managed by Alembic (`backend/migrations`); the container applies migrations before
starting the API. Change a table here, then add a migration:
`uv run alembic -c backend/alembic.ini revision --autogenerate -m "..."`.

`predictions.posting` holds the posting as the model saw it (after `fjd.serving.prepare`, so
links, emails and phone numbers are already placeholders). Drift monitoring compares it with the
training data, and a later `feedback_export` stage turns feedback into labelled postings.
"""

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, String, Text, create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    model_name: Mapped[str] = mapped_column(String(64))
    model_version: Mapped[str | None] = mapped_column(String(32))
    posting: Mapped[dict[str, Any]] = mapped_column(JSON().with_variant(JSONB(), "postgresql"))
    probability: Mapped[float]
    fraud: Mapped[bool]
    threshold: Mapped[float]
    latency_ms: Mapped[float]


class Feedback(Base):
    """One per prediction: whether the user agrees with the verdict. Sending it again updates it."""

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(primary_key=True)
    prediction_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("predictions.id", ondelete="CASCADE"), unique=True
    )
    correct: Mapped[bool]
    comment: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )


def make_engine(url: str) -> Engine:
    # Neon suspends idle computes and drops their connections; pre-ping replaces dead ones.
    return create_engine(url, pool_pre_ping=True, pool_recycle=300)


def session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False)
