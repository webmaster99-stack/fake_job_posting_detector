"""FastAPI app: fraud predictions with explanations, user feedback, health and model info.

Run locally: `uv run uvicorn backend.main:create_app --factory --reload`
(interactive docs at http://localhost:8000/docs). See backend/settings.py for configuration.

The model is loaded once at startup, so the first request after a cold start doesn't pay for it.
Predictions are scored one posting at a time, as `select_and_register` measured them.
"""

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from importlib.metadata import version
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend import db
from backend.model import ServedModel, load
from backend.ratelimit import RateLimit
from backend.schemas import (
    FeedbackIn,
    FeedbackResult,
    Health,
    ModelInfo,
    Posting,
    PredictionResult,
    Signal,
)
from backend.settings import Settings
from fjd.models.explain import explain
from fjd.serving import prepare

log = logging.getLogger(__name__)


def _session(request: Request) -> Iterator[Session]:
    with request.app.state.sessions() as session:
        yield session


def _model(request: Request) -> ServedModel:
    model: ServedModel = request.app.state.model
    return model


SessionDep = Annotated[Session, Depends(_session)]
ModelDep = Annotated[ServedModel, Depends(_model)]


def _store(session: Session, row: db.Prediction) -> uuid.UUID | None:
    """Save a prediction; None if the database is unavailable (the user still gets a verdict)."""
    try:
        session.add(row)
        session.commit()
    except SQLAlchemyError:
        session.rollback()
        log.exception("could not store the prediction")
        return None
    return row.id


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    logging.basicConfig(level=logging.INFO)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.model = model = load(settings.model_uri)
        log.info("serving %s (%s) from %s", model.name, model.version, model.uri)
        # Fill lazy caches (feature names, first-call code paths) before the first real request.
        warm_up = prepare(pd.DataFrame([{"title": "warm-up"}]))
        model.detector.predict(warm_up)
        explain(model.detector.classifier, warm_up)
        engine = db.make_engine(settings.database_url)
        app.state.sessions = db.session_factory(engine)
        yield
        engine.dispose()

    app = FastAPI(title="Fake job posting detector", version=version("fjd"), lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    limited = [Depends(RateLimit(settings.rate_limit_per_minute))]

    @app.exception_handler(SQLAlchemyError)
    async def database_unavailable(request: Request, exc: SQLAlchemyError) -> JSONResponse:
        log.error("database error: %s", exc)
        return JSONResponse({"detail": "database unavailable"}, status.HTTP_503_SERVICE_UNAVAILABLE)

    @app.get("/health")
    def health() -> Health:
        """Answers once the model is loaded; the frontend pings it to wake the API up."""
        return Health(status="ok")

    @app.get("/model-info")
    def model_info(model: ModelDep) -> ModelInfo:
        return ModelInfo(
            model_name=model.name,
            registered_model=model.registered_model,
            version=model.version,
            alias=model.alias,
            threshold=model.detector.config.threshold_probability,
            explanation_method=model.explanation_method,
            metrics=model.metrics,
        )

    @app.post("/predict", dependencies=limited)
    def predict(posting: Posting, model: ModelDep, session: SessionDep) -> PredictionResult:
        start = time.perf_counter()
        df = prepare(pd.DataFrame([posting.model_dump()]))
        result = model.detector.predict(df).iloc[0]
        signals = [
            Signal(kind=c.kind, feature=c.feature, contribution=c.value)
            for c in explain(model.detector.classifier, df, settings.explanation_terms)
        ]
        row = db.Prediction(
            model_name=model.name,
            model_version=model.version,
            posting=json.loads(df.to_json(orient="records"))[0],
            probability=float(result["probability"]),
            fraud=bool(result["fraud"]),
            threshold=float(result["threshold"]),
            latency_ms=(time.perf_counter() - start) * 1000,
        )
        return PredictionResult(
            prediction_id=_store(session, row),
            fraud=row.fraud,
            probability=row.probability,
            threshold=row.threshold,
            model_name=model.name,
            model_version=model.version,
            explanation=signals,
        )

    @app.post("/feedback", dependencies=limited)
    def feedback(body: FeedbackIn, session: SessionDep) -> FeedbackResult:
        if session.get(db.Prediction, body.prediction_id) is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown prediction_id")
        row = session.scalar(
            select(db.Feedback).where(db.Feedback.prediction_id == body.prediction_id)
        )
        if row is None:
            row = db.Feedback(prediction_id=body.prediction_id)
            session.add(row)
        row.correct, row.comment = body.correct, body.comment
        session.commit()
        return FeedbackResult(prediction_id=body.prediction_id, correct=body.correct)

    return app
