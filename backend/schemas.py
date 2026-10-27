"""Request and response bodies. A posting has one field per canonical input column."""

import uuid
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from fjd.models.explain import Kind

LongText = Annotated[str | None, Field(max_length=20_000)]
ShortText = Annotated[str | None, Field(max_length=300)]


class Posting(BaseModel):
    """A job posting. Only the title is required: a missing field is itself a signal, so leave
    unknown fields out (or empty) rather than guessing. The three flags are required because the
    model treats a missing logo or missing screening questions as suspicious."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str = Field(min_length=1, max_length=300)
    company_profile: LongText = None
    description: LongText = None
    requirements: LongText = None
    benefits: LongText = None
    location: ShortText = Field(None, description='"<country>, <region>, <city>", e.g. "US, NY"')
    department: ShortText = None
    salary_range: ShortText = None
    employment_type: ShortText = None
    required_experience: ShortText = None
    required_education: ShortText = None
    industry: ShortText = None
    function: ShortText = None
    telecommuting: bool
    has_company_logo: bool
    has_questions: bool


class Signal(BaseModel):
    kind: Kind
    feature: str = Field(description="a term, a tabular feature or <column>=<value>")
    contribution: float = Field(description="effect on the model's raw score; > 0 toward fraud")


class PredictionResult(BaseModel):
    prediction_id: uuid.UUID | None = Field(
        description="for /feedback; null if the prediction could not be stored"
    )
    fraud: bool
    probability: float = Field(description="calibrated P(fraud)")
    threshold: float = Field(description="fraud when probability >= threshold")
    model_name: str
    model_version: str | None
    explanation: list[Signal] = Field(description="largest contributions first")


class FeedbackIn(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    prediction_id: uuid.UUID
    correct: bool = Field(description="whether the verdict was right")
    comment: str | None = Field(None, max_length=2_000)


class FeedbackResult(BaseModel):
    prediction_id: uuid.UUID
    correct: bool


class Health(BaseModel):
    status: Literal["ok"]


class ModelInfo(BaseModel):
    model_name: str
    registered_model: str | None
    version: str | None
    alias: str | None
    threshold: float = Field(description="as a probability")
    explanation_method: str | None
    metrics: dict[str, float] = Field(description="from the model's registration run")
