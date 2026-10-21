"""Canonical column layout shared by every dataset after ingestion."""

import pandera.pandas as pa

TARGET = "fraudulent"
GROUP = "group_id"  # added by the `split` stage; see groups.py

TEXT_COLUMNS = ["title", "company_profile", "description", "requirements", "benefits"]
CATEGORICAL_COLUMNS = [
    "location",
    "department",
    "salary_range",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
]
BINARY_COLUMNS = ["telecommuting", "has_company_logo", "has_questions"]
ID_COLUMNS = ["source", "job_id"]

COLUMNS = ID_COLUMNS + TEXT_COLUMNS + CATEGORICAL_COLUMNS + BINARY_COLUMNS + [TARGET]

_REQUIRED = {"source", "job_id", "title"}


def _column(name: str) -> pa.Column:
    if name in BINARY_COLUMNS or name == TARGET:
        return pa.Column(int, pa.Check.isin([0, 1]))
    dtype = int if name == "job_id" else str
    return pa.Column(dtype, nullable=name not in _REQUIRED)


SCHEMA = pa.DataFrameSchema(
    {name: _column(name) for name in COLUMNS},
    unique=ID_COLUMNS,
    strict=True,
    ordered=True,
)
