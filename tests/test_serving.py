import re

import numpy as np
import pandas as pd
import pytest

from fjd.data.schema import BINARY_COLUMNS
from fjd.features.build import INPUT_COLUMNS, PLACEHOLDER, combine_text, tabular_frame
from fjd.models import explain
from fjd.serving import anonymize, prepare
from tests.test_models import make_postings
from tests.test_registry import fit_pipeline


@pytest.mark.parametrize(
    ("text", "kinds"),
    [
        ("apply at https://jobs.example.com/apply?id=3 today", ["URL"]),
        ("see www.example.com", ["URL"]),
        ("mail hr@example.co.uk now", ["EMAIL"]),
        ("call +1 (555) 123-4567 or 555.123.4567", ["PHONE", "PHONE"]),
        ("salary 40,000 in 2019-2020, ref 12345", []),  # too few digits for a phone
        ("already anonymized #URL_abc123#", ["URL"]),
    ],
)
def test_anonymize(text: str, kinds: list[str]) -> None:
    out = anonymize(text)
    assert sorted(re.findall(PLACEHOLDER, out)) == sorted(kinds)


def test_anonymize_feeds_the_placeholder_features() -> None:
    df = prepare(pd.DataFrame([{"title": "x", "description": "email me: a.b@c.com"}]))
    assert tabular_frame(df)["has_email"].iloc[0] == 1
    assert "xxEMAIL" in combine_text(df).iloc[0]


def test_prepare_matches_the_clean_stage() -> None:
    raw = {"title": "  Data   engineer ", "benefits": "   ", "location": "US,  NY"}
    df = prepare(pd.DataFrame([raw | {"has_questions": 1}]))
    assert list(df.columns) == INPUT_COLUMNS
    row = df.iloc[0]
    assert row["title"] == "Data engineer" and row["location"] == "US, NY"
    assert row["benefits"] is None and row["description"] is None  # blank and absent: missing
    assert row[BINARY_COLUMNS].tolist() == [0, 0, 1]


def test_prepare_leaves_training_rows_unchanged() -> None:
    df = make_postings()[INPUT_COLUMNS]
    out = prepare(df)
    pd.testing.assert_frame_equal(out.isna(), df.isna())
    pd.testing.assert_frame_equal(out.fillna(""), df.fillna(""), check_dtype=False)


@pytest.mark.parametrize(
    ("name", "hp"),
    [
        ("logreg", {}),
        ("linear_svm", {}),
        ("naive_bayes", {}),
        ("xgboost", {"n_estimators": 10, "k_features": 20}),
        ("lightgbm", {"n_estimators": 10, "k_features": 20, "min_child_samples": 2}),
    ],
)
def test_contributions_add_up_to_the_raw_score(name: str, hp: dict) -> None:
    df = make_postings()
    pipeline = fit_pipeline(name, df, **hp)
    values, names = explain.contributions(pipeline, df)
    assert values.shape == (len(df), len(names))
    model = pipeline.named_steps["model"]
    X = pipeline.named_steps["features"].transform(df)
    if name == "naive_bayes":
        jll = model.predict_joint_log_proba(X)
        raw = jll[:, 1] - jll[:, 0]
    elif name in ("logreg", "linear_svm"):
        raw = pipeline.decision_function(df)
    else:  # boosters: log-odds
        p = pipeline.predict_proba(df)[:, 1]
        raw = np.log(p / (1 - p))
    residual = raw - values.sum(axis=1)  # the intercept / base value
    np.testing.assert_allclose(residual, residual[0], atol=1e-4)


def test_explain_names_and_ranks_features() -> None:
    df = make_postings()
    pipeline = fit_pipeline("logreg", df)
    out = explain.explain(pipeline, df.head(1), top_k=5)
    assert 0 < len(out) <= 5
    assert all(c.kind in {"text", "tabular", "categorical"} for c in out)
    assert abs(out[0].value) == max(abs(c.value) for c in out)
    assert all("=" in c.feature for c in out if c.kind == "categorical")


def test_models_without_an_explainer() -> None:
    pipeline = fit_pipeline("random_forest", make_postings(), n_estimators=5, k_features=10)
    assert explain.method(pipeline) is None
    assert explain.explain(pipeline, make_postings().head(1)) == []
