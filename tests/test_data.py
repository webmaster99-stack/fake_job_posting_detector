import pandas as pd
import pandera.errors
import pytest

from fjd.data.clean import clean
from fjd.data.groups import template_groups
from fjd.data.schema import COLUMNS, SCHEMA, TARGET, TEXT_COLUMNS
from fjd.data.split import split


def make_df(n: int, fraud_every: int = 5) -> pd.DataFrame:
    rows = []
    for i in range(n):
        row: dict[str, object] = {c: None for c in COLUMNS}
        row.update(
            source="test",
            job_id=i,
            title=f"Job {i}",
            description=f"Description  number {i}",
            telecommuting=0,
            has_company_logo=1,
            has_questions=0,
            fraudulent=int(i % fraud_every == 0),
        )
        rows.append(row)
    df = pd.DataFrame(rows, columns=COLUMNS)
    obj_cols = [c for c in COLUMNS if df[c].isna().all()]
    df[obj_cols] = df[obj_cols].astype("str").where(df[obj_cols].notna())
    return df


def test_schema_accepts_valid_frame() -> None:
    SCHEMA.validate(make_df(10))


def test_schema_rejects_bad_label() -> None:
    df = make_df(10)
    df.loc[0, TARGET] = 2
    with pytest.raises(pandera.errors.SchemaError):
        SCHEMA.validate(df)


def test_clean_drops_whitespace_and_case_duplicates() -> None:
    df = make_df(4)
    df.loc[1, ["title", "description"]] = ["JOB 0", "description number   0"]
    out, stats = clean(df, TEXT_COLUMNS)
    assert stats["duplicates_removed"] == 1
    assert out["job_id"].tolist() == [0, 2, 3]  # keeps the lowest id
    assert out.loc[0, "description"] == "Description number 0"


def test_template_groups_link_shared_description_or_company() -> None:
    df = make_df(5)
    long_a, long_b = "a" * 400, "b" * 60  # long_a exceeds template_chars, so suffixes differ
    df["description"] = [long_a, long_a + " tail", "unique one " * 10, "unique two " * 10, "short"]
    df["company_profile"] = [None, None, long_b, long_b, None]
    groups = template_groups(df, template_chars=300, min_key_chars=50)
    # 0 and 1 share a template prefix, 2 and 3 share a company profile, 4 is alone.
    assert groups[0] == groups[1] and groups[2] == groups[3]
    assert len({groups[0], groups[2], groups[4]}) == 3


def test_split_keeps_groups_together_and_is_deterministic() -> None:
    df = make_df(400)
    df["group_id"] = df["job_id"] // 4  # 100 groups of 4 rows
    a = split(df, test_size=0.15, val_size=0.15, seed=1)
    b = split(df, test_size=0.15, val_size=0.15, seed=1)
    ids = [set(p["job_id"]) for p in a.values()]
    assert sum(map(len, ids)) == 400 and not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    groups = [set(p["group_id"]) for p in a.values()]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    for name in a:
        assert a[name]["job_id"].tolist() == b[name]["job_id"].tolist()
        assert a[name][TARGET].mean() == pytest.approx(0.2, abs=0.05)
