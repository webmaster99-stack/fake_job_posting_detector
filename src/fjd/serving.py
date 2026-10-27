"""Turn a posting submitted to the API into model input, the way training data was prepared.

EMSCAD's text is already anonymized: links, emails and phone numbers appear as placeholders like
`#URL_<sha256>#`, and the featurizer turns them into `has_url`/`has_email`/`has_phone` features
and `xxurl`-style terms. A real posting contains real links, so they are replaced the same way
here, or the model would never see those signals. Whitespace is normalized like the
`clean_dedupe` stage, which also makes blank fields missing (a strong fraud signal).
"""

import hashlib
import re

import pandas as pd

from fjd.data.clean import normalize_whitespace
from fjd.data.schema import BINARY_COLUMNS, CATEGORICAL_COLUMNS, TEXT_COLUMNS
from fjd.features.build import INPUT_COLUMNS

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
URL = re.compile(r"(?:https?://|www\.)[^\s<>\"']+", re.IGNORECASE)
# A candidate phone number; it only counts with PHONE_DIGITS digits (see `_phone`).
PHONE = re.compile(r"(?<![\w#])\+?\(?\d[\d\s().-]{6,}\d(?!\w)")
PHONE_DIGITS = range(9, 16)


def _placeholder(kind: str, text: str) -> str:
    return f"#{kind}_{hashlib.sha256(text.encode()).hexdigest()}#"


def _phone(match: re.Match[str]) -> str:
    text = match.group()
    digits = sum(c.isdigit() for c in text)
    return _placeholder("PHONE", text) if digits in PHONE_DIGITS else text


def anonymize(text: str) -> str:
    """Replace emails, links and phone numbers with EMSCAD-style placeholders."""
    # Emails first, so their domain isn't taken for a link; phones last, so URL digits are kept.
    text = EMAIL.sub(lambda m: _placeholder("EMAIL", m.group()), text)
    text = URL.sub(lambda m: _placeholder("URL", m.group()), text)
    return PHONE.sub(_phone, text)


def prepare(postings: pd.DataFrame) -> pd.DataFrame:
    """Canonical model input (`INPUT_COLUMNS`) from raw postings. Absent fields become missing,
    and absent binary flags 0."""
    df = postings.reindex(columns=INPUT_COLUMNS).astype(
        {c: object for c in TEXT_COLUMNS + CATEGORICAL_COLUMNS}
    )
    for col in TEXT_COLUMNS + CATEGORICAL_COLUMNS:
        df[col] = normalize_whitespace(df[col].astype("string")).astype(object)
        df[col] = df[col].where(df[col].notna(), None)
    for col in TEXT_COLUMNS:
        df[col] = df[col].map(lambda v: anonymize(v) if isinstance(v, str) else v)
    for col in BINARY_COLUMNS:
        df[col] = df[col].fillna(0).astype(int)
    return df
