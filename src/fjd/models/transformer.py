"""The fine-tuned transformer at inference time: posting text, tokenizer and the ONNX model.

Training needs torch and a GPU (`fjd.models.train_transformer`, run on Colab). This module only
needs onnxruntime and tokenizers, so the pipeline and the API can score the model without torch.

The model directory (`models/transformer/`) is self-contained: `model.onnx`, the `tokenizer.json`
it was trained with, `text.json` (how a posting becomes model input) and `threshold.json`.

Postings are scored one at a time, never in padded batches. The int8 model quantizes each
input's activations with a scale computed over the whole tensor, so a posting's score would
otherwise depend on what it was batched with. One at a time is also how the API scores them.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd
from tokenizers import Tokenizer

from fjd.utils.config import PROJECT_ROOT

MODEL_DIR = PROJECT_ROOT / "models" / "transformer"
ONNX_FILE = "model.onnx"
TOKENIZER_FILE = "tokenizer.json"
TEXT_CONFIG_FILE = "text.json"
THRESHOLD_FILE = "threshold.json"

# EMSCAD anonymized links/contacts as e.g. `#URL_<hash>#`; keep the kind as a plain word.
PLACEHOLDER = r"#(URL|EMAIL|PHONE)_[^#]*#"
# Short fields, in the order they appear in the model input, with the label written before each.
META_FIELDS = {
    "location": "location",
    "department": "department",
    "salary_range": "salary",
    "employment_type": "employment type",
    "required_experience": "experience",
    "required_education": "education",
    "industry": "industry",
    "function": "function",
}
FLAG_LABELS = {
    "telecommuting": "remote",
    "has_company_logo": "company logo",
    "has_questions": "screening questions",
}
# Long fields come last, each cut to its word budget (`params.yaml: transformer.field_words`),
# so truncation to `max_length` tokens drops the least useful text (benefits) first.
LONG_FIELDS = {
    "company_profile": "company profile",
    "description": "description",
    "requirements": "requirements",
    "benefits": "benefits",
}
MISSING = "none"  # a missing field is a strong fraud signal, so it is spelled out


def _clean(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return MISSING
    text = " ".join(str(value).split())
    return text or MISSING


def _first_words(text: str, n: int) -> str:
    words = text.split()
    return " ".join(words[:n]) + (" ..." if len(words) > n else "")


def posting_text(df: pd.DataFrame, field_words: dict[str, int]) -> list[str]:
    """One model input per posting: title, short fields, flags, then the long fields.

    e.g. "title: Data Engineer. location: GB, LND, London. ... company logo: yes. ...
    description: We are looking for ..."
    """
    df = df.replace(PLACEHOLDER, r" \1 ", regex=True)
    texts = []
    for row in df.to_dict("records"):
        parts = [f"title: {_clean(row['title'])}."]
        parts += [f"{label}: {_clean(row[col])}." for col, label in META_FIELDS.items()]
        parts += [
            f"{label}: {'yes' if row[col] == 1 else 'no'}." for col, label in FLAG_LABELS.items()
        ]
        parts += [
            f"{label}: {_first_words(_clean(row[col]), field_words[col])}"
            for col, label in LONG_FIELDS.items()
        ]
        texts.append(" ".join(parts))
    return texts


def load_tokenizer(path: Path, max_length: int) -> Tokenizer:
    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.enable_truncation(max_length)
    tokenizer.no_padding()
    return tokenizer


def softmax_fraud(logits: np.ndarray) -> np.ndarray:
    """P(fraud) from (n, 2) logits."""
    z = logits.astype(np.float64)
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e[:, 1] / e.sum(axis=1)


class TransformerClassifier:
    """The exported model with sklearn's `predict_proba` interface, on raw canonical columns."""

    def __init__(self, model_dir: Path = MODEL_DIR, threads: int | None = None) -> None:
        self.model_dir = model_dir
        self.text_config: dict[str, Any] = json.loads(
            (model_dir / TEXT_CONFIG_FILE).read_text(encoding="utf-8")
        )
        self.tokenizer = load_tokenizer(model_dir / TOKENIZER_FILE, self.text_config["max_length"])
        options = ort.SessionOptions()
        if threads is not None:
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_dir / ONNX_FILE), options, providers=["CPUExecutionProvider"]
        )

    def texts(self, X: pd.DataFrame) -> list[str]:
        return posting_text(X, self.text_config["field_words"])

    def logits(self, text: str) -> np.ndarray:
        encoding = self.tokenizer.encode(text)
        feed = {
            "input_ids": np.array([encoding.ids], dtype=np.int64),
            "attention_mask": np.array([encoding.attention_mask], dtype=np.int64),
        }
        return np.asarray(self.session.run(["logits"], feed)[0])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        logits = np.vstack([self.logits(t) for t in self.texts(X)])
        fraud = softmax_fraud(logits)
        return np.column_stack([1 - fraud, fraud])


def load_threshold(model_dir: Path = MODEL_DIR) -> float:
    data = json.loads((model_dir / THRESHOLD_FILE).read_text(encoding="utf-8"))
    return float(data["threshold"])
