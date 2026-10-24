"""The transformer's text, export and ONNX inference, on a tiny randomly initialised DistilBERT.

Training tests need the `transformer` dependency group (`uv sync --group transformer`) and are
skipped without it. Nothing is downloaded: the tokenizer is trained on the toy postings.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from fjd.data.schema import BINARY_COLUMNS, CATEGORICAL_COLUMNS, TARGET, TEXT_COLUMNS
from fjd.models import deployability
from fjd.models.transformer import (
    FLAG_LABELS,
    LONG_FIELDS,
    META_FIELDS,
    TransformerClassifier,
    load_threshold,
    posting_text,
)
from tests.test_data import make_df
from tests.test_models import make_postings

FIELD_WORDS = {"company_profile": 5, "description": 8, "requirements": 5, "benefits": 5}
CFG: dict[str, Any] = {
    "pretrained": "tiny",
    "revision": "none",
    "max_length": 64,
    "field_words": FIELD_WORDS,
    "epochs": 4,
    "batch_size": 16,
    "learning_rate": 3e-4,
    "weight_decay": 0.0,
    "warmup_ratio": 0.0,
    "fraud_weight": 1.0,
    # A tiny model's scores sit close together, so int8 rounding would scramble their order.
    # The fixture trains in fp32; test_quantized_export covers quantization.
    "quantize": False,
    "latency_sample": 10,
}
LIMITS = {"max_model_size_mb": 300, "max_p95_latency_ms": 300}


def test_posting_text_covers_every_input_column() -> None:
    covered = {"title"} | set(META_FIELDS) | set(FLAG_LABELS) | set(LONG_FIELDS)
    assert covered == set(TEXT_COLUMNS + CATEGORICAL_COLUMNS + BINARY_COLUMNS)


def test_posting_text_spells_out_missing_fields_and_placeholders() -> None:
    df = make_df(1)
    df.loc[0, "description"] = "call #PHONE_abc123# or mail #EMAIL_0a1b# " + "word " * 20
    df.loc[0, "has_company_logo"] = 0
    text = posting_text(df, FIELD_WORDS)[0]
    assert text.startswith("title: Job 0. location: none.")
    assert "company logo: no." in text and "company profile: none" in text
    assert "description: call PHONE or mail EMAIL word word word ..." in text


def tiny_model_and_tokenizer(texts: list[str]) -> tuple[Any, Any]:
    torch = pytest.importorskip("torch")
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers
    from transformers import DistilBertConfig, DistilBertForSequenceClassification

    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.train_from_iterator(texts, trainers.WordPieceTrainer(special_tokens=specials))
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[(t, tokenizer.token_to_id(t)) for t in ("[CLS]", "[SEP]")],
    )
    tokenizer.enable_truncation(CFG["max_length"])
    torch.manual_seed(0)
    config = DistilBertConfig(
        vocab_size=tokenizer.get_vocab_size(),
        dim=32,
        n_layers=2,
        n_heads=2,
        hidden_dim=64,
        max_position_embeddings=CFG["max_length"],
        num_labels=2,
        pad_token_id=0,
    )
    return DistilBertForSequenceClassification(config), tokenizer


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    pytest.importorskip("torch")
    from fjd.models.train_transformer import train

    df = make_postings()
    data = {"train": df, "val": df}
    model, tokenizer = tiny_model_and_tokenizer(posting_text(df, FIELD_WORDS))
    out_dir = tmp_path_factory.mktemp("transformer")
    report = train(model, tokenizer, CFG, LIMITS, seed=0, data=data, out_dir=out_dir)
    return {"df": df, "model": model, "tokenizer": tokenizer, "dir": out_dir, "report": report}


def test_training_learns_and_reports(trained: dict[str, Any]) -> None:
    report = trained["report"]
    assert report["val"]["pr_auc"] > 0.9
    assert list(report["history"]) == [f"epoch_{i}" for i in range(1, CFG["epochs"] + 1)]
    assert report["deployability"]["deployable"]
    assert {p.name for p in trained["dir"].iterdir()} == {
        "model.onnx",
        "tokenizer.json",
        "text.json",
        "threshold.json",
    }


def test_exported_model_scores_raw_postings_deterministically(trained: dict[str, Any]) -> None:
    df: pd.DataFrame = trained["df"]
    clf = TransformerClassifier(trained["dir"])
    proba = clf.predict_proba(df.drop(columns=[TARGET, "group_id"]))
    assert proba.shape == (len(df), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1)
    # A posting's score doesn't depend on what else is scored with it.
    np.testing.assert_array_equal(clf.predict_proba(df.iloc[[3]])[0], proba[3])
    assert 0 <= load_threshold(trained["dir"]) <= 1


def test_onnx_export_matches_torch(trained: dict[str, Any]) -> None:
    torch = pytest.importorskip("torch")
    from fjd.models.train_transformer import encode

    df: pd.DataFrame = trained["df"].head(8)
    model, tokenizer = trained["model"], trained["tokenizer"]
    onnx_scores = TransformerClassifier(trained["dir"]).predict_proba(df)[:, 1]
    with torch.no_grad():
        torch_scores = [
            torch.softmax(model(input_ids=torch.tensor([ids])).logits, -1)[0, 1].item()
            for ids in encode(tokenizer, posting_text(df, FIELD_WORDS))
        ]
    np.testing.assert_allclose(onnx_scores, torch_scores, atol=1e-5)


def test_quantized_export(trained: dict[str, Any], tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from fjd.models.train_transformer import save_model

    df: pd.DataFrame = trained["df"]
    texts = posting_text(df, FIELD_WORDS)
    save_model(trained["model"], trained["tokenizer"], CFG | {"quantize": True}, texts, tmp_path)
    proba = TransformerClassifier(tmp_path).predict_proba(df)
    assert np.isfinite(proba).all() and ((proba >= 0) & (proba <= 1)).all()
    fp32_size = (trained["dir"] / "model.onnx").stat().st_size
    assert (tmp_path / "model.onnx").stat().st_size < fp32_size


def test_size_mb_sums_a_directory(tmp_path: Path) -> None:
    (tmp_path / "a.bin").write_bytes(b"x" * 1_000_000)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"x" * 500_000)
    assert deployability.size_mb(tmp_path) == 1.5


def test_check_flags_slow_models(tmp_path: Path) -> None:
    (tmp_path / "m.bin").write_bytes(b"x" * 10)
    postings = make_df(10)
    ok = deployability.check(tmp_path, lambda df: None, postings, LIMITS)
    assert ok["deployable"] and ok["p95_latency_ms"] < 300
    strict = LIMITS | {"max_p95_latency_ms": -1}
    assert not deployability.check(tmp_path, lambda df: None, postings, strict)["deployable"]
