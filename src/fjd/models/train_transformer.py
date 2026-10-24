"""Stage `train_transformer`: fine-tune DistilBERT on the train split and export it to ONNX.

Needs a GPU and the `transformer` dependency group, so it runs on Colab
(notebooks/04_transformer_colab.ipynb). The stage is frozen in dvc.yaml: `dvc repro` never runs
it and uses the model pulled from DVC instead.

The best epoch is chosen by val PR-AUC. The model is exported to ONNX and, if
`params.yaml: transformer.quantize`, quantized to int8. The decision threshold and the reported
val metrics come from the exported model, since that is what gets deployed. GPU training is not
bit-for-bit reproducible, so a retrained model only has to match within
`transformer.repro_tolerance` PR-AUC.
"""

import copy
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pandas as pd
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from tokenizers import Tokenizer
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    get_linear_schedule_with_warmup,
)

from fjd.data.io import REPORTS, read_split, write_json
from fjd.data.schema import TARGET
from fjd.models import deployability
from fjd.models.metrics import classification_metrics, pr_auc, tune_threshold
from fjd.models.transformer import (
    MODEL_DIR,
    ONNX_FILE,
    TEXT_CONFIG_FILE,
    THRESHOLD_FILE,
    TOKENIZER_FILE,
    TransformerClassifier,
    load_tokenizer,
    posting_text,
)
from fjd.utils.config import load_params
from fjd.utils.seed import set_seed

REPORT = REPORTS / "train" / "transformer.json"
EVAL_BATCH_SIZE = 64


def encode(tokenizer: Tokenizer, texts: list[str]) -> list[list[int]]:
    return [e.ids for e in tokenizer.encode_batch(texts)]


def batches(
    ids: list[list[int]],
    labels: np.ndarray | None,
    batch_size: int,
    pad_id: int,
    rng: np.random.Generator | None = None,
) -> Iterator[dict[str, torch.Tensor]]:
    """Padded batches, shuffled if `rng` is given. Padding only matters during training: the
    exported model scores one posting at a time."""
    order = rng.permutation(len(ids)) if rng is not None else np.arange(len(ids))
    for start in range(0, len(ids), batch_size):
        rows = order[start : start + batch_size]
        width = max(len(ids[i]) for i in rows)
        input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        for j, i in enumerate(rows):
            input_ids[j, : len(ids[i])] = torch.tensor(ids[i])
            mask[j, : len(ids[i])] = 1
        batch = {"input_ids": input_ids, "attention_mask": mask}
        if labels is not None:
            batch["labels"] = torch.as_tensor(labels[rows], dtype=torch.long)
        yield batch


@torch.no_grad()
def predict_scores(
    model: PreTrainedModel, ids: list[list[int]], pad_id: int, device: torch.device
) -> np.ndarray:
    """P(fraud) from the torch model, used to pick the best epoch."""
    model.eval()
    out = []
    for batch in batches(ids, None, EVAL_BATCH_SIZE, pad_id):
        logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
        out.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def fine_tune(
    model: PreTrainedModel,
    tokenizer: Tokenizer,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: dict[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[PreTrainedModel, dict[str, dict[str, float]]]:
    """Train for `cfg["epochs"]` epochs; return the model at its best val PR-AUC and the history."""
    pad_id = model.config.pad_token_id or 0
    train_ids = encode(tokenizer, posting_text(train_df, cfg["field_words"]))
    val_ids = encode(tokenizer, posting_text(val_df, cfg["field_words"]))
    y_train = train_df[TARGET].to_numpy()

    model.to(device)  # type: ignore[arg-type]  # transformers mistypes `.to`
    no_decay = ("bias", "LayerNorm.weight")
    groups = [
        {
            "params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)],
            "weight_decay": cfg["weight_decay"],
        },
        {
            "params": [p for n, p in model.named_parameters() if n.endswith(no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(groups, lr=cfg["learning_rate"])
    steps = cfg["epochs"] * -(-len(train_ids) // cfg["batch_size"])
    scheduler = get_linear_schedule_with_warmup(optimizer, int(cfg["warmup_ratio"] * steps), steps)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, cfg["fraud_weight"]], device=device))
    use_amp = device.type == "cuda"  # fp16 on GPU, fp32 on CPU
    scaler = torch.amp.GradScaler(enabled=use_amp)
    rng = np.random.default_rng(seed)

    history: dict[str, dict[str, float]] = {}
    best_auc, best_state = -1.0, None
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        losses = []
        for batch in batches(train_ids, y_train, cfg["batch_size"], pad_id, rng):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(**batch).logits
            loss = loss_fn(logits.float(), labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(loss.item())
        val_auc = pr_auc(val_df[TARGET], predict_scores(model, val_ids, pad_id, device))
        history[f"epoch_{epoch}"] = {
            "train_loss": round(float(np.mean(losses)), 4),
            "val_pr_auc": round(val_auc, 4),
        }
        print(f"epoch {epoch}: train loss {np.mean(losses):.4f}, val PR-AUC {val_auc:.4f}")
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, history


class _Logits(nn.Module):
    """The model with plain tensor inputs and a single output, for the ONNX exporter."""

    def __init__(self, model: PreTrainedModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def export_onnx(
    model: PreTrainedModel,
    tokenizer: Tokenizer,
    sample_texts: list[str],
    path: Path,
    quantize: bool,
) -> None:
    """Export to ONNX with dynamic batch and sequence axes; optionally quantize to int8."""
    model = model.cpu().float().eval()
    # Two rows of different lengths, so the exporter doesn't specialize on batch size 1.
    ids = encode(tokenizer, sample_texts[:2])
    sample = next(batches(ids, None, 2, model.config.pad_token_id or 0))
    dynamic = {0: torch.export.Dim.DYNAMIC, 1: torch.export.Dim.DYNAMIC}
    with tempfile.TemporaryDirectory() as tmp:
        fp32 = Path(tmp) / "fp32.onnx" if quantize else path
        torch.onnx.export(
            _Logits(model).eval(),
            (sample["input_ids"], sample["attention_mask"]),
            fp32,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_shapes=(dynamic, dynamic),
            dynamo=True,
            external_data=False,
            verbose=False,  # its progress lines contain emoji, which crash cp1252 stdout
        )
        if quantize:
            # The quantizer rewrites nodes but re-checks the exporter's stored intermediate
            # shapes, which then conflict; it infers what it needs without them.
            graph = onnx.load(fp32)
            del graph.graph.value_info[:]
            onnx.save(graph, fp32)
            # uint8 weights: the int8-weight kernels on AVX2 CPUs without VNNI can saturate,
            # which would make scores depend on the CPU the model runs on.
            quantize_dynamic(fp32, path, weight_type=QuantType.QUInt8)


def save_model(
    model: PreTrainedModel,
    tokenizer: Tokenizer,
    cfg: dict[str, Any],
    sample_texts: list[str],
    out_dir: Path,
) -> None:
    """Write the self-contained model directory that `TransformerClassifier` loads."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_dir / TOKENIZER_FILE))
    text_config = {
        "pretrained": cfg["pretrained"],
        "revision": cfg["revision"],
        "max_length": cfg["max_length"],
        "field_words": cfg["field_words"],
    }
    write_json(text_config, out_dir / TEXT_CONFIG_FILE)
    export_onnx(model, tokenizer, sample_texts, out_dir / ONNX_FILE, cfg["quantize"])


def load_pretrained(cfg: dict[str, Any]) -> tuple[PreTrainedModel, Tokenizer]:
    """The pinned pretrained checkpoint with a new 2-class head, and its fast tokenizer."""
    hf_tokenizer = AutoTokenizer.from_pretrained(cfg["pretrained"], revision=cfg["revision"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["pretrained"], revision=cfg["revision"], num_labels=2
    )
    # Train with the exact tokenizer the exported model will use: the same tokenizer.json.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / TOKENIZER_FILE
        hf_tokenizer.backend_tokenizer.save(str(path))
        tokenizer = load_tokenizer(path, cfg["max_length"])
    return model, tokenizer


def train(
    model: PreTrainedModel,
    tokenizer: Tokenizer,
    cfg: dict[str, Any],
    limits: dict[str, Any],
    seed: int,
    data: dict[str, pd.DataFrame],
    out_dir: Path = MODEL_DIR,
) -> dict[str, Any]:
    """Fine-tune, export to `out_dir`, tune the threshold on val, and return the train report."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, val_df = data["train"], data["val"]
    model, history = fine_tune(model, tokenizer, train_df, val_df, cfg, seed, device)
    save_model(model, tokenizer, cfg, posting_text(val_df, cfg["field_words"]), out_dir)

    clf = TransformerClassifier(out_dir, threads=1)
    scores = clf.predict_proba(val_df)[:, 1]
    threshold = tune_threshold(val_df[TARGET], scores)
    write_json({"threshold": threshold}, out_dir / THRESHOLD_FILE)
    sample = val_df.sample(min(cfg["latency_sample"], len(val_df)), random_state=seed)
    return {
        "val": classification_metrics(val_df[TARGET], scores, threshold),
        "best_torch_val_pr_auc": max(h["val_pr_auc"] for h in history.values()),
        "history": history,
        # Latency was measured on the training machine's CPU; `evaluate` re-measures it.
        "deployability": deployability.check(out_dir, clf.predict_proba, sample, limits),
    }


def main() -> None:
    # Deterministic cuBLAS; must be set before CUDA initializes.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    params = load_params()
    set_seed(params["seed"])
    torch.use_deterministic_algorithms(True, warn_only=True)
    cfg = params["transformer"]
    model, tokenizer = load_pretrained(cfg)
    data = {split: read_split(split) for split in ("train", "val")}
    report = train(model, tokenizer, cfg, params["selection"], params["seed"], data)
    write_json(report, REPORT)
    print(f"val PR-AUC (exported model): {report['val']['pr_auc']:.4f}")
    print(f"deployability: {report['deployability']}")


if __name__ == "__main__":
    main()
