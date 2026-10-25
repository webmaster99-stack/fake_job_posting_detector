"""Stage `evaluate`: score every trained model on val and test, and log each one to MLflow.

Models are the classical ones and the transformer. They are selected on val metrics; test metrics
are only reported. Each model is scored at the threshold its training stage tuned on val. Every
evaluation is one parent MLflow run with a nested run per model (params, metrics, PR curves,
confusion matrix, git SHA, data hash, model). The transformer's run also logs its deployability
(size and latency on this machine), which is not a `dvc metrics` value because latency varies.
"""

import json
from collections.abc import Callable
from functools import partial
from importlib.metadata import version
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from fjd.data.io import REPORTS, read_split, write_json
from fjd.data.schema import TARGET
from fjd.features.build import INPUT_COLUMNS
from fjd.models import deployability
from fjd.models.metrics import classification_metrics, fraud_scores, pr_curve
from fjd.models.train import MODELS_DIR, threshold_path
from fjd.models.train import REPORTS_DIR as TRAIN_REPORTS
from fjd.models.transformer import MODEL_DIR as TRANSFORMER_DIR
from fjd.models.transformer import TransformerClassifier, load_threshold
from fjd.utils.config import PROJECT_ROOT, load_params
from fjd.utils.plotting import CLASS_LABELS, SERIES, set_style
from fjd.utils.seed import set_seed
from fjd.utils.tracking import flatten, run_tags, setup_mlflow

OUT = REPORTS / "metrics.json"
PLOTS_DIR = REPORTS / "plots" / "pr_curve"  # test-split PR curves for `dvc plots`
SPLITS = ("val", "test")
PACKAGE_DIR = PROJECT_ROOT / "src" / "fjd"
REQUIREMENTS = ["scikit-learn", "pandas", "numpy", "scipy", "xgboost", "lightgbm"]
TRANSFORMER = "transformer"  # its name in reports, plots and MLflow


def pr_figure(name: str, curves: dict[str, pd.DataFrame], metrics: dict[str, Any]) -> Figure:
    fig = Figure(figsize=(5, 4.2))
    ax = fig.subplots()
    for color, (split, curve) in zip(SERIES, curves.items(), strict=False):
        label = f"{split} (PR-AUC {metrics[split]['pr_auc']:.3f})"
        ax.plot(curve["recall"], curve["precision"], color=color, label=label, drawstyle="steps")
    ax.set(xlabel="recall", ylabel="precision", xlim=(0, 1), ylim=(0, 1.02))
    ax.set_title(f"{name}: precision-recall (fraud class)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


def confusion_figure(name: str, m: dict[str, Any]) -> Figure:
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig = Figure(figsize=(3.8, 3.4))
    ax = fig.subplots()
    ax.imshow(cm, cmap="Blues")
    ax.grid(False)
    labels = [CLASS_LABELS[0], CLASS_LABELS[1]]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set(xlabel="predicted", ylabel="actual")
    for (i, j), v in np.ndenumerate(cm):
        color = "white" if v > cm.max() / 2 else "black"
        ax.text(j, i, f"{v:,}", ha="center", va="center", color=color)
    ax.set_title(f"{name}: test confusion matrix", fontsize=10)
    fig.tight_layout()
    return fig


def score(model: Any, threshold: float, data: dict[str, pd.DataFrame]) -> dict[str, Any]:
    metrics, curves = {}, {}
    for split, df in data.items():
        scores = fraud_scores(model, df)
        metrics[split] = classification_metrics(df[TARGET], scores, threshold)
        curves[split] = pr_curve(df[TARGET], scores)
    return {"model": model, "threshold": threshold, "metrics": metrics, "curves": curves}


def train_report(name: str) -> dict[str, Any]:
    report: dict[str, Any] = json.loads(
        (TRAIN_REPORTS / f"{name}.json").read_text(encoding="utf-8")
    )
    return report


def evaluate_classical(name: str, data: dict[str, pd.DataFrame]) -> dict[str, Any]:
    pipeline = joblib.load(MODELS_DIR / f"{name}.joblib")
    threshold = json.loads(threshold_path(name).read_text(encoding="utf-8"))["threshold"]
    result = score(pipeline, threshold, data)
    if result["metrics"]["val"] != train_report(name)["val"]:
        raise RuntimeError(f"{name}: saved model does not reproduce its train_classical metrics")
    return result


def evaluate_transformer(
    data: dict[str, pd.DataFrame], cfg: dict[str, Any], limits: dict[str, Any], seed: int
) -> dict[str, Any]:
    """Like `evaluate_classical`, but the model was exported on another machine (Colab), so its
    val PR-AUC only has to match the training report within `cfg["repro_tolerance"]`."""
    result = score(TransformerClassifier(), load_threshold(), data)
    report = train_report(TRANSFORMER)
    delta = abs(result["metrics"]["val"]["pr_auc"] - report["val"]["pr_auc"])
    if delta > cfg["repro_tolerance"]:
        raise RuntimeError(f"{TRANSFORMER}: val PR-AUC is {delta:.4f} off its train report")
    val = data["val"]
    sample = val.sample(min(cfg["latency_sample"], len(val)), random_state=seed)
    one_thread = TransformerClassifier(threads=1)
    result["deployability"] = deployability.check(
        TRANSFORMER_DIR, one_thread.predict_proba, sample, limits
    )
    result["history"] = report["history"]
    return result


def log_model_run(
    name: str,
    result: dict[str, Any],
    run_params: dict[str, Any],
    seed: int,
    log_model: Callable[[], None],
) -> None:
    with mlflow.start_run(run_name=name, nested=True, tags=run_tags("pipeline") | {"model": name}):
        mlflow.log_params({"model_name": name, "seed": seed} | flatten(run_params))
        mlflow.log_metric("threshold", result["threshold"])
        for split, m in result["metrics"].items():
            mlflow.log_metrics({f"{split}_{k}": v for k, v in m.items()})
        if "deployability" in result:
            mlflow.log_metrics({k: float(v) for k, v in result["deployability"].items()})
        for epoch, h in enumerate(result.get("history", {}).values(), start=1):
            mlflow.log_metrics({f"epoch_{k}": v for k, v in h.items()}, step=epoch)
        mlflow.log_figure(pr_figure(name, result["curves"], result["metrics"]), "pr_curve.png")
        mlflow.log_figure(confusion_figure(name, result["metrics"]["test"]), "confusion_test.png")
        log_model()


def log_classical_model(pipeline: Any, example: Any) -> None:
    mlflow.sklearn.log_model(
        pipeline,
        name="model",
        input_example=example,
        code_paths=[str(PACKAGE_DIR)],
        # skops (MLflow's default) needs minutes per model to walk the TF-IDF vocabulary.
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        pip_requirements=[f"{pkg}=={version(pkg)}" for pkg in REQUIREMENTS],
    )


def main() -> None:
    params = load_params()
    set_seed(params["seed"])
    set_style()
    data = {split: read_split(split) for split in SPLITS}
    results = {name: evaluate_classical(name, data) for name in params["classical"]}
    results[TRANSFORMER] = evaluate_transformer(
        data, params["transformer"], params["selection"], params["seed"]
    )

    write_json({name: r["metrics"] for name, r in results.items()}, OUT)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, r in results.items():
        r["curves"]["test"].to_csv(PLOTS_DIR / f"{name}.csv", index=False, lineterminator="\n")

    setup_mlflow()
    example = data["val"][INPUT_COLUMNS].head(3)
    with mlflow.start_run(run_name="evaluate", tags=run_tags("pipeline")):
        for name, hp in params["classical"].items():
            log_model_run(
                name,
                results[name],
                {"model": hp, "features": params["features"]},
                params["seed"],
                partial(log_classical_model, results[name]["model"], example),
            )
        log_model_run(
            TRANSFORMER,
            results[TRANSFORMER],
            {"model": params["transformer"]},
            params["seed"],
            # Packaged as a registry pyfunc (with the classical models) by `select_and_register`.
            lambda: mlflow.log_artifacts(str(TRANSFORMER_DIR), "model"),
        )
        summary = pd.DataFrame({n: r["metrics"]["val"] for n, r in results.items()}).T
        mlflow.log_table(summary.reset_index(names="model"), "val_metrics.json")


if __name__ == "__main__":
    main()
