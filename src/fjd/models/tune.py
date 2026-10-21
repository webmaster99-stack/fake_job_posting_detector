"""Optuna tuning with group-aware CV (notebook 03, or `uv run poe tune [models] [--trials N]`).

Not a DVC stage: tuning is slow, so the best hyperparameters are copied into
`params.yaml: classical`, and `dvc repro` retrains with them deterministically. A summary of each
study is written to `reports/tuning/<model>.json` as a record of where those values came from.

The objective is mean PR-AUC over StratifiedGroupKFold folds of the train split (see `cv.py`).
The val split is not used here; it is kept for threshold tuning and model comparison.
"""

import argparse
import time
from collections.abc import Callable
from typing import Any

import mlflow
import numpy as np
import optuna
import yaml

from fjd.data.io import REPORTS, read_split, write_json
from fjd.models.classical import MODELS
from fjd.models.cv import Fold, cross_val_pr_auc, group_folds
from fjd.utils.config import load_params
from fjd.utils.seed import set_seed
from fjd.utils.tracking import run_tags, setup_mlflow

REPORTS_DIR = REPORTS / "tuning"


def _logreg(t: optuna.Trial) -> dict[str, Any]:
    return {"C": t.suggest_float("C", 1e-3, 1e2, log=True)}


def _linear_svm(t: optuna.Trial) -> dict[str, Any]:
    return {"C": t.suggest_float("C", 1e-4, 10.0, log=True)}


def _naive_bayes(t: optuna.Trial) -> dict[str, Any]:
    return {
        "alpha": t.suggest_float("alpha", 1e-3, 10.0, log=True),
        "norm": t.suggest_categorical("norm", [False, True]),
    }


def _k_features(t: optuna.Trial, high: int) -> dict[str, Any]:
    return {"k_features": t.suggest_int("k_features", 500, high, log=True)}


def _random_forest(t: optuna.Trial) -> dict[str, Any]:
    return _k_features(t, 20000) | {
        "n_estimators": t.suggest_int("n_estimators", 200, 800, step=100),
        "max_features": t.suggest_categorical("max_features", ["sqrt", "log2", 0.05, 0.15]),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 10),
        "max_depth": t.suggest_categorical("max_depth", [None, 20, 40, 80]),
    }


def _boosting(t: optuna.Trial) -> dict[str, Any]:
    """Shared by XGBoost and LightGBM (same sklearn-API names). Capped at 600 trees and 10k
    features: wider ranges made a 40-trial XGBoost study run for more than 16 hours."""
    return _k_features(t, 10000) | {
        "n_estimators": t.suggest_int("n_estimators", 100, 600, step=50),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": t.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.1, 1.0),
        "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "scale_pos_weight": t.suggest_float("scale_pos_weight", 1.0, 25.0, log=True),
    }


def _xgboost(t: optuna.Trial) -> dict[str, Any]:
    return _boosting(t) | {
        "max_depth": t.suggest_int("max_depth", 3, 10),
        "min_child_weight": t.suggest_float("min_child_weight", 0.5, 20.0, log=True),
    }


def _lightgbm(t: optuna.Trial) -> dict[str, Any]:
    return _boosting(t) | {
        "num_leaves": t.suggest_int("num_leaves", 7, 127, log=True),
        "min_child_samples": t.suggest_int("min_child_samples", 5, 100, log=True),
    }


SEARCH_SPACES: dict[str, Callable[[optuna.Trial], dict[str, Any]]] = {
    "logreg": _logreg,
    "linear_svm": _linear_svm,
    "naive_bayes": _naive_bayes,
    "random_forest": _random_forest,
    "xgboost": _xgboost,
    "lightgbm": _lightgbm,
}
assert SEARCH_SPACES.keys() == MODELS.keys()


def tune(name: str, folds: list[Fold], n_trials: int, seed: int) -> optuna.Study:
    """Deterministic for a given seed: sequential TPE sampling, seeded models, fixed folds.

    Trials run one at a time (parallel trials would make TPE order-dependent); the folds within
    a trial run in parallel instead.
    """

    def objective(trial: optuna.Trial) -> float:
        scores = cross_val_pr_auc(name, SEARCH_SPACES[name](trial), folds, seed)
        trial.set_user_attr("fold_pr_auc", scores)
        return float(np.mean(scores))

    study = optuna.create_study(
        study_name=name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    start = time.monotonic()

    def progress(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        elapsed = (time.monotonic() - start) / 60
        print(
            f"{name} trial {trial.number + 1}/{n_trials}: {trial.value:.4f} "
            f"(best {study.best_value:.4f}, {elapsed:.0f} min)",
            flush=True,
        )

    study.optimize(objective, n_trials=n_trials, callbacks=[progress])
    return study


def summarize(study: optuna.Study) -> dict[str, Any]:
    return {
        "best_params": study.best_params,
        "cv_pr_auc": round(study.best_value, 6),
        "cv_pr_auc_folds": [round(s, 6) for s in study.best_trial.user_attrs["fold_pr_auc"]],
        "trials": len(study.trials),
    }


def log_study(name: str, summary: dict[str, Any], source: str, study: optuna.Study) -> None:
    """One MLflow run per tuned model: best params, CV score and the full trials table."""
    with mlflow.start_run(run_name=f"tune_{name}", tags=run_tags(source) | {"model": name}):
        mlflow.log_params({"model_name": name, **summary["best_params"]})
        mlflow.log_metrics({"cv_pr_auc": summary["cv_pr_auc"], "trials": summary["trials"]})
        trials = study.trials_dataframe(attrs=("number", "value", "params", "state"))
        mlflow.log_table(trials.astype({"state": str}), "trials.json")


def run_tuning(
    name: str, folds: list[Fold], n_trials: int, seed: int, source: str
) -> tuple[optuna.Study, dict[str, Any]]:
    """Tune, write `reports/tuning/<name>.json` and log the study to MLflow."""
    study = tune(name, folds, n_trials, seed)
    summary = summarize(study)
    write_json(summary, REPORTS_DIR / f"{name}.json")
    log_study(name, summary, source, study)
    return study, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="*", choices=SEARCH_SPACES, help="default: all")
    parser.add_argument("--trials", type=int, help="default: params.yaml tuning.n_trials")
    args = parser.parse_args()

    params = load_params()
    seed, cfg = params["seed"], params["tuning"]
    set_seed(seed)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    setup_mlflow()
    folds = group_folds(read_split("train"), params["features"], cfg["cv_folds"], seed)
    for name in args.models or SEARCH_SPACES:
        _, summary = run_tuning(name, folds, args.trials or cfg["n_trials"], seed, "tune_cli")
        print(f"{name}: CV PR-AUC {summary['cv_pr_auc']:.4f} ({summary['trials']} trials)")
        print(yaml.safe_dump({name: summary["best_params"]}, sort_keys=False))


if __name__ == "__main__":
    main()
