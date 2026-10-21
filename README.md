# Fake Job Posting Detector

Classifies job postings as real or fraudulent. The ML pipeline is versioned with DVC, experiments and the
model registry live on DagsHub (MLflow), the API runs on Render (FastAPI), and the UI runs on Vercel (Next.js).

> Status: phase 3 of 8. The pipeline (ingest → validate → dedupe → split → featurize → train six
> classical models → evaluate) is reproducible with the steps below. The transformer, model
> registry, API and UI are not built yet.

## Reproduce from a fresh clone

You need a **free DagsHub account** (about 2 minutes). DagsHub only serves DVC data to
signed-in users, even for public repos. No other accounts or credentials are needed.

1. Sign up at https://dagshub.com and create a token at https://dagshub.com/user/settings/tokens.
2. Copy `.env.example` to `.env` and fill in `DAGSHUB_USER` and `DAGSHUB_TOKEN`.

```bash
git clone https://github.com/webmaster99-stack/fake_job_posting_detector.git
cd fake_job_posting_detector
uv sync                # exact dependency versions from uv.lock (Python 3.12)
cp .env.example .env   # then fill in DAGSHUB_USER and DAGSHUB_TOKEN
uv run poe dvc-auth    # stores them in .dvc/config.local (git-ignored)
uv run dvc pull        # data + pipeline outputs (including trained models)
uv run dvc repro       # re-run the pipeline; no-op if nothing changed
uv run dvc metrics show
uv run dvc plots show  # PR curves on the test split -> dvc_plots/index.html
```

Or with one command: `uv run poe reproduce`.

`dvc repro -f` (or any change to code or `params.yaml`) retrains models. The `evaluate` stage logs
each model to MLflow. Without extra configuration, runs go to a local store, which you can browse
with `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`. Only the maintainer logs to the
shared DagsHub MLflow server.

## Models

Six classical models share one featurizer: TF-IDF over all text fields, plus missing-field flags,
binary flags, word counts and one-hot categoricals including country. The models are logistic
regression, linear SVM, Complement Naive Bayes, random forest, XGBoost and LightGBM. They are
compared on **PR-AUC for the fraud class**, and F1/precision/recall are reported at a threshold
tuned on the validation split. The data is split by template/company group (see
`src/fjd/data/groups.py`), so expect lower but more honest numbers than published random-split
results on this dataset.

Hyperparameters in `params.yaml: classical` come from Optuna with group-aware 5-fold CV
(`notebooks/03_classical_tuning.ipynb` or `uv run poe tune`). Records are in `reports/tuning/`.

Current results (PR-AUC, fraud class; `uv run dvc metrics show` for the full set):

| Model | CV (5 grouped folds) | Val | Test |
| --- | --- | --- | --- |
| Linear SVM | 0.558 | 0.701 | 0.794 |
| Logistic regression | 0.551 | 0.700 | 0.791 |
| Random forest | 0.549 | 0.604 | 0.801 |
| LightGBM | 0.547 | 0.686 | 0.809 |
| XGBoost | 0.544 | 0.686 | 0.828 |
| Complement Naive Bayes | 0.497 | 0.525 | 0.694 |

Val and test each hold only ~100 fraud postings, so their rankings are noisy; the CV mean is the
more reliable comparison.

## Dev tasks

| Command | What it does |
| --- | --- |
| `uv run poe test` | pytest |
| `uv run poe lint` / `fmt` | ruff |
| `uv run poe typecheck` | mypy |
| `uv run poe tune [models] [--trials N]` | Optuna tuning with group-aware CV |

## Layout

- `src/fjd/` — the importable package shared by notebooks, the DVC pipeline and the API
- `params.yaml` — every seed, split ratio and hyperparameter
- `dvc.yaml` / `dvc.lock` — the versioned pipeline
- `reports/` — metrics (`metrics.json`, `train/`, `tuning/`) and PR-curve data for `dvc plots`
- `notebooks/` — exploration (outputs stripped on commit)
- `backend/`, `frontend/` — the deployed app (empty for now)
