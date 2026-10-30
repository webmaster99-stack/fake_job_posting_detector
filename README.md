# Fake Job Posting Detector

Classifies job postings as real or fraudulent. The ML pipeline is versioned with DVC, experiments and the
model registry live on DagsHub (MLflow), the API runs on Render (FastAPI), and the UI runs on Vercel (Next.js).

> Status: phase 8 of 8 in progress (CI/CD and monitoring). The pipeline (ingest → validate →
> dedupe → split → featurize → train six classical models → fine-tune DistilBERT → evaluate →
> select and register → drift reference) is reproducible with the steps below. The API
> (`backend/`) runs on Render and the UI (`frontend/`) on Vercel.

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
| Logistic regression | 0.551 | 0.699 | 0.791 |
| Random forest | 0.549 | 0.604 | 0.801 |
| LightGBM | 0.547 | 0.686 | 0.809 |
| XGBoost | 0.541 | 0.575 | 0.804 |
| Complement Naive Bayes | 0.497 | 0.525 | 0.694 |

Val and test each hold only ~100 fraud postings, so their rankings are noisy; the CV mean is the
more reliable comparison.

### Transformer

DistilBERT (`distilbert-base-uncased`, pinned revision) is fine-tuned on a GPU in
`notebooks/04_transformer_colab.ipynb`, exported to ONNX and quantized to int8, so it runs on a
CPU with onnxruntime and no torch. Each posting becomes one text: the title, the short fields, the
binary flags as words, then the long fields, each cut to a word budget, up to 256 tokens. The
size and single-posting latency are checked against the deployment limits (300 MB, p95 ≤ 300 ms).

The `train_transformer` stage is frozen: `dvc repro` uses the model pulled from DVC and never
retrains it. To retrain it yourself, open the notebook in Colab with a T4 GPU and add your
DagsHub credentials as Colab secrets (`DAGSHUB_USER`, `DAGSHUB_TOKEN`). The notebook checks that
your val PR-AUC matches the committed one within 0.005; GPU training isn't bit-for-bit
reproducible.

## Model selection and registry

The `select_and_register` stage ranks every model by val PR-AUC and keeps the best one that fits
the deployment limits in `params.yaml: selection` (≤300 MB, p95 ≤300 ms per posting on one CPU
thread, measured on your machine). It wraps the model in an MLflow pyfunc with its val-tuned
threshold and a Platt calibration fitted on val. The calibration turns SVM margins and
class-weighted scores into real fraud probabilities, without changing the ranking or the
labels. The stage then registers the pyfunc as a new version of `fake-job-detector` with the
alias `@challenger`. `reports/selection.json` records the choice.

The API serves `@champion`. Promoting a challenger is a deliberate step:

```bash
uv run poe promote   # compares @challenger with @champion, then asks before moving the alias
```

The challenger is only promoted if it fits the limits and beats the champion's val PR-AUC.
`notebooks/05_model_comparison.ipynb` checks the choice first: bootstrap intervals on the
ranking, deployability of every model, and calibration on the test split.

Without `MLFLOW_TRACKING_URI`, the registry is the local `mlflow.db`, so `dvc repro` registers
versions there and never touches the shared DagsHub registry.

## API

`backend/` is a FastAPI app that serves `models:/fake-job-detector@champion`:

| Endpoint | What it does |
| --- | --- |
| `POST /predict` | verdict, calibrated fraud probability, threshold and the features that drove the score |
| `POST /feedback` | 👍/👎 on a prediction (`{"prediction_id", "correct", "comment"}`) |
| `GET /health` | answers once the model is loaded (the UI pings it to wake the API up) |
| `GET /model-info` | registered version, model type, threshold and its registration metrics |

A posting has one field per dataset column; only `title` and the three flags (`telecommuting`,
`has_company_logo`, `has_questions`) are required. Leave unknown fields out: a missing field is
itself a fraud signal. Links, emails and phone numbers are replaced by placeholders before
scoring, as in the training data (`src/fjd/serving.py`). Explanations are exact weight × value
contributions for linear models and TreeSHAP for XGBoost/LightGBM (`src/fjd/models/explain.py`).

Every prediction (the posting as the model saw it, the score and the model version) and every
piece of feedback is stored in Postgres (Neon in production) for drift monitoring and future
training data. Run it locally:

```bash
uv run poe migrate   # creates backend/local.db (or migrates DATABASE_URL)
uv run poe api       # http://localhost:8000/docs
```

Without `MLFLOW_TRACKING_URI` the API loads `@champion` from the local `mlflow.db`, which only
has one after `dvc repro` and `uv run poe promote`. `MODEL_URI` can point at any MLflow model
instead. In production the API runs in Docker on Render (`backend/Dockerfile`, `render.yaml`),
with a read-only DagsHub token, and CORS limited to `ALLOWED_ORIGINS`:

```bash
docker build -f backend/Dockerfile -t fjd-api .
```

## CI/CD and monitoring

GitHub Actions workflows (`.github/workflows/`):

| Workflow | When | What it does |
| --- | --- | --- |
| `ci.yml` | every push to `main` and every PR | ruff, mypy and pytest; frontend lint, typecheck and build |
| `pipeline-pr.yml` | PRs that touch `src/`, `params.yaml`, `dvc.yaml`, `dvc.lock`, `*.dvc`, `pyproject.toml` or `uv.lock` | `dvc pull`, then `dvc repro --force`. Fails unless the metrics committed in the PR are reproduced (`python -m fjd.utils.repro_check`), and posts a CML comment with `dvc metrics diff` against `main` |
| `reproduce.yml` | weekly and by hand | the steps under "Reproduce from a fresh clone" on a clean runner: `dvc repro` must change nothing, and a forced re-run must give the committed metrics |
| `deploy.yml` | pushes to `main` that change the API's code, and by hand (e.g. after `poe promote`) | calls the Render deploy hook. Vercel deploys the frontend through its Git integration |
| `drift.yml` | weekly and by hand | compares the last week's live predictions with the training data (Evidently), uploads the report, logs it to MLflow, and opens (or comments on) a `drift` issue if at least half of the monitored columns drifted |

Drift monitoring compares the featurizer's tabular inputs: missing-field flags, binary flags,
link/email/phone flags, word counts and categoricals (`src/fjd/monitoring/features.py`). The
`drift_reference` stage computes them on the training split. Thresholds and the window are in
`params.yaml: monitoring`. With fewer than `min_rows` predictions in the window there is no
report. Run it locally against `DATABASE_URL` (or `backend/local.db`):

```bash
uv sync --group monitoring
uv run dvc pull drift_reference
uv run python -m fjd.monitoring.drift --days 30   # -> drift_report/
```

The workflows need these repository secrets (Settings → Secrets and variables → Actions):

| Secret | Used by | Value |
| --- | --- | --- |
| `DAGSHUB_USER`, `DAGSHUB_TOKEN` | pipeline, reproduce, drift | a DagsHub account and token with read access (for `dvc pull`) |
| `RENDER_DEPLOY_HOOK_URL` | deploy | the service's deploy hook (Render dashboard → Settings → Deploy Hook) |
| `DRIFT_DATABASE_URL` | drift | a read-only Neon connection string (direct endpoint) |
| `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` | drift | DagsHub credentials with write access, to log drift runs |

PRs from forks get no secrets, so the pipeline workflow skips them.

## Dev tasks

| Command | What it does |
| --- | --- |
| `uv run poe test` | pytest |
| `uv run poe lint` / `fmt` | ruff |
| `uv run poe typecheck` | mypy |
| `uv run poe tune [models] [--trials N]` | Optuna tuning with group-aware CV |
| `uv run poe promote [--yes]` | promote the registered `@challenger` to `@champion` |
| `uv run poe api` / `migrate` | run the API locally / apply database migrations |
| `uv sync --group transformer` | add torch/transformers (transformer training and its tests) |
| `uv sync --group monitoring` | add Evidently (drift report and its tests) |

## Layout

- `src/fjd/` — the importable package shared by notebooks, the DVC pipeline and the API
- `backend/` — the FastAPI app, its database tables and Alembic migrations, and its Dockerfile
- `params.yaml` — every seed, split ratio and hyperparameter
- `dvc.yaml` / `dvc.lock` — the versioned pipeline
- `reports/` — metrics (`metrics.json`, `selection.json`, `train/`, `tuning/`) and PR-curve data for `dvc plots`
- `notebooks/` — exploration (outputs stripped on commit)
- `frontend/` — the Next.js UI (see `frontend/README.md`)
- `.github/workflows/` — CI, pipeline reproduction, deploys and drift monitoring
