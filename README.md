# Fake Job Posting Detector

Classifies job postings as real or fraudulent. The ML pipeline is versioned with DVC, experiments and the
model registry live on DagsHub (MLflow), the API runs on Render (FastAPI), and the UI runs on Vercel (Next.js).

> Status: phase 4 of 8. The pipeline (ingest → validate → dedupe → split → featurize → train six
> classical models → fine-tune DistilBERT → evaluate) is reproducible with the steps below. The
> model registry, API and UI are not built yet.

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

## Dev tasks

| Command | What it does |
| --- | --- |
| `uv run poe test` | pytest |
| `uv run poe lint` / `fmt` | ruff |
| `uv run poe typecheck` | mypy |
| `uv run poe tune [models] [--trials N]` | Optuna tuning with group-aware CV |
| `uv sync --group transformer` | add torch/transformers (transformer training and its tests) |

## Layout

- `src/fjd/` — the importable package shared by notebooks, the DVC pipeline and the API
- `params.yaml` — every seed, split ratio and hyperparameter
- `dvc.yaml` / `dvc.lock` — the versioned pipeline
- `reports/` — metrics (`metrics.json`, `train/`, `tuning/`) and PR-curve data for `dvc plots`
- `notebooks/` — exploration (outputs stripped on commit)
- `backend/`, `frontend/` — the deployed app (empty for now)
