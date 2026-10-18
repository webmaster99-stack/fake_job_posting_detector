# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Binary classifier for real vs. fraudulent job postings (EMSCAD dataset, target column `fraudulent`). Phase 2 of 8 (data) is built. `dvc.yaml` has the data stages `ingest → validate → clean_dedupe → split` (code in `src/fjd/data/`; canonical columns and the pandera schema in `src/fjd/data/schema.py`), and `notebooks/01_eda.ipynb` exists. `src/fjd/{features,models,monitoring}`, `backend/`, `frontend/` and `.github/workflows/` are still empty. Don't assume training stages, an API, or CI exist — check first.

- Raw data: `data/raw/emscad.csv` (tracked by `data/raw/emscad.csv.dvc`). Maintainer re-download: `uv run python -m fjd.data.download` (Kaggle creds from `.env`).
- Every DVC stage is a `python -m fjd.<module>` entry point. Its `deps` must list the stage module **and** every shared helper it imports (`io.py`, `schema.py`, `utils/*.py`), or edits to those won't invalidate it.
- Write text artifacts with LF line endings (see `fjd.data.io.write_json`) so `dvc.lock` hashes match across Windows and Linux. Parquet is written with `index=False`.

## Commands

Python 3.12 only (`>=3.12,<3.13`), managed with `uv`. Tasks are defined via poethepoet in `pyproject.toml`:

```bash
uv sync                      # install from uv.lock (dev + notebooks groups by default)
uv run poe test              # pytest (testpaths = tests/)
uv run pytest tests/test_utils.py::test_set_seed_is_deterministic   # single test
uv run poe lint              # ruff check .
uv run poe fmt               # ruff format .
uv run poe typecheck         # mypy src
uv run poe reproduce         # dvc pull -> dvc repro -> dvc metrics show
uv run pre-commit run --all-files
```

Ruff: line length 100, rules `E,F,I,B,UP,SIM`. Pre-commit also runs nbstripout (notebook outputs are stripped on commit), blocks files >1000 KB, and detects private keys.

## Architecture

Planned stack: DVC pipeline for data/models → MLflow tracking + model registry on DagsHub (also the DVC remote) → FastAPI backend on Render → Next.js frontend on Vercel.

- `src/fjd/` is the single importable package (built with hatchling) shared by notebooks, DVC stages, and the backend. Put reusable logic here, not in notebooks or `backend/`.
- `params.yaml` is the single source of truth for every seed, split ratio, and hyperparameter; read it via `fjd.utils.config.load_params()`. Don't hardcode tunables. DVC stages will depend on it, so edits invalidate downstream stages.
- `fjd.utils.config.PROJECT_ROOT` is resolved as `parents[3]` of `config.py`; moving that file breaks path resolution.
- Call `fjd.utils.seed.set_seed(params["seed"])` at entry points for reproducibility (seeds Python, NumPy, and torch if installed).
- Model selection (`params.yaml: selection`) optimizes `pr_auc` under deployment constraints: model ≤300 MB and p95 latency ≤300 ms. Candidate models must respect these to be servable on Render.
- Data and model artifacts (`data/raw|interim|processed/*`, `models/*`) are DVC-tracked and git-ignored; only `.gitkeep` files are committed. `.dvc/config` has `autostage = true`, so DVC commands auto-`git add` their `.dvc`/lock files.

## Build plan

Agreed with the client before implementation. Everything below is **planned**, not built, unless the Project status section says otherwise. Ask before changing any of these decisions.

**Hard requirement:** a stranger can clone the repo, run `uv sync && uv run poe dvc-auth && uv run dvc pull && uv run dvc repro`, and get identical data, metrics and models. The only credential needed is their own free DagsHub account/token (see below). Classical models must match exactly on CPU. The transformer must match within tolerance (|ΔPR-AUC| ≤ 0.005).

### Decisions
- **Data:** start with EMSCAD (~17.9k rows, ~4.8% fraud). The maintainer adds the raw file to DVC once, from Kaggle. Everyone else gets it with `dvc pull`. More datasets get added later through the `datasets:` list in `params.yaml`; the ingest stage merges them and adds a `source` column.
- **Models:** start with TF-IDF features plus tabular ones (missing-field flags, `has_company_logo`, …). Compare LogReg, LinearSVM, Naive Bayes, Random Forest, XGBoost and LightGBM, tuned with Optuna. Then fine-tune DistilBERT on Google Colab (GPU). The transformer is only deployable if an ONNX/quantized export meets the size and latency limits.
- **Metric:** choose models by PR-AUC on the fraud class. Also report F1, precision and recall at a tuned decision threshold. Don't use accuracy: about 95% of rows are real, so it is misleading.
- **MLOps:** DagsHub hosts MLflow tracking, the MLflow Model Registry and the DVC remote. The repos on GitHub and DagsHub are public, but **DagsHub only serves DVC data (and its web UI) to signed-in users, even for public repos**. Verified 2026-10-18: anonymous requests to real DVC files return 401, including on DagsHub's own official repos. The client chose to require a free DagsHub account for reproduction rather than add a public mirror. Don't promise anonymous `dvc pull` anywhere.
- **Serving:** Render free tier (512 MB RAM, ~50 s cold start). The API loads `models:/fake-job-detector@champion` from the registry at startup.
- **App features:** prediction with an explanation, and a feedback loop. Predictions and feedback are stored in Neon Postgres.
- **Monitoring:** prediction logging, plus Evidently drift reports on a schedule.
- **CI/CD:** full GitHub Actions.

### Pipeline (`dvc.yaml`)
Stages, in order: `ingest → validate (pandera) → clean_dedupe → split → featurize → train_classical (foreach model) → train_transformer → evaluate → select_and_register → drift_reference`. A `feedback_export` stage will later turn feedback rows into a versioned dataset.
- Remove duplicates **before** the split. EMSCAD contains near-duplicate postings, which would otherwise leak between train and test.
- Use a **stratified, group-aware** split seeded from `params.yaml` (decided with the client after the EDA). About 25% of val/test postings shared a description template with train under a random split, so postings linked by a shared description opening or company profile form a group (`src/fjd/data/groups.py`), and whole groups go to one split. The processed parquet files keep a `group_id` column. **All cross-validation (Optuna tuning etc.) must be group-aware** (`StratifiedGroupKFold` on `group_id`), or the leak comes back. Metrics will be lower than published EMSCAD random-split numbers; that's expected.
- Track metrics and PR curves with `dvc metrics` / `dvc plots`.
- `train_transformer` is `frozen` by default, so local `dvc repro` uses the ONNX artifact trained on Colab and pulled from DVC. `dvc repro -f train_transformer` retrains it.
- Notebooks call the same `src/fjd` functions as the pipeline, so notebook runs and pipeline runs can be compared in MLflow.

### Tracking and registry
- Every run logs to MLflow: params, metrics, PR curve, confusion matrix, git SHA, the data hash from `dvc.lock`, and the model.
- The registered model is `fake-job-detector`. The model is a pyfunc that wraps preprocessing, the model and the threshold.
- `select_and_register` registers the best run and sets the alias `@challenger`.
- Promotion to `@champion` is a deliberate step (a script plus manual approval). It happens only if the challenger beats the current champion on PR-AUC and passes the size and latency limits.

### Backend (`backend/`, FastAPI on Render, Docker)
- **Endpoints:** `POST /predict` returns the label, probability, threshold and top explanation terms. Also `POST /feedback`, `GET /health` and `GET /model-info`.
- **Explanations:** coefficient × TF-IDF for linear models and TreeSHAP for boosted trees. Token occlusion is used only if the transformer becomes champion.
- **Storage:** SQLAlchemy + Alembic, with `predictions` and `feedback` tables in Neon.
- **Security:** CORS is restricted to the Vercel domain, and requests are rate-limited. The registry is read with a read-only DagsHub token.

### Frontend (`frontend/`, Next.js + TypeScript + Tailwind on Vercel)
- A form with one field per posting section, or a single box to paste the whole posting.
- A result card with the verdict, a probability gauge, highlighted suspicious terms, the model version, and 👍/👎 feedback buttons.
- Handles Render's cold start: pings `/health` on page load and shows a "warming up" state.

### CI/CD and monitoring (`.github/workflows/`)
- `ci.yml`: ruff, mypy and pytest, plus frontend lint, typecheck and build.
- `pipeline-pr.yml`: runs on PRs that touch `src/`, `params.yaml` or `dvc.yaml`. It runs `dvc pull` and `dvc repro` for the classical models, posts a CML comment with `dvc metrics diff` against `main`, and asserts that the reproduced classical metrics equal the committed ones.
- `deploy.yml`: on pushes to `main`, calls the Render deploy hook. Vercel deploys through its Git integration.
- `drift.yml`: runs weekly and on demand. It exports recent predictions from Neon, runs an Evidently drift report against the training reference, uploads the report, logs it to MLflow, and opens an issue if drift exceeds the threshold.

### Phases
1. Scaffold (done)
2. Data stages + notebook `01_eda`
3. Classical models: notebooks `02_baselines` and `03_classical_tuning`, pipeline stages, MLflow logging
4. Transformer on Colab: `04_transformer_colab`, ONNX export, size and latency check
5. Comparison and registry: `05_model_comparison`, select/register/promote
6. Backend + Neon + Docker → Render
7. Frontend → Vercel
8. CI/CD, monitoring, and a reproduction test from a fresh clone

## Credentials

Reproducing from a clone needs only the user's own free DagsHub token (`DAGSHUB_USER`/`DAGSHUB_TOKEN` in `.env`, applied with `uv run poe dvc-auth`, which writes the git-ignored `.dvc/config.local`). Write access (DagsHub/MLflow, Kaggle, backend `DATABASE_URL`) uses `.env`; see `.env.example` for the variables. `.env` exists locally — don't read, print, or commit it.
