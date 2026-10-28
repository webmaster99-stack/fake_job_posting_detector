# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Binary classifier for real vs. fraudulent job postings (EMSCAD dataset, target column `fraudulent`). Phases 2 (data), 3 (classical models), 4 (transformer), 5 (selection and registry) and 6 (backend) are built. `dvc.yaml` has `ingest → validate → clean_dedupe → split → featurize → train_classical@<model> → train_transformer (frozen) → evaluate → select_and_register` (data code in `src/fjd/data/`, canonical columns and the pandera schema in `src/fjd/data/schema.py`; features in `src/fjd/features/`; models in `src/fjd/models/`). Notebooks `01_eda` to `05_model_comparison` exist. `backend/` (FastAPI, see Architecture) is live on Render at https://fake-job-detector-api-dud2.onrender.com (deployed from `main`, storing predictions and feedback in Neon), serving `fake-job-detector` v1 `@champion` (linear SVM). `frontend/` (Next.js, phase 7) is built but not yet deployed to Vercel; see `frontend/README.md`. `drift_reference`, `src/fjd/monitoring` and `.github/workflows/` don't exist yet. Don't assume CI exists — check first.

- Raw data: `data/raw/emscad.csv` (tracked by `data/raw/emscad.csv.dvc`). Maintainer re-download: `uv run python -m fjd.data.download` (Kaggle creds from `.env`).
- Every DVC stage is a `python -m fjd.<module>` entry point. Its `deps` must list the stage module **and** every shared helper it imports (`io.py`, `schema.py`, `utils/*.py`), or edits to those won't invalidate it.
  `tests/test_pipeline.py` enforces this by tracing imports. It can't see imports that happen while unpickling, so list those by hand (e.g. `train_classical` lists `features/build.py`).
- Write text artifacts with LF line endings (see `fjd.data.io.write_json`) so `dvc.lock` hashes match across Windows and Linux. Parquet is written with `index=False`.
- Anything that gets pickled (featurizer functions, models) must live in an importable module, never in a `python -m` stage module, or it pickles as `__main__.<name>`. Renaming or moving `src/fjd/features/build.py` functions breaks every saved model.

## Commands

Python 3.12 only (`>=3.12,<3.13`), managed with `uv`. Tasks are defined via poethepoet in `pyproject.toml`:

```bash
uv sync                      # install from uv.lock (dev + notebooks + backend groups by default)
uv sync --group transformer  # + torch/transformers/onnx: transformer training and its tests (skipped without)
uv run poe test              # pytest (testpaths = tests/)
uv run pytest tests/test_utils.py::test_set_seed_is_deterministic   # single test
uv run poe lint              # ruff check .
uv run poe fmt               # ruff format .
uv run poe typecheck         # mypy src backend
uv run poe reproduce         # dvc pull -> dvc repro -> dvc metrics show
uv run poe tune logreg --trials 20   # Optuna, group-aware CV; omit model names to tune all
uv run poe migrate           # alembic upgrade head on DATABASE_URL (default backend/local.db)
uv run poe api               # FastAPI on :8000 with --reload
uv run pre-commit run --all-files
```

Frontend (`frontend/`, Node 24, npm): `npm run dev`, `npm run lint`, `npm run typecheck` (runs `next typegen` first), `npm run build`. Next.js 16: read `frontend/AGENTS.md` and the bundled docs in `node_modules/next/dist/docs/` before changing framework-level code.

Ruff: line length 100, rules `E,F,I,B,UP,SIM`. Pre-commit also runs nbstripout (notebook outputs are stripped on commit), blocks files >1000 KB, and detects private keys.

## Architecture

Planned stack: DVC pipeline for data/models → MLflow tracking + model registry on DagsHub (also the DVC remote) → FastAPI backend on Render → Next.js frontend on Vercel.

- `src/fjd/` is the single importable package (built with hatchling) shared by notebooks, DVC stages, and the backend. Put reusable logic here, not in notebooks or `backend/`.
- `params.yaml` is the single source of truth for every seed, split ratio, and hyperparameter; read it via `fjd.utils.config.load_params()`. Don't hardcode tunables. DVC stages will depend on it, so edits invalidate downstream stages.
- `fjd.utils.config.PROJECT_ROOT` is resolved as `parents[3]` of `config.py`; moving that file breaks path resolution.
- Call `fjd.utils.seed.set_seed(params["seed"])` at entry points for reproducibility (seeds Python, NumPy, and torch if installed).
- Model selection (`params.yaml: selection`) optimizes `pr_auc` under deployment constraints: model ≤300 MB and p95 latency ≤300 ms, measured locally on one CPU thread. Candidate models must respect these to be servable on Render. Live latency on Render's free tier is higher (~500–700 ms of model time per `/predict`, ~0.7–1.6 s per request, measured 2026-10-27), and the client accepted that for now: the 300 ms target is relaxed for the live service, not for selection, so don't change `params.yaml` to match it.
- Classical models: `featurize` fits one shared featurizer (`fjd.features.build`: TF-IDF over all text + tabular flags/counts + one-hot categoricals, sparse and non-negative) on train. Each `train_classical@<model>` saves `models/classical/<model>.joblib`, a sklearn `Pipeline(features, model)` that predicts from raw canonical columns, writes its F1-optimal val threshold to `models/classical/<model>.threshold.json` (an artifact, not a metric: its last digits differ across OSes), and its val metrics to `reports/train/<model>.json`. Rounded metrics must be byte-identical between Windows and Linux (check in Docker after changing a model or its params). Two settings exist only for that: XGBoost's row/column sampling is fixed off (it draws differently per C++ standard library, and passing `subsample`/`colsample_*` raises), and logreg uses `tol=1e-6`. LightGBM (`deterministic=True`) and the random forest may sample. Model constructors and fixed settings (class weights, seeds, booster threads, χ² `k_features` selection for tree models) are in `fjd.models.classical`. LinearSVC outputs margins, not probabilities.
- Transformer: `fjd.models.train_transformer` (torch, `transformer` dependency group) fine-tunes DistilBERT (pinned Hub `revision` in `params.yaml: transformer`). It keeps the best epoch by val PR-AUC and exports `models/transformer/` (`model.onnx` int8, `tokenizer.json`, `text.json`, `threshold.json`). `fjd.models.transformer` scores that directory with onnxruntime + tokenizers only (no torch), for the pipeline and the API. Its `TransformerClassifier.predict_proba` takes raw canonical columns. `posting_text` turns a posting into model input, and changing it changes what a trained model expects. Postings are scored **one at a time**: dynamic int8 quantization scales activations per tensor, so a padded batch would change each posting's score. Weights are quantized as uint8 (`QUInt8`), because int8-weight kernels can saturate on AVX2 CPUs without VNNI and give CPU-dependent scores. The threshold and reported val metrics come from the exported model. `evaluate` requires the re-scored val PR-AUC to match `reports/train/transformer.json` within `transformer.repro_tolerance`. Classical models must match exactly.
- Deployability (`fjd.models.deployability`): model size, plus p50/p95 latency of single postings on one CPU thread, against `params.yaml: selection`. Latency depends on the machine, so it is logged to MLflow (`evaluate`) and the train report, never compared as a reproducible metric.
- Registry (`fjd.models.registry`, `select`, `promote`): `select_and_register` ranks models by val `selection.metric` from `reports/metrics.json`, measures deployability in rank order (one posting, one thread, as served) and takes the first within the limits. It wraps the winner in the pyfunc `FraudDetectorModel` (model + val-tuned threshold + Platt calibration fitted on val, decided with the client so LinearSVC margins and class-weighted probabilities become real P(fraud); the sigmoid is monotone, so metrics and labels are unchanged), registers it as `fake-job-detector` and sets `@challenger`. Predict returns `probability`, `fraud` (0/1) and `threshold` (as a probability); the API gets the raw model for explanations via `unwrap_python_model().detector`. `reports/selection.json` holds only reproducible values; calibration and latencies go to MLflow. Every run of the stage registers a new version (locally in `mlflow.db` unless `MLFLOW_TRACKING_URI` is set). `uv run poe promote` moves `@champion`, after a prompt, only if the challenger is within the limits and beats the champion's val metric (read from their `select_and_register` runs).
- Backend (`backend/`, package `backend`, run with `uv run poe api`): `create_app()` factory (`uvicorn backend.main:create_app --factory`); config from env in `backend/settings.py`. At startup it resolves `MODEL_URI` (default `models:/fake-job-detector@champion`) to a version, loads it and warms it up; moving `@champion` needs a restart. `/predict` runs `fjd.serving.prepare` (clean-stage whitespace normalization + EMSCAD-style `#URL_…#`/`#EMAIL_…#`/`#PHONE_…#` anonymization) then the `Detector`, plus `fjd.models.explain` (linear: weight × value; XGBoost/LightGBM: TreeSHAP via `pred_contrib`; random forest/transformer: none). Predictions (the prepared posting as JSON) and feedback (one row per prediction, upserted) go to Postgres through SQLAlchemy; if the insert fails, `/predict` still answers with `prediction_id: null`. Schema changes need an Alembic migration (`backend/migrations`, `uv run alembic -c backend/alembic.ini revision --autogenerate`); `tests/test_backend.py` fails if models and migrations differ. Without `DATABASE_URL` it uses `backend/local.db`. Rate limiting is in-memory per IP (one worker). Docker: `backend/Dockerfile`, built from the repo root, installs only the `backend` dependency group, runs `alembic upgrade head` then uvicorn. `render.yaml` defines the Render service (Frankfurt, `autoDeploy: false`; the phase 8 deploy hook triggers deploys). The live service (`srv-dbg87v2jnfac73ficjg0`) was created through the Render API, not from the Blueprint. Its health check is `/health`, and it reads the registry with DagsHub credentials the maintainer set in the Render dashboard. `ALLOWED_ORIGINS` is `http://localhost:3000` on purpose until the Vercel frontend exists (phase 7); leave it. The Neon project is `fake-job-postings-detector` (aws-eu-central-1). `DATABASE_URL` uses its direct endpoint, not the `-pooler` one.
- Optuna tuning (`fjd.models.tune`, notebook 03) is deliberately **not** a DVC stage. Its best params are copied by hand into `params.yaml: classical`, and records go in `reports/tuning/`. Studies are saved in the git-ignored `optuna.db` under `<model>-<fingerprint>` (code in `tune.py`, `classical.py`, `cv.py`, `build.py`, plus seed, features, folds and train data md5), so re-running `poe tune <model>` after an interruption resumes it, and any change starts a fresh study. The sampler is re-seeded per completed trial, so a resumed study equals an uninterrupted one. Run one tuning process at a time.
- MLflow (`fjd.utils.tracking.setup_mlflow`): logs to DagsHub only if `MLFLOW_TRACKING_URI` is set (env or `.env`), otherwise to a local `mlflow.db` + `mlartifacts/`. `evaluate` logs one parent run with a nested run per model. Models are logged with cloudpickle, because skops takes minutes per model on the TF-IDF vocabulary. When testing locally, set `MLFLOW_TRACKING_URI=` (empty) to avoid writing runs to DagsHub. In MLflow 3 a logged model is a separate entity: load it as `models:/<model_id>` (find it with `mlflow.search_logged_models(filter_string="source_run_id = '<run_id>'")`), not `runs:/<run_id>/model`. `setup_mlflow` switches stdout to UTF-8, because MLflow's remote-run messages contain emoji that crash cp1252 output on Windows.
- Data and model artifacts (`data/raw|interim|processed/*`, `models/*`) are DVC-tracked and git-ignored; only `.gitkeep` files are committed. `.dvc/config` has `autostage = true`, so DVC commands auto-`git add` their `.dvc`/lock files.

## Build plan

Agreed with the client before implementation. Everything below is **planned**, not built, unless the Project status section says otherwise. Ask before changing any of these decisions.

**Hard requirement:** a stranger can clone the repo, run `uv sync && uv run poe dvc-auth && uv run dvc pull && uv run dvc repro`, and get identical data, metrics and models. The only credential needed is their own free DagsHub account/token (see below). Classical models must match exactly on CPU. The transformer must match within tolerance (|ΔPR-AUC| ≤ 0.005).

### Decisions
- **Data:** start with EMSCAD (~17.9k rows, ~4.8% fraud). The maintainer adds the raw file to DVC once, from Kaggle. Everyone else gets it with `dvc pull`. More datasets get added later through the `datasets:` list in `params.yaml`; the ingest stage merges them and adds a `source` column.
- **Models:** start with TF-IDF features plus tabular ones (missing-field flags, `has_company_logo`, …). Compare LogReg, LinearSVM, Naive Bayes, Random Forest, XGBoost and LightGBM, tuned with Optuna. Then fine-tune DistilBERT on Google Colab (GPU). The transformer is only deployable if an ONNX/quantized export meets the size and latency limits.
- **Metric:** choose models by PR-AUC on the fraud class. Also report F1, precision and recall at a tuned decision threshold. Don't use accuracy: about 95% of rows are real, so it is misleading.
- **MLOps:** DagsHub hosts MLflow tracking, the MLflow Model Registry and the DVC remote. The repos on GitHub and DagsHub are public, but **DagsHub only serves DVC data (and its web UI) to signed-in users, even for public repos**. Verified 2026-10-18: anonymous requests to real DVC files return 401, including on DagsHub's own official repos. The client chose to require a free DagsHub account for reproduction rather than add a public mirror. Don't promise anonymous `dvc pull` anywhere.
- **Serving:** Render free tier (512 MB RAM, ~50 s cold start). The API loads `models:/fake-job-detector@champion` from the registry at startup. Decided 2026-10-27: live latency above 300 ms is accepted for now (see Architecture → Model selection).
- **App features:** prediction with an explanation, and a feedback loop. Predictions and feedback are stored in Neon Postgres.
- **Monitoring:** prediction logging, plus Evidently drift reports on a schedule.
- **CI/CD:** full GitHub Actions.

### Pipeline (`dvc.yaml`)
Stages, in order: `ingest → validate (pandera) → clean_dedupe → split → featurize → train_classical (foreach model) → train_transformer → evaluate → select_and_register → drift_reference`. A `feedback_export` stage will later turn feedback rows into a versioned dataset.
- Remove duplicates **before** the split. EMSCAD contains near-duplicate postings, which would otherwise leak between train and test.
- Use a **stratified, group-aware** split seeded from `params.yaml` (decided with the client after the EDA). About 25% of val/test postings shared a description template with train under a random split, so postings linked by a shared description opening or company profile form a group (`src/fjd/data/groups.py`), and whole groups go to one split. The processed parquet files keep a `group_id` column. **All cross-validation (Optuna tuning etc.) must be group-aware** (`StratifiedGroupKFold` on `group_id`), or the leak comes back. Metrics will be lower than published EMSCAD random-split numbers; that's expected.
- Track metrics and PR curves with `dvc metrics` / `dvc plots`.
- `train_transformer` is `frozen`, so local `dvc repro` uses the ONNX artifact trained on Colab and pulled from DVC. DVC never runs a frozen stage's command, not even with `dvc repro -f`. To retrain, run `notebooks/04_transformer_colab.ipynb` on Colab. It runs `python -m fjd.models.train_transformer` and downloads the outputs. Then record them locally with `dvc commit -f train_transformer`.
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
2. Data stages + notebook `01_eda` (done)
3. Classical models: notebooks `02_baselines` and `03_classical_tuning`, pipeline stages, MLflow logging (done)
4. Transformer on Colab: `04_transformer_colab`, ONNX export, size and latency check (done)
5. Comparison and registry: `05_model_comparison`, select/register/promote (done)
6. Backend + Neon + Docker → Render (done)
7. Frontend → Vercel (frontend built; Vercel deploy pending)
8. CI/CD, monitoring, and a reproduction test from a fresh clone

## Credentials

Reproducing from a clone needs only the user's own free DagsHub token (`DAGSHUB_USER`/`DAGSHUB_TOKEN` in `.env`, applied with `uv run poe dvc-auth`, which writes the git-ignored `.dvc/config.local`). Write access (DagsHub/MLflow, Kaggle, backend `DATABASE_URL`) uses `.env`; see `.env.example` for the variables. `.env` exists locally — don't read, print, or commit it.
