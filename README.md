# Fake Job Posting Detector

Classifies job postings as real or fraudulent. The ML pipeline is versioned with DVC, experiments and the
model registry live on DagsHub (MLflow), the API runs on Render (FastAPI), and the UI runs on Vercel (Next.js).

> Status: scaffolding (phase 1 of 8). The DVC pipeline, DagsHub remote, API and UI are not set up yet,
> so the reproduce steps below describe the target workflow and won't work until they are.

## Reproduce from a fresh clone

No accounts or credentials are needed. Data and models are pulled anonymously from the public DagsHub remote.

```bash
git clone <repo-url> && cd fake_job_posting_detector
uv sync                # exact dependency versions from uv.lock (Python 3.12)
uv run dvc pull        # data + pipeline outputs
uv run dvc repro       # re-run the pipeline; no-op if nothing changed
uv run dvc metrics show
```

Or with one command: `uv run poe reproduce`.

## Dev tasks

| Command | What it does |
| --- | --- |
| `uv run poe test` | pytest |
| `uv run poe lint` / `fmt` | ruff |
| `uv run poe typecheck` | mypy |

## Layout

- `src/fjd/` — the importable package shared by notebooks, the DVC pipeline and the API
- `params.yaml` — every seed, split ratio and hyperparameter
- `dvc.yaml` / `dvc.lock` — the versioned pipeline (not created yet)
- `notebooks/` — exploration (outputs stripped on commit)
- `backend/`, `frontend/` — the deployed app (empty for now)
