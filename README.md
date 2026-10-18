# Fake Job Posting Detector

Classifies job postings as real or fraudulent. The ML pipeline is versioned with DVC, experiments and the
model registry live on DagsHub (MLflow), the API runs on Render (FastAPI), and the UI runs on Vercel (Next.js).

> Status: phase 2 of 8. The data pipeline (ingest → validate → dedupe → split) is reproducible
> with the steps below. Model training, the API and the UI are not built yet.

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
- `dvc.yaml` / `dvc.lock` — the versioned pipeline (data stages so far)
- `notebooks/` — exploration (outputs stripped on commit)
- `backend/`, `frontend/` — the deployed app (empty for now)
