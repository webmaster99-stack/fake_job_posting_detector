"""One-time raw data download from Kaggle (maintainer only).

Everyone else gets the raw data with `dvc pull`. After running this:
    uv run dvc add data/raw/emscad.csv && uv run dvc push
"""

import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from fjd.utils.config import PROJECT_ROOT

KAGGLE_DATASET = "shivamb/real-or-fake-fake-jobposting-prediction"
KAGGLE_FILE = "fake_job_postings.csv"
DEST = PROJECT_ROOT / "data" / "raw" / "emscad.csv"


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")  # KAGGLE_USERNAME / KAGGLE_KEY
    from kaggle.api.kaggle_api_extended import KaggleApi  # authenticates on import in some versions

    api = KaggleApi()
    api.authenticate()
    with tempfile.TemporaryDirectory() as tmp:
        api.dataset_download_files(KAGGLE_DATASET, path=tmp, unzip=True)
        DEST.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(tmp) / KAGGLE_FILE, DEST)
    print(f"Wrote {DEST.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
