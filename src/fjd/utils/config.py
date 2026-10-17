"""Load params.yaml and resolve project paths."""

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def load_params(path: Path = PARAMS_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        params: dict[str, Any] = yaml.safe_load(f)
    return params
