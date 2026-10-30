"""Check that a pipeline run reproduced the committed metrics (`python -m fjd.utils.repro_check`).

CI runs it after `dvc repro -f` (`.github/workflows/pipeline-pr.yml`, `reproduce.yml`). Every
metrics file under `reports/` (top level and `train/`) must equal the version committed at
`--rev` (default HEAD), value for value. The exception is the transformer's re-scored
metrics in `reports/metrics.json`: they come from a model trained on another machine, so only
their PR-AUC is compared, within `transformer.repro_tolerance` (as `evaluate` does for val).
"""

import argparse
import json
import subprocess
import sys
from typing import Any

from fjd.data.io import REPORTS
from fjd.utils.config import PROJECT_ROOT, load_params

METRICS_GLOBS = ["*.json", "train/*.json"]
TRANSFORMER = "transformer"  # its name in reports (`fjd.models.evaluate.TRANSFORMER`)


def committed(rev: str, path: str) -> Any:
    result = subprocess.run(
        ["git", "show", f"{rev}:{path}"], cwd=PROJECT_ROOT, capture_output=True, text=True
    )
    return json.loads(result.stdout) if result.returncode == 0 else None


def differences(old: Any, new: Any, prefix: str = "") -> list[str]:
    """Paths (`a.b.c: old -> new`) where two JSON values differ."""
    if isinstance(old, dict) and isinstance(new, dict):
        out = []
        for key in sorted(old.keys() | new.keys()):
            out += differences(old.get(key), new.get(key), f"{prefix}{key}.")
        return out
    return [] if old == new else [f"{prefix.rstrip('.')}: {old} -> {new}"]


def check(rev: str = "HEAD") -> list[str]:
    """Every difference from the committed metrics; empty if the run reproduced them."""
    tolerance = load_params()["transformer"]["repro_tolerance"]
    paths = sorted(
        p.relative_to(PROJECT_ROOT).as_posix() for g in METRICS_GLOBS for p in REPORTS.glob(g)
    )
    problems = []
    for path in paths:
        old, new = committed(rev, path), json.loads((PROJECT_ROOT / path).read_text("utf-8"))
        if old is None:
            problems.append(f"{path}: not committed at {rev}")
            continue
        if path == "reports/metrics.json":
            old_t, new_t = old.pop(TRANSFORMER, {}), new.pop(TRANSFORMER, {})
            for split in old_t.keys() | new_t.keys():
                a = old_t.get(split, {}).get("pr_auc")
                b = new_t.get(split, {}).get("pr_auc")
                if a is None or b is None or abs(a - b) > tolerance:
                    problems.append(f"{path}: {TRANSFORMER}.{split}.pr_auc: {a} -> {b}")
        problems += [f"{path}: {d}" for d in differences(old, new)]
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--rev", default="HEAD", help="git revision to compare with")
    rev = parser.parse_args().rev
    problems = check(rev)
    if problems:
        print(f"Metrics differ from {rev}:", *problems, sep="\n  ")
        sys.exit(1)
    print(f"All metrics under reports/ match {rev}.")


if __name__ == "__main__":
    main()
