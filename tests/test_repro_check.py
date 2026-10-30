import json
from typing import Any

import pytest

from fjd.utils import repro_check
from fjd.utils.config import PROJECT_ROOT, load_params

TOLERANCE = load_params()["transformer"]["repro_tolerance"]


def test_differences_lists_changed_added_and_removed_values() -> None:
    old = {"a": {"f1": 0.5, "tp": 3}, "b": 1}
    new = {"a": {"f1": 0.6, "tp": 3}, "c": 2}
    assert repro_check.differences(old, new) == ["a.f1: 0.5 -> 0.6", "b: 1 -> None", "c: None -> 2"]


@pytest.mark.parametrize(
    ("model", "key", "shift", "flagged"),
    [
        ("logreg", "f1", 1e-6, True),  # classical models must match exactly
        ("transformer", "pr_auc", TOLERANCE * 2, True),
        ("transformer", "pr_auc", TOLERANCE / 2, False),
        ("transformer", "tp", 1, False),  # only the transformer's PR-AUC is compared
    ],
)
def test_check_compares_with_the_committed_metrics(
    monkeypatch: pytest.MonkeyPatch, model: str, key: str, shift: float, flagged: bool
) -> None:
    def committed(rev: str, path: str) -> Any:
        data = json.loads((PROJECT_ROOT / path).read_text("utf-8"))  # the working tree...
        if path == "reports/metrics.json":
            data[model]["val"][key] += shift  # ...with one value changed
        return data

    monkeypatch.setattr(repro_check, "committed", committed)
    problems = repro_check.check()
    assert bool(problems) == flagged
    assert all(p.startswith(f"reports/metrics.json: {model}.val.") for p in problems)
