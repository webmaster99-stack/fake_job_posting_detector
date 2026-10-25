"""Consistency checks between dvc.yaml, params.yaml and the code."""

import ast
import re
from pathlib import Path

import yaml

from fjd.models.classical import MODELS
from fjd.models.evaluate import TRANSFORMER
from fjd.utils.config import PROJECT_ROOT, load_params

SRC = PROJECT_ROOT / "src"


def _module_file(module: str) -> Path | None:
    base = SRC.joinpath(*module.split("."))
    for path in (base.with_suffix(".py"), base / "__init__.py"):
        if path.exists():
            return path
    return None


def fjd_imports(module: str, seen: set[Path] | None = None) -> set[Path]:
    """Every non-empty `src/fjd` file that `module` imports, directly or transitively."""
    seen = set() if seen is None else seen
    path = _module_file(module)
    if path is None or path in seen:
        return seen
    seen.add(path)
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            # `from fjd.a import b` may import the submodule fjd.a.b
            names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
        for name in names:
            if name.startswith("fjd."):
                fjd_imports(name, seen)
    return {p for p in seen if p.stat().st_size > 0}


def _stages() -> dict[str, dict]:
    stages = yaml.safe_load((PROJECT_ROOT / "dvc.yaml").read_text(encoding="utf-8"))["stages"]
    out = {}
    for name, stage in stages.items():
        if "foreach" in stage:  # check the template once; ${key} doesn't affect imports
            stage = stage["do"]
        out[name] = stage
    return out


def test_every_stage_lists_the_fjd_modules_it_imports() -> None:
    """Otherwise editing a shared helper would not invalidate the stage (see CLAUDE.md)."""
    problems = []
    for name, stage in _stages().items():
        module = re.fullmatch(r"python -m (fjd[\w.]*)(?: .*)?", stage["cmd"])
        assert module, f"stage {name}: expected `python -m fjd.<module>`, got {stage['cmd']!r}"
        deps = {(PROJECT_ROOT / d).resolve() for d in stage["deps"]}
        for path in sorted(fjd_imports(module.group(1))):
            if path.resolve() not in deps:
                problems.append(f"{name}: {path.relative_to(PROJECT_ROOT).as_posix()}")
    assert not problems, "missing deps in dvc.yaml:\n" + "\n".join(problems)


def test_classical_params_match_the_model_registry() -> None:
    assert set(load_params()["classical"]) == set(MODELS)


def test_pr_curve_plot_lists_every_model() -> None:
    plots = yaml.safe_load((PROJECT_ROOT / "dvc.yaml").read_text(encoding="utf-8"))["plots"]
    files = next(p["test_pr_curve"]["y"] for p in plots if "test_pr_curve" in p)
    assert {Path(f).stem for f in files} == set(load_params()["classical"]) | {TRANSFORMER}


def test_train_transformer_is_frozen() -> None:
    """`dvc repro` must not try to fine-tune on a CPU; the model comes from DVC."""
    assert _stages()["train_transformer"].get("frozen") is True
