import subprocess
from pathlib import Path

from dvc.repo import Repo

from fjd.utils.config import PROJECT_ROOT


def test_every_committed_dvc_file_is_visible_to_dvc() -> None:
    """DVC silently skips git-ignored .dvc files, so their data is never pushed or pulled."""
    tracked = subprocess.run(
        ["git", "ls-files", "*.dvc"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    with Repo(str(PROJECT_ROOT)) as repo:
        indexed = {Path(s.path).resolve() for s in repo.index.stages}
    missing = [f for f in tracked if (PROJECT_ROOT / f).resolve() not in indexed]
    assert tracked, "expected at least one committed .dvc file"
    assert not missing, f"DVC does not see these .dvc files (git-ignored?): {missing}"
