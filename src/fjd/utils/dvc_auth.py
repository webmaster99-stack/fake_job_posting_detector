"""Configure DagsHub credentials for the DVC remote (`uv run poe dvc-auth`).

DagsHub requires a signed-in user to read DVC data, even from public repos, so everyone
(including strangers reproducing the project) needs a free DagsHub account and token.
Reads DAGSHUB_USER and DAGSHUB_TOKEN from the environment or `.env` and writes them to
`.dvc/config.local`, which is git-ignored. The token is never printed.
"""

import os
import subprocess
import sys

from dotenv import load_dotenv

from fjd.utils.config import PROJECT_ROOT

REMOTE = "origin"


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    user, token = os.environ.get("DAGSHUB_USER"), os.environ.get("DAGSHUB_TOKEN")
    if not user or not token:
        sys.exit("Set DAGSHUB_USER and DAGSHUB_TOKEN in .env (see .env.example).")
    for key, value in [("auth", "basic"), ("user", user), ("password", token)]:
        subprocess.run(
            ["dvc", "remote", "modify", REMOTE, "--local", key, value],
            cwd=PROJECT_ROOT,
            check=True,
        )
    print(f"DVC remote '{REMOTE}' configured for DagsHub user {user}.")


if __name__ == "__main__":
    main()
