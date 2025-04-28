#!/usr/bin/env python
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# sys.executable guarantees we stay inside the pre‑commit venv
PYTHON = [sys.executable]
# "-m" lets us call std‑lib modules (e.g. pip) the same way
PYTHON_MINUS_M = [*PYTHON, "-m"]


def run(cmd: list[str], **kwargs: Any) -> None:
    subprocess.check_call(cmd, cwd=ROOT, **kwargs)


def ensure_optimagic_is_locally_installed() -> None:
    try:
        run(PYTHON_MINUS_M + ["pip", "show", "optimagic"], stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        run(PYTHON_MINUS_M + ["pip", "install", "-e", "."])


def main() -> int:
    ensure_optimagic_is_locally_installed()
    run(PYTHON + [".tools/create_algo_selection_code.py"])

    ruff_args = [
        "--silent",
        "--config",
        "pyproject.toml",
        "src/optimagic/algorithms.py",
    ]
    run(["ruff", "format", *ruff_args])
    run(["ruff", "check", "--fix", *ruff_args])
    return 0  # explicit success code


if __name__ == "__main__":
    sys.exit(main())
