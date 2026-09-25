#!/usr/bin/env python
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# sys.executable guarantees we stay inside the pre‑commit venv
PYTHON = [sys.executable]


def run(cmd: list[str], **kwargs: Any) -> None:
    subprocess.check_call(cmd, cwd=ROOT, **kwargs)


def install_optimagic_locally() -> None:
    # Always (re)install so that dependencies added after the hook environment was
    # created are picked up. This is fast if nothing changed.
    run(["uv", "pip", "install", "--quiet", "--python", sys.executable, "-e", "."])


def main() -> int:
    install_optimagic_locally()
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
