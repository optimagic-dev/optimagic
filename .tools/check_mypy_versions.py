import re
import sys
from pathlib import Path

import yaml


def _get_mypy_version_from_precommit_config() -> str:
    config = yaml.safe_load(Path(".pre-commit-config.yaml").read_text())
    mypy_config = [
        hook
        for hook in config["repos"]
        if hook["repo"] == "https://github.com/pre-commit/mirrors-mypy"
    ][0]
    match = re.search(r"v([\d.]+)", mypy_config["rev"])  # Remove "v" prefix
    if match:
        return match.group(1)
    raise ValueError("Mypy version not found in pre-commit config.")


def _get_mypy_version_from_conda_environment() -> str:
    config = yaml.safe_load(Path("environment.yml").read_text())
    mypy_line = [dep for dep in config["dependencies"] if "mypy" in dep][0]
    match = re.search(r"mypy=([\d.]+)", mypy_line)
    if match:
        return match.group(1)
    raise ValueError("Mypy version not found in conda environment.")


def main() -> None:
    v_precommit = _get_mypy_version_from_precommit_config()
    v_conda = _get_mypy_version_from_conda_environment()
    if v_precommit != v_conda:
        print(
            f"Error: Mypy versions do not match:\n"
            f"  Pre-commit config: {v_precommit}\n"
            f"  Conda environment: {v_conda}\n"
        )
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
