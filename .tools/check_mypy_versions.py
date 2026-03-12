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
    match = re.search(r"v([\d.]+)", mypy_config["rev"])
    if match:
        return match.group(1)
    raise ValueError("Mypy version not found in pre-commit config.")


def _get_mypy_version_from_pixi() -> str:
    text = Path("pyproject.toml").read_text()
    match = re.search(r'mypy\s*=\s*"==([\d.]+)"', text)
    if match:
        return match.group(1)
    raise ValueError("Mypy version not found in pyproject.toml pixi config.")


def main() -> None:
    v_precommit = _get_mypy_version_from_precommit_config()
    v_pixi = _get_mypy_version_from_pixi()
    if v_precommit != v_pixi:
        print(
            f"Error: Mypy versions do not match:\n"
            f"  Pre-commit config: {v_precommit}\n"
            f"  Pixi config:      {v_pixi}\n"
        )
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
