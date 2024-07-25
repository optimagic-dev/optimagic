import toml
from pathlib import Path

file_path = Path("pyproject.toml")

with file_path.open("r") as f:
    config = toml.load(f)

config["project"]["name"] = "estimagic"

with file_path.open("w") as f:
    toml.dump(config, f)
