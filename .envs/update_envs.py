from copy import deepcopy
from pathlib import Path


def _keep_line(line: str, flag: str) -> bool:
    """Return True if line contains flag and does not include a comment.

    Examples:
    >>> assert _keep_line("  - jax  # tests", "tests") is True
    >>> assert _keep_line("name: env", "tests") is True
    >>> assert _keep_line("  - jax  # run", "tests") is False

    """
    return flag in line or "#" not in line


def main():
    lines = Path("environment.yml").read_text().splitlines()

    # create standard testing environments

    test_env = [line for line in lines if _keep_line(line, "tests")]
    test_env.append("      - -e ../")  # add local installation

    # find index to insert additional dependencies
    _insert_idx = [i for i, line in enumerate(lines) if "dependencies:" in line][0] + 1

    ## linux
    test_env_linux = deepcopy(test_env)
    test_env_linux.insert(_insert_idx, "  - pygmo")
    test_env_linux.insert(_insert_idx, "  - jax")

    ## test environment others
    test_env_others = deepcopy(test_env)
    test_env_others.insert(_insert_idx, "  - cyipopt")

    # create docs testing environment

    docs_env = [line for line in lines if _keep_line(line, "docs")]
    docs_env.append("      - -e ../")  # add local installation

    # write environments
    for name, env in zip(["linux", "others"], [test_env_linux, test_env_others]):
        # Specify newline to avoid wrong line endings on Windows.
        # See: https://stackoverflow.com/a/23434608
        with Path(f".envs/testenv-{name}.yml").open("w", newline="\n") as file:
            file.write("\n".join(env) + "\n")


if __name__ == "__main__":
    main()
