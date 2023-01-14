from copy import deepcopy
from pathlib import Path


def _keep_line(line, flag):
    return flag in line or "#" not in line


def _add_local_installation(lines):
    lines.append("    - -e ../")
    return lines


def main():

    lines = Path("environment.yml").read_text().splitlines()

    # ==================================================================================
    # Create test environments
    # ==================================================================================

    # standard testing

    test_env = [line for line in lines if _keep_line(line, "tests")]
    test_env = _add_local_installation(test_env)

    # find index to insert additional dependencies
    _insert_idx = [i for i, line in enumerate(lines) if "dependencies:" in line][0] + 1

    ## linux
    test_env_linux = deepcopy(test_env)
    test_env_linux.insert(_insert_idx, "  - pygmo")
    test_env_linux.insert(_insert_idx, "  - jax")

    ## test environment others
    test_env_others = deepcopy(test_env)
    test_env_others.insert(_insert_idx, "  - cyipopt")

    ## docs testing
    docs_env = [line for line in lines if _keep_line(line, "docs")]
    docs_env = _add_local_installation(docs_env)

    # Write environments
    Path(".envs/testenv-linux.yml").write_text("\n".join(test_env_linux) + "\n")
    Path(".envs/testenv-others.yml").write_text("\n".join(test_env_others) + "\n")
    Path(".envs/testenv-linkcheck.yml").write_text("\n".join(docs_env) + "\n")


if __name__ == "__main__":
    main()
