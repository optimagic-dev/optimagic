"""Test that every optimizer library is tested in its own environment and CI job."""

import ast
import tomllib
from pathlib import Path

import yaml

from optimagic.algorithms import ALL_ALGORITHMS
from tests.conftest import LIBRARIES, OPTIMIZER_MODULE_TO_LIBRARY

ROOT = Path(__file__).resolve().parents[1]

# Optimizers that only need scipy or nothing at all. They are tested in the main test
# environments and need no library marker.
MODULES_WITHOUT_LIBRARY = {"bhhh", "neldermead", "pounders", "scipy_optimizers"}


def test_every_optimizer_module_has_a_library_marker():
    modules = {algo.__module__.rsplit(".", 1)[-1] for algo in ALL_ALGORITHMS.values()}
    expected = set(OPTIMIZER_MODULE_TO_LIBRARY) | MODULES_WITHOUT_LIBRARY
    assert modules == expected


def test_every_library_marker_is_registered_and_has_a_pixi_environment():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    registered = {
        marker.split(":")[0]
        for marker in pyproject["tool"]["pytest"]["ini_options"]["markers"]
    }
    environments = pyproject["tool"]["pixi"]["environments"]
    assert LIBRARIES <= registered
    assert {f"tests-{marker}" for marker in LIBRARIES} <= set(environments)


def test_every_library_marker_has_a_ci_job():
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/main.yml").read_text(encoding="utf-8")
    )
    job = workflow["jobs"]["run-optimizer-library-tests"]
    assert set(job["strategy"]["matrix"]["library"]) == LIBRARIES


def test_tests_do_not_skip_on_missing_optimizer_libraries_themselves():
    """Tests declare that they need a library only with its marker.

    A test that skips itself if a library is missing, e.g. with
    ``pytest.mark.skipif(not IS_NLOPT_INSTALLED, ...)``, but has no library marker is
    skipped in the main test environments and not selected in the library's CI job, so
    it never runs. tests/conftest.py adds the skip to all tests with a library marker.

    The availability flags may only guard imports, as in ``if IS_NLOPT_INSTALLED:
    import nlopt``. An ``else`` branch may assign placeholders for the imported names.

    """
    flags = _get_availability_flags_of_optimizer_modules()
    violations = []
    for path in sorted((ROOT / "tests").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        allowed = _get_flag_nodes_that_guard_imports(tree, flags)
        violations += [
            f"{path.relative_to(ROOT)}:{node.lineno}: {node.id}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id in flags and node not in allowed
        ]
    assert not violations, "\n".join(violations)


def _get_availability_flags_of_optimizer_modules():
    """Return the names of the flags that optimizer modules import from config."""
    flags = set()
    for path in (ROOT / "src" / "optimagic" / "optimizers").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module == "optimagic.config":
                flags |= {a.name for a in node.names if a.name.startswith("IS_")}
    return flags


def _get_flag_nodes_that_guard_imports(tree, flags):
    """Return the flags that are the condition of an ``if`` that only imports."""
    return [
        node.test
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id in flags
        and all(isinstance(stmt, ast.Import | ast.ImportFrom) for stmt in node.body)
        and all(isinstance(stmt, ast.Assign) for stmt in node.orelse)
    ]
