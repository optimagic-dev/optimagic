"""Test that every optimizer library is tested in its own environment and CI job."""

import tomllib
from pathlib import Path

import yaml

from optimagic.algorithms import ALL_ALGORITHMS
from tests.conftest import OPTIMIZER_MODULE_TO_LIBRARY_MARKER

ROOT = Path(__file__).resolve().parents[1]

# Optimizers that only need scipy or nothing at all. They are tested in the main test
# environments and need no library marker.
MODULES_WITHOUT_LIBRARY = {"bhhh", "neldermead", "pounders", "scipy_optimizers"}

LIBRARY_MARKERS = set(OPTIMIZER_MODULE_TO_LIBRARY_MARKER.values())


def test_every_optimizer_module_has_a_library_marker():
    modules = {algo.__module__.rsplit(".", 1)[-1] for algo in ALL_ALGORITHMS.values()}
    expected = set(OPTIMIZER_MODULE_TO_LIBRARY_MARKER) | MODULES_WITHOUT_LIBRARY
    assert modules == expected


def test_every_library_marker_is_registered_and_has_a_pixi_environment():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    registered = {
        marker.split(":")[0]
        for marker in pyproject["tool"]["pytest"]["ini_options"]["markers"]
    }
    environments = pyproject["tool"]["pixi"]["environments"]
    assert LIBRARY_MARKERS <= registered
    assert {f"tests-{marker}" for marker in LIBRARY_MARKERS} <= set(environments)


def test_every_library_marker_has_a_ci_job():
    workflow = yaml.safe_load((ROOT / ".github/workflows/main.yml").read_text())
    job = workflow["jobs"]["run-optimizer-library-tests"]
    assert set(job["strategy"]["matrix"]["library"]) == LIBRARY_MARKERS
