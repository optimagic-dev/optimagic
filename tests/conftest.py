import functools
import os
from collections.abc import Generator

import pandas as pd
import pytest
import statsmodels.api as sm

from optimagic.algorithms import ALL_ALGORITHMS, AVAILABLE_ALGORITHMS
from optimagic.config import IS_MATPLOTLIB_INSTALLED

# Maps modules in optimagic.optimizers to the pytest marker of the wrapped library. The
# markers equal the names of the pixi features and environments (tests-<marker>) of
# the libraries. Each library has a CI job that runs ``pytest --library <marker>``.
OPTIMIZER_MODULE_TO_LIBRARY = {
    "bayesian_optimizer": "bayesopt",
    "fides": "fides",
    "gfo_optimizers": "gfo",
    "iminuit_migrad": "iminuit",
    "ipopt": "ipopt",
    "nag_optimizers": "nag",
    "nevergrad_optimizers": "nevergrad",
    "nlopt_optimizers": "nlopt",
    "pygad_optimizer": "pygad",
    "pygmo_optimizers": "pygmo",
    "pyswarms_optimizers": "pyswarms",
    "tao_optimizers": "tao",
    "tranquilo": "tranquilo",
}

LIBRARIES = frozenset(OPTIMIZER_MODULE_TO_LIBRARY.values())


def _get_library(algorithm: str) -> str | None:
    """Return the marker of the library that provides an algorithm.

    Args:
        algorithm: Name of an algorithm in ``ALL_ALGORITHMS``.

    Returns:
        The library marker, or None if the algorithm needs no optional library.

    """
    module = ALL_ALGORITHMS[algorithm].__module__.rsplit(".", maxsplit=1)[-1]
    return OPTIMIZER_MODULE_TO_LIBRARY.get(module)


@functools.cache
def _get_unavailable_algorithms(library: str) -> tuple[str, ...]:
    """Return the algorithms of a library that are not available in this environment.

    A library counts as available if this is empty. The availability of an algorithm
    includes version requirements, e.g. bayes_opt >= 2.0 for the algorithm
    ``bayes_opt``.

    Args:
        library: A library marker.

    Returns:
        The sorted names of the unavailable algorithms.

    """
    return tuple(
        sorted(
            name
            for name in ALL_ALGORITHMS
            if _get_library(name) == library and name not in AVAILABLE_ALGORITHMS
        )
    )


# The marker of an optimizer library (e.g. ``nlopt``) is the only way to declare that a
# test needs the library:
#
# - Tests that are parametrized with an algorithm get the marker of its library.
#   Test modules of one library set it with ``pytestmark``.
# - Tests with a library marker are skipped if one of the library's algorithms is not
#   available. Tests need no ``skipif`` for this, and
#   tests/test_optimizer_library_markers.py fails if they use one.
# - ``pytest --library nlopt`` runs only the tests of nlopt. It fails if an nlopt
#   algorithm is not available or if a selected test is skipped, so that a broken
#   environment or an empty parametrization cannot hide tests.


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--library",
        choices=sorted(LIBRARIES),
        help=(
            "Run only the tests of this optimizer library. Fail if one of its "
            "algorithms is not available or if a selected test is skipped."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    library = config.getoption("library")
    if library is None:
        return
    unavailable = _get_unavailable_algorithms(library)
    if unavailable:
        msg = (
            f"--library {library}: the algorithms {list(unavailable)} are not "
            f"available. Run the tests in the environment tests-{library}."
        )
        raise pytest.UsageError(msg)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]):
    """Add library markers, skip tests of missing libraries and apply ``--library``."""
    library = config.getoption("library")
    selected = []
    deselected = []
    for item in items:
        libraries = _get_libraries(item)
        for name in sorted(libraries):
            item.add_marker(name)
        missing = sorted(
            name for name in libraries if _get_unavailable_algorithms(name)
        )
        if missing:
            reason = f"Needs the optimizer libraries {missing}."
            item.add_marker(pytest.mark.skip(reason=reason))
        if library is None or library in libraries:
            selected.append(item)
        else:
            deselected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item,
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    """Turn skipped tests into failures if one library is selected with --library."""
    report = yield
    library = item.config.getoption("library")
    if library is not None and report.skipped and not hasattr(report, "wasxfail"):
        longrepr = report.longrepr
        reason = longrepr[2] if isinstance(longrepr, tuple) else str(longrepr)
        report.outcome = "failed"
        report.longrepr = (
            f"The test was skipped, but with --library {library} every selected test "
            f"must run. Reason for the skip: {reason}"
        )
    return report


def _get_libraries(item: pytest.Item) -> set[str]:
    """Return the libraries an item needs, from its markers and its parameters."""
    libraries = {
        marker.name for marker in item.iter_markers() if marker.name in LIBRARIES
    }
    callspec = getattr(item, "callspec", None)
    if callspec is not None:
        for value in callspec.params.values():
            if isinstance(value, str) and value in ALL_ALGORITHMS:
                library = _get_library(value)
                if library is not None:
                    libraries.add(library)
    return libraries


@pytest.fixture(autouse=True)
def fresh_directory(tmp_path):  # noqa: PT004
    """Each test is executed in a fresh directory."""
    os.chdir(tmp_path)


@pytest.fixture()
def logit_inputs():
    spector_data = sm.datasets.spector.load_pandas()
    spector_data.exog = sm.add_constant(spector_data.exog)
    x_df = sm.add_constant(spector_data.exog)
    out = {
        "y": spector_data.endog,
        "x": x_df.to_numpy(),
        "params": pd.DataFrame([-10, 2, 0.2, 2], index=x_df.columns, columns=["value"]),
    }
    return out


@pytest.fixture()
def logit_object():
    spector_data = sm.datasets.spector.load_pandas()
    spector_data.exog = sm.add_constant(spector_data.exog)
    logit_mod = sm.Logit(spector_data.endog, spector_data.exog)
    return logit_mod


@pytest.fixture()
def close_mpl_figures():
    """Close all matplotlib figures after test execution."""
    yield
    if IS_MATPLOTLIB_INSTALLED:
        import matplotlib.pyplot as plt

        plt.close("all")
