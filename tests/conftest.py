import os

import pandas as pd
import pytest
import statsmodels.api as sm

from optimagic.algorithms import ALL_ALGORITHMS
from optimagic.config import IS_MATPLOTLIB_INSTALLED

# Maps modules in optimagic.optimizers to the pytest marker of the wrapped library. The
# markers equal the names of the pixi features and environments (tests-<marker>) of
# the libraries.
OPTIMIZER_MODULE_TO_LIBRARY_MARKER = {
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


def pytest_collection_modifyitems(items):
    """Mark tests that are parametrized with an algorithm with its library's marker.

    Tests that loop over all installed algorithms, e.g. in test_many_algorithms.py,
    thereby run in the CI job of each optimizer library (``pytest -m nlopt`` in the
    environment ``tests-nlopt``). Test modules of a single library set the marker with
    ``pytestmark``.

    """
    for item in items:
        callspec = getattr(item, "callspec", None)
        if callspec is None:
            continue
        for value in callspec.params.values():
            marker = _get_library_marker(value)
            if marker is not None:
                item.add_marker(marker)


def _get_library_marker(value):
    if not isinstance(value, str) or value not in ALL_ALGORITHMS:
        return None
    module = ALL_ALGORITHMS[value].__module__.rsplit(".", maxsplit=1)[-1]
    return OPTIMIZER_MODULE_TO_LIBRARY_MARKER.get(module)


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
