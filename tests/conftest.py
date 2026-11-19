import os

import pandas as pd
import pytest
import statsmodels.api as sm

from optimagic.config import IS_MATPLOTLIB_INSTALLED


def pytest_collection_modifyitems(config, items):
    """Run wrapper tests only if they are selected with ``-m``.

    Wrapper tests carry a library-level marker that ends on ``_wrappers``, e.g.
    ``fides_wrappers``. They test options of wrapped optimizers in detail, which is
    only needed when a wrapper or the wrapped library changes. ``-m fides_wrappers``
    selects the tests of one library, ``-m wrappers`` selects all of them.

    """
    wrapper_tests = [
        item
        for item in items
        if any(marker.name.endswith("_wrappers") for marker in item.iter_markers())
    ]
    for item in wrapper_tests:
        item.add_marker("wrappers")

    if "wrappers" not in config.getoption("markexpr", ""):
        config.hook.pytest_deselected(items=wrapper_tests)
        deselected = {id(item) for item in wrapper_tests}
        items[:] = [item for item in items if id(item) not in deselected]


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
