import numpy as np
import pytest

from optimagic.exceptions import InvalidPlottingBackendError, NotInstalledError
from optimagic.visualization.backends import (
    BACKEND_AVAILABILITY_AND_LINE_PLOT_FUNCTION,
    line_plot,
)
from optimagic.visualization.plotting_utilities import LineData


@pytest.fixture()
def sample_lines():
    lines = [
        LineData(x=np.array([0, 1, 2]), y=np.array([0, 1, 2])),
        LineData(x=np.array([0, 1, 2]), y=np.array([2, 1, 0])),
    ]
    return lines


@pytest.mark.parametrize("backend", BACKEND_AVAILABILITY_AND_LINE_PLOT_FUNCTION.keys())
def test_line_plot_all_backends(sample_lines, backend):
    line_plot(sample_lines, backend=backend)


def test_line_plot_invalid_backend(sample_lines):
    with pytest.raises(InvalidPlottingBackendError):
        line_plot(sample_lines, backend="bla")


def test_line_plot_unavailable_backend(sample_lines, monkeypatch):
    # Use monkeypatch to simulate that 'matplotlib' backend is not installed.
    monkeypatch.setitem(
        BACKEND_AVAILABILITY_AND_LINE_PLOT_FUNCTION, "matplotlib", (False, None, None)
    )

    with pytest.raises(NotInstalledError):
        line_plot(sample_lines, backend="matplotlib")
