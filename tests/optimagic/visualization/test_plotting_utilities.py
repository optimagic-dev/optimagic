import base64

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from optimagic.visualization.plotting_utilities import (
    _decode_base64_data,
    _ensure_array_from_plotly_data,
)


def test_decode_base64_data():
    expected = np.arange(10, dtype=float)
    encoded = base64.b64encode(expected.tobytes()).decode("ascii")
    got = _decode_base64_data(encoded, dtype="float")
    assert_array_equal(expected, got)


def test_ensure_array_from_plotly_data_case_array():
    expected = np.arange(10, dtype=float)
    got = _ensure_array_from_plotly_data(expected)
    assert_array_equal(expected, got)


def test_ensure_array_from_plotly_data_case_list():
    expected = np.arange(10, dtype=float)
    got = _ensure_array_from_plotly_data(expected.tolist())
    assert_array_equal(expected, got)


def test_ensure_array_from_plotly_data_case_base64():
    expected = np.arange(10, dtype=float)
    encoded = base64.b64encode(expected.tobytes()).decode("ascii")
    got = _ensure_array_from_plotly_data({"bdata": encoded, "dtype": "float"})
    assert_array_equal(expected, got)


@pytest.mark.parametrize(
    "invalid_input",
    [
        None,
        "not a valid input",
        1234,
        [{"a": 1}, {"b": 2}],
    ],
)
def test_ensure_array_from_plotly_data_case_invalid(invalid_input):
    with pytest.raises(ValueError, match="Failed to convert input to numpy array."):
        _ensure_array_from_plotly_data(invalid_input)
