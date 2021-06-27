import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal

from estimagic.examples.criterion_functions_optimization_tests import (
    rosenbrock_criterion_and_gradient,
)
from estimagic.examples.criterion_functions_optimization_tests import (
    rosenbrock_dict_criterion,
)
from estimagic.examples.criterion_functions_optimization_tests import (
    rosenbrock_gradient,
)
from estimagic.examples.criterion_functions_optimization_tests import (
    rosenbrock_pandas_gradient,
)
from estimagic.examples.criterion_functions_optimization_tests import (
    rosenbrock_scalar_criterion,
)
from estimagic.examples.criterion_functions_optimization_tests import (
    rotated_hyper_ellipsoid_criterion_and_gradient,
)
from estimagic.examples.criterion_functions_optimization_tests import (
    rotated_hyper_ellipsoid_dict_criterion,
)
from estimagic.examples.criterion_functions_optimization_tests import (
    rotated_hyper_ellipsoid_gradient,
)
from estimagic.examples.criterion_functions_optimization_tests import (
    rotated_hyper_ellipsoid_pandas_gradient,
)
from estimagic.examples.criterion_functions_optimization_tests import (
    rotated_hyper_ellipsoid_scalar_criterion,
)
from estimagic.examples.criterion_functions_optimization_tests import (
    trid_criterion_and_gradient,
)
from estimagic.examples.criterion_functions_optimization_tests import (
    trid_dict_criterion,
)
from estimagic.examples.criterion_functions_optimization_tests import trid_gradient
from estimagic.examples.criterion_functions_optimization_tests import (
    trid_pandas_gradient,
)
from estimagic.examples.criterion_functions_optimization_tests import (
    trid_scalar_criterion,
)


# Fix input params to test every criterion function
@pytest.fixture
def input_params():
    params = pd.DataFrame({"value": [9, 9, 6, 7, -5]})
    return params


# Fixtures for trid function output
@pytest.fixture
def trid_criterion_output():
    trid_crit = 83
    return trid_crit


@pytest.fixture
def trid_gradient_output():
    trid_gradient = np.array([7, 1, -6, 11, -19])
    return trid_gradient


def test_trid_scalar_criterion(input_params, trid_criterion_output):
    out = trid_scalar_criterion(input_params)
    assert out == trid_criterion_output


def test_trid_gradient(input_params, trid_gradient_output):
    out = trid_gradient(input_params)
    assert_array_equal(out, trid_gradient_output)
    assert isinstance(out, np.ndarray)


def test_trid_pandas_gradient(input_params, trid_gradient_output):
    out = trid_pandas_gradient(input_params)
    assert_array_equal(out, trid_gradient_output)
    assert isinstance(out, pd.Series)


def test_trid_dict_criterion(input_params):
    out_dict = trid_dict_criterion(input_params)
    assert isinstance(out_dict, dict)
    assert out_dict == {"value": 83}


def test_trid_criterion_and_gradient(
    input_params, trid_gradient_output, trid_criterion_output
):
    out = trid_criterion_and_gradient(input_params)
    expected = trid_criterion_output, trid_gradient_output
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert out[0] == expected[0]
    assert isinstance(out[1], np.ndarray)
    assert_array_equal(out[1], expected[1])


# Fixtures for rhe function output
@pytest.fixture
def rhe_criterion_output():
    rhe_crit = 960
    return rhe_crit


@pytest.fixture
def rhe_gradient_output():
    rhe_gradient = np.array([90, 72, 36, 28, -10])
    return rhe_gradient


def test_rhe_scalar_criterion(input_params, rhe_criterion_output):
    out = rotated_hyper_ellipsoid_scalar_criterion(input_params)
    assert out == rhe_criterion_output


def test_rhe_gradient(input_params, rhe_gradient_output):
    out = rotated_hyper_ellipsoid_gradient(input_params)
    assert_array_equal(out, rhe_gradient_output)
    assert isinstance(out, np.ndarray)


def test_rhe_pandas_gradient(input_params, rhe_gradient_output):
    out = rotated_hyper_ellipsoid_pandas_gradient(input_params)
    assert_array_equal(out, rhe_gradient_output)
    assert isinstance(out, pd.Series)


def test_rhe_dict_criterion(input_params):
    out_dict = rotated_hyper_ellipsoid_dict_criterion(input_params)
    expected_dict = {
        "value": 960,
        "contributions": np.array([81, 162, 198, 247, 272]),
        "root_contributions": np.array(
            [9, 12.72792206, 14.07124728, 15.71623365, 16.4924225]
        ),
    }
    assert isinstance(out_dict, dict)
    assert len(out_dict) == 3
    assert isinstance(out_dict["contributions"], np.ndarray)
    assert isinstance(out_dict["root_contributions"], np.ndarray)

    for key in expected_dict.keys():
        assert_allclose(out_dict[key], expected_dict[key])


def test_rhe_criterion_and_gradient(
    input_params, rhe_gradient_output, rhe_criterion_output
):
    out = rotated_hyper_ellipsoid_criterion_and_gradient(input_params)
    expected = rhe_criterion_output, rhe_gradient_output
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert isinstance(out[1], np.ndarray)
    assert out[0] == expected[0]
    assert_array_equal(out[1], expected[1])


# Fixtures for rosenbrock function output
@pytest.fixture
def rosenbrock_criterion_output():
    rosenbrock_crit = 1456789
    return rosenbrock_crit


@pytest.fixture
def rosenbrock_gradient_output():
    rosenbrock_gradient = np.array([259216, 255616, 54610, 145412, -10800])
    return rosenbrock_gradient


def test_rosenbrock_scalar_criterion(input_params, rosenbrock_criterion_output):
    out = rosenbrock_scalar_criterion(input_params)
    assert out == rosenbrock_criterion_output


def test_rosenbrock_gradient(input_params, rosenbrock_gradient_output):
    out = rosenbrock_gradient(input_params)
    assert isinstance(out, np.ndarray)
    assert_array_equal(out, rosenbrock_gradient_output)


def test_rosenbrock_pandas_gradient(input_params, rosenbrock_gradient_output):
    out = rosenbrock_pandas_gradient(input_params)
    assert isinstance(out, pd.Series)
    assert_array_equal(out, rosenbrock_gradient_output)


def test_rosenbrock_criterion_and_gradient(
    input_params, rosenbrock_gradient_output, rosenbrock_criterion_output
):
    out = rosenbrock_criterion_and_gradient(input_params)
    expected = rosenbrock_criterion_output, rosenbrock_gradient_output
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert isinstance(out[1], np.ndarray)
    assert out[0] == expected[0]
    assert_array_equal(out[1], expected[1])


def test_rosenbrock_dict_criterion(input_params):
    out_dict = rosenbrock_dict_criterion(input_params)
    expected_dict = {
        "value": 1456789,
        "contributions": np.array([518464, 562564, 84125, 291636, 0]),
        "root_contributions": np.array(
            [720.04444307, 750.04266545, 290.04310025, 540.0333323, 0]
        ),
    }
    assert isinstance(out_dict, dict)
    assert len(out_dict) == 3
    assert isinstance(out_dict["contributions"], np.ndarray)
    assert isinstance(out_dict["root_contributions"], np.ndarray)

    for key in expected_dict.keys():
        assert_allclose(out_dict[key], expected_dict[key])
