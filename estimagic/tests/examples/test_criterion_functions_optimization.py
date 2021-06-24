from unittest import TestCase

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

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


# Fix input params to test all criterion functions
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
    assert_allclose(out, trid_gradient_output)
    assert isinstance(out, np.ndarray)


def test_trid_pandas_gradient(input_params, trid_gradient_output):
    out = trid_pandas_gradient(input_params)
    assert_allclose(out, trid_gradient_output)
    assert isinstance(out, pd.Series)


def test_trid_dict_criterion(input_params):
    out = trid_dict_criterion(input_params)
    assert isinstance(out, dict)
    assert out == {"value": 83}


def test_trid_criterion_and_gradient(
    input_params, trid_gradient_output, trid_criterion_output
):
    out = trid_criterion_and_gradient(input_params)
    res = trid_criterion_output, trid_gradient_output
    # res = 83, np.array([7, 1, -6, 11, -19]
    assert_allclose(out, res)


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
    assert_allclose(out, rhe_gradient_output)
    assert isinstance(out, np.ndarray)


def test_rhe_pandas_gradient(input_params, rhe_gradient_output):
    out = rotated_hyper_ellipsoid_pandas_gradient(input_params)
    assert_allclose(out, rhe_gradient_output)
    assert isinstance(out, pd.Series)


def test_rhe_dict_criterion(input_params):
    out = rotated_hyper_ellipsoid_dict_criterion(input_params)
    assert isinstance(out, dict)
    expected = {
        "value": 960,
        "contributions": np.array([81, 162, 198, 247, 272]),
        "root_contributions": np.array(
            [9, 12.72792206, 14.07124728, 15.71623365, 16.4924225]
        ),
    }
    TestCase().assertDictEqual(out, expected)


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
    assert_allclose(out, rosenbrock_gradient_output)
    assert isinstance(out, np.ndarray)


def test_rosenbrock_pandas_gradient(input_params, rosenbrock_gradient_output):
    out = rosenbrock_pandas_gradient(input_params)
    assert_allclose(out, rosenbrock_gradient_output)
    assert isinstance(out, pd.Series)
