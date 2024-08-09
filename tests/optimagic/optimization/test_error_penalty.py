import functools

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from optimagic.differentiation.derivatives import first_derivative
from optimagic.optimization.error_penalty import (
    _penalty_loglikes,
    _penalty_loglikes_derivative,
    _penalty_residuals,
    _penalty_residuals_derivative,
    _penalty_value,
    _penalty_value_derivative,
    get_error_penalty_function,
)
from optimagic.optimization.fun_value import (
    LeastSquaresFunctionValue,
    LikelihoodFunctionValue,
    ScalarFunctionValue,
)
from optimagic.typing import AggregationLevel
from optimagic.utilities import get_rng


@pytest.mark.parametrize("seed", range(10))
def test_penalty_aggregations(seed):
    rng = get_rng(seed)
    x = rng.uniform(size=5)
    x0 = rng.uniform(size=5)
    slope = 0.3
    constant = 3
    dim_out = 10

    scalar = _penalty_value(x, constant, slope, x0).value
    contribs = _penalty_loglikes(x, constant, slope, x0, dim_out).value
    root_contribs = _penalty_residuals(x, constant, slope, x0, dim_out).value

    assert np.isclose(scalar, contribs.sum())
    assert np.isclose(scalar, (root_contribs**2).sum())


pairs = [
    (_penalty_value, _penalty_value_derivative, AggregationLevel.SCALAR),
    (
        _penalty_loglikes,
        _penalty_loglikes_derivative,
        AggregationLevel.LIKELIHOOD,
    ),
    (
        _penalty_residuals,
        _penalty_residuals_derivative,
        AggregationLevel.LEAST_SQUARES,
    ),
]


@pytest.mark.parametrize("func, deriv, solver_type", pairs)
def test_penalty_derivatives(func, deriv, solver_type):
    rng = get_rng(seed=5)
    x = rng.uniform(size=5)
    x0 = rng.uniform(size=5)
    slope = 0.3
    constant = 3
    dim_out = 8

    calculated = deriv(x, constant, slope, x0, dim_out)

    partialed = functools.partial(
        func, constant=constant, slope=slope, x0=x0, dim_out=dim_out
    )
    expected = first_derivative(
        partialed, x, unpacker=lambda x: x.internal_value(solver_type)
    )

    aaae(calculated, expected.derivative)


@pytest.mark.parametrize("seed", range(10))
def test_penalty_aggregations_via_get_error_penalty(seed):
    rng = get_rng(seed)
    x = rng.uniform(size=5)
    x0 = rng.uniform(size=5)
    slope = 0.3
    constant = 3

    scalar_func = get_error_penalty_function(
        error_handling="continue",
        start_x=x0,
        start_criterion=ScalarFunctionValue(3),
        error_penalty={"slope": slope, "constant": constant},
        solver_type=AggregationLevel.SCALAR,
        direction="minimize",
    )

    contribs_func = get_error_penalty_function(
        error_handling="continue",
        start_x=x0,
        start_criterion=LikelihoodFunctionValue(np.ones(10)),
        error_penalty={"slope": slope, "constant": constant},
        solver_type=AggregationLevel.LIKELIHOOD,
        direction="minimize",
    )

    root_contribs_func = get_error_penalty_function(
        error_handling="continue",
        start_x=x0,
        start_criterion=LeastSquaresFunctionValue(np.ones(10)),
        error_penalty={"slope": slope, "constant": constant},
        solver_type=AggregationLevel.LEAST_SQUARES,
        direction="minimize",
    )

    scalar = scalar_func(x, task="criterion").value
    contribs = contribs_func(x, task="criterion").value
    root_contribs = root_contribs_func(x, task="criterion").value

    assert np.isclose(scalar, contribs.sum())
    assert np.isclose(scalar, (root_contribs**2).sum())
