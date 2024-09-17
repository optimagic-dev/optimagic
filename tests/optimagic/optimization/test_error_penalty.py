import functools

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from optimagic.differentiation.derivatives import first_derivative
from optimagic.optimization.error_penalty import (
    _likelihood_penalty,
    _penalty_residuals,
    _scalar_penalty,
    get_error_penalty_function,
)
from optimagic.optimization.fun_value import (
    LeastSquaresFunctionValue,
    LikelihoodFunctionValue,
    ScalarFunctionValue,
)
from optimagic.typing import AggregationLevel, Direction
from optimagic.utilities import get_rng


@pytest.mark.parametrize("seed", range(10))
def test_penalty_aggregations(seed):
    rng = get_rng(seed)
    x = rng.uniform(size=5)
    x0 = rng.uniform(size=5)
    slope = 0.3
    constant = 3
    dim_out = 10

    scalar, _ = _scalar_penalty(x, constant, slope, x0)
    contribs, _ = _likelihood_penalty(x, constant, slope, x0, dim_out)
    root_contribs, _ = _penalty_residuals(x, constant, slope, x0, dim_out)

    assert np.isclose(scalar.value, contribs.value.sum())
    assert np.isclose(scalar.value, (root_contribs.value**2).sum())


pairs = [
    (_scalar_penalty, AggregationLevel.SCALAR),
    (_likelihood_penalty, AggregationLevel.LIKELIHOOD),
    (_penalty_residuals, AggregationLevel.LEAST_SQUARES),
]


@pytest.mark.parametrize("func, solver_type", pairs)
def test_penalty_derivatives(func, solver_type):
    rng = get_rng(seed=5)
    x = rng.uniform(size=5)
    x0 = rng.uniform(size=5)
    slope = 0.3
    constant = 3
    dim_out = 8

    _, calculated = func(x, constant, slope, x0, dim_out)

    partialed = functools.partial(
        func, constant=constant, slope=slope, x0=x0, dim_out=dim_out
    )
    expected = first_derivative(
        partialed, x, unpacker=lambda x: x[0].internal_value(solver_type)
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
        start_x=x0,
        start_criterion=ScalarFunctionValue(3),
        error_penalty={"slope": slope, "constant": constant},
        solver_type=AggregationLevel.SCALAR,
        direction=Direction.MINIMIZE,
    )

    contribs_func = get_error_penalty_function(
        start_x=x0,
        start_criterion=LikelihoodFunctionValue(np.ones(10)),
        error_penalty={"slope": slope, "constant": constant},
        solver_type=AggregationLevel.LIKELIHOOD,
        direction=Direction.MINIMIZE,
    )

    root_contribs_func = get_error_penalty_function(
        start_x=x0,
        start_criterion=LeastSquaresFunctionValue(np.ones(10)),
        error_penalty={"slope": slope, "constant": constant},
        solver_type=AggregationLevel.LEAST_SQUARES,
        direction=Direction.MINIMIZE,
    )

    scalar, _ = scalar_func(x)
    contribs, _ = contribs_func(x)
    root_contribs, _ = root_contribs_func(x)

    assert np.isclose(scalar.value, contribs.value.sum())
    assert np.isclose(scalar.value, (root_contribs.value**2).sum())
