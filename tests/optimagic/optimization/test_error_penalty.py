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

    assert np.isclose(scalar.value, contribs.value.sum())  # ty:ignore[unresolved-attribute]
    assert np.isclose(scalar.value, (root_contribs.value**2).sum())  # ty:ignore[unresolved-attribute]


SUPPORTED_COMBINATIONS = [
    (ScalarFunctionValue(-3.0), AggregationLevel.SCALAR, Direction.MINIMIZE),
    (ScalarFunctionValue(-3.0), AggregationLevel.SCALAR, Direction.MAXIMIZE),
    (
        LikelihoodFunctionValue(np.arange(-3.0, 2.0)),
        AggregationLevel.SCALAR,
        Direction.MINIMIZE,
    ),
    (
        LikelihoodFunctionValue(np.arange(-3.0, 2.0)),
        AggregationLevel.SCALAR,
        Direction.MAXIMIZE,
    ),
    (
        LikelihoodFunctionValue(np.arange(-3.0, 2.0)),
        AggregationLevel.LIKELIHOOD,
        Direction.MINIMIZE,
    ),
    (
        LikelihoodFunctionValue(np.arange(-3.0, 2.0)),
        AggregationLevel.LIKELIHOOD,
        Direction.MAXIMIZE,
    ),
    (
        LeastSquaresFunctionValue(np.arange(-2.0, 3.0)),
        AggregationLevel.SCALAR,
        Direction.MINIMIZE,
    ),
    (
        LeastSquaresFunctionValue(np.arange(-2.0, 3.0)),
        AggregationLevel.LIKELIHOOD,
        Direction.MINIMIZE,
    ),
    (
        LeastSquaresFunctionValue(np.arange(-2.0, 3.0)),
        AggregationLevel.LEAST_SQUARES,
        Direction.MINIMIZE,
    ),
]


@pytest.mark.parametrize(
    "start_criterion, solver_type, direction", SUPPORTED_COMBINATIONS
)
def test_default_penalty_matches_the_scalar_penalty_of_the_start_value(
    start_criterion, solver_type, direction
):
    """Without explicit options, every aggregation penalizes the same scalar value."""
    x = np.array([0.3, 0.7])
    x0 = np.array([0.1, 0.2])
    scalar_func = get_error_penalty_function(
        start_x=x0,
        start_criterion=ScalarFunctionValue(
            start_criterion.internal_value(AggregationLevel.SCALAR)
        ),
        error_penalty=None,
        solver_type=AggregationLevel.SCALAR,
        direction=direction,
    )
    func = get_error_penalty_function(
        start_x=x0,
        start_criterion=start_criterion,
        error_penalty=None,
        solver_type=solver_type,
        direction=direction,
    )
    expected, _ = scalar_func(x)
    got, _ = func(x)
    assert np.isclose(
        got.internal_value(AggregationLevel.SCALAR),
        expected.internal_value(AggregationLevel.SCALAR),
    )


@pytest.mark.parametrize(
    "start_criterion, solver_type, direction", SUPPORTED_COMBINATIONS
)
def test_default_penalty_is_worse_than_the_start_value(
    start_criterion, solver_type, direction
):
    x0 = np.array([0.1, 0.2])
    func = get_error_penalty_function(
        start_x=x0,
        start_criterion=start_criterion,
        error_penalty=None,
        solver_type=solver_type,
        direction=direction,
    )
    start_value = start_criterion.internal_value(AggregationLevel.SCALAR)
    for x in [x0 + 1e-8, np.array([0.3, 0.7]), np.array([-5.0, 10.0])]:
        penalty, _ = func(x)
        penalty_value = penalty.internal_value(AggregationLevel.SCALAR)
        if direction == Direction.MINIMIZE:
            assert penalty_value > start_value
        else:
            assert penalty_value < start_value
