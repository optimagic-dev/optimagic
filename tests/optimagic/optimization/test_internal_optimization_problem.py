from copy import copy

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from optimagic import NumdiffOptions
from optimagic.batch_evaluators import process_batch_evaluator
from optimagic.config import CRITERION_PENALTY_CONSTANT, CRITERION_PENALTY_SLOPE
from optimagic.exceptions import UserFunctionRuntimeError
from optimagic.optimization.error_penalty import get_error_penalty_function
from optimagic.optimization.fun_value import (
    LeastSquaresFunctionValue,
    ScalarFunctionValue,
)
from optimagic.optimization.internal_optimization_problem import (
    InternalBounds,
    InternalOptimizationProblem,
)
from optimagic.parameters.conversion import Converter
from optimagic.typing import AggregationLevel, Direction, ErrorHandling, EvalTask


@pytest.fixture
def base_problem():
    """Set up a basic InternalOptimizationProblem that can be modified for tests."""

    def fun(params):
        return LeastSquaresFunctionValue(value=params, info={"mean": params.mean()})

    def jac(params):
        return 2 * params

    def fun_and_jac(params):
        return fun(params), jac(params)

    converter = Converter(
        params_to_internal=lambda x: x,
        params_from_internal=lambda x: x,
        derivative_to_internal=lambda d, x: d,
        has_transforming_constraints=False,
    )

    solver_type = AggregationLevel.SCALAR

    direction = Direction.MINIMIZE

    bounds = InternalBounds(lower=None, upper=None)

    numdiff_options = NumdiffOptions()

    error_handling = ErrorHandling.RAISE

    batch_evaluator = process_batch_evaluator(batch_evaluator="joblib")

    linear_constraints = None

    nonlinear_constraints = None

    problem = InternalOptimizationProblem(
        fun=fun,
        jac=jac,
        fun_and_jac=fun_and_jac,
        converter=converter,
        solver_type=solver_type,
        direction=direction,
        bounds=bounds,
        numdiff_options=numdiff_options,
        error_handling=error_handling,
        error_penalty_func=None,
        batch_evaluator=batch_evaluator,
        linear_constraints=linear_constraints,
        nonlinear_constraints=nonlinear_constraints,
        logger=None,
    )

    return problem


# ======================================================================================
# Test fun, jac, fun_and_jac
# ======================================================================================


def test_base_problem_fun(base_problem):
    got = base_problem.fun(np.array([1, 2, 3]))
    expected = 14
    assert got == expected


def test_base_problem_jac(base_problem):
    got = base_problem.jac(np.array([1, 2, 3]))
    expected = 2 * np.array([1, 2, 3])
    aaae(got, expected)


def test_base_problem_fun_and_jac(base_problem):
    got_fun, got_jac = base_problem.fun_and_jac(np.array([1, 2, 3]))
    expected_fun, expected_jac = (14, 2 * np.array([1, 2, 3]))
    assert got_fun == expected_fun
    aaae(got_jac, expected_jac)


def test_fun_and_jac_is_called_for_jac_if_jac_is_not_given(base_problem):
    """This makes sure we don't use numdiff if we don't have to."""
    call_log = []

    def fun_and_jac(params):
        call_log.append("fun_and_jac")
        return LeastSquaresFunctionValue(value=params), 2 * np.array([1, 2, 3])

    base_problem._jac = None
    base_problem._fun_and_jac = fun_and_jac
    base_problem.jac(np.array([1, 2, 3]))

    assert call_log == ["fun_and_jac"]


def test_jac_is_called_for_fun_and_jac_if_fun_is_not_given(base_problem):
    """This makes sure we don't use numdiff if we don't have to."""
    call_log = []

    def jac(params):
        call_log.append("jac")
        return 2 * np.array([1, 2, 3])

    base_problem._fun_and_jac = None
    base_problem._jac = jac

    base_problem.fun_and_jac(np.array([1, 2, 3]))

    assert call_log == ["jac"]


def test_base_problem_jac_via_numdiff(base_problem):
    base_problem._jac = None
    base_problem._fun_and_jac = None

    got = base_problem.jac(np.array([1, 2, 3]))
    expected = 2 * np.array([1, 2, 3])
    aaae(got, expected)


def test_base_problem_fun_and_jac_via_numdiff(base_problem):
    base_problem._jac = None
    base_problem._fun_and_jac = None

    got_fun, got_jac = base_problem.fun_and_jac(np.array([1, 2, 3]))
    expected_fun, expected_jac = (14, 2 * np.array([1, 2, 3]))
    assert got_fun == expected_fun
    aaae(got_jac, expected_jac)


def test_error_in_fun_with_error_handling_raise(base_problem):
    def fun(params):
        raise ValueError("Test error")

    base_problem._fun = fun

    with pytest.raises(UserFunctionRuntimeError):
        base_problem.fun(np.array([1, 2, 3]))


def test_error_in_fun_during_numdiff_with_error_handling_raise(base_problem):
    def fun(params):
        raise ValueError("Test error")

    base_problem._fun = fun
    base_problem._jac = None
    base_problem._fun_and_jac = None

    with pytest.raises(UserFunctionRuntimeError):
        base_problem.jac(np.array([1, 2, 3]))


def test_base_problem_different_jac_versions(base_problem):
    got_jac_1 = base_problem.jac(np.array([1, 2, 3]))
    _, got_jac_2 = base_problem.fun_and_jac(np.array([1, 2, 3]))

    base_problem._jac = None
    base_problem._fun_and_jac = None
    got_jac_3 = base_problem.jac(np.array([1, 2, 3]))

    aaae(got_jac_1, got_jac_2)
    aaae(got_jac_1, got_jac_3)


def test_base_problem_fun_for_ls_optimizer(base_problem):
    base_problem._solver_type = AggregationLevel.LEAST_SQUARES

    got = base_problem.fun(np.array([1, 2, 3]))
    expected = np.array([1, 2, 3])
    aaae(got, expected)


def test_base_problem_exploration_fun(base_problem):
    got = base_problem.exploration_fun(
        [np.array([1, 2, 3]), np.array([4, 5, 6])], n_cores=1
    )
    expected = [14, 77]
    assert got == expected


# ======================================================================================
# test history
# ======================================================================================


def test_history_with_fun(base_problem):
    base_problem.fun(np.array([1, 2, 3]))

    assert len(base_problem.history.params) == 1
    aaae(base_problem.history.params[0], [1, 2, 3])
    assert base_problem.history.fun == [14]
    assert base_problem.history.task == [EvalTask.FUN]
    assert base_problem.history.batches == [0]


def test_history_with_batch_fun(base_problem):
    base_problem.batch_fun(
        [np.array([1, 2, 3]), np.array([4, 5, 6])], n_cores=1, batch_size=2
    )
    assert len(base_problem.history.params) == 2
    aaae(base_problem.history.params[0], [1, 2, 3])
    aaae(base_problem.history.params[1], [4, 5, 6])
    assert base_problem.history.fun == [14, 77]
    assert base_problem.history.task == [EvalTask.FUN, EvalTask.FUN]
    assert base_problem.history.batches == [0, 0]


def test_history_with_jac(base_problem):
    base_problem.jac(np.array([1, 2, 3]))

    assert len(base_problem.history.params) == 1
    aaae(base_problem.history.params[0], [1, 2, 3])
    assert base_problem.history.fun == [None]
    assert base_problem.history.task == [EvalTask.JAC]
    assert base_problem.history.batches == [0]


def test_history_with_batch_jac(base_problem):
    base_problem.batch_jac(
        [np.array([1, 2, 3]), np.array([4, 5, 6])], n_cores=1, batch_size=2
    )
    assert len(base_problem.history.params) == 2
    aaae(base_problem.history.params[0], [1, 2, 3])
    aaae(base_problem.history.params[1], [4, 5, 6])
    assert base_problem.history.fun == [None, None]
    assert base_problem.history.task == [EvalTask.JAC, EvalTask.JAC]
    assert base_problem.history.batches == [0, 0]


def test_history_with_fun_and_jac(base_problem):
    base_problem.fun_and_jac(np.array([1, 2, 3]))

    assert len(base_problem.history.params) == 1
    aaae(base_problem.history.params[0], [1, 2, 3])
    assert base_problem.history.fun == [14]
    assert base_problem.history.task == [EvalTask.FUN_AND_JAC]
    assert base_problem.history.batches == [0]


def test_history_with_batch_fun_and_jac(base_problem):
    base_problem.batch_fun_and_jac(
        [np.array([1, 2, 3]), np.array([4, 5, 6])], n_cores=1, batch_size=2
    )
    assert len(base_problem.history.params) == 2
    aaae(base_problem.history.params[0], [1, 2, 3])
    aaae(base_problem.history.params[1], [4, 5, 6])
    assert base_problem.history.fun == [14, 77]
    assert base_problem.history.task == [EvalTask.FUN_AND_JAC, EvalTask.FUN_AND_JAC]
    assert base_problem.history.batches == [0, 0]


def test_history_with_exploration_fun(base_problem):
    base_problem.exploration_fun(
        [np.array([1, 2, 3]), np.array([4, 5, 6])], n_cores=1, batch_size=2
    )
    assert len(base_problem.history.params) == 2
    aaae(base_problem.history.params[0], [1, 2, 3])
    aaae(base_problem.history.params[1], [4, 5, 6])
    assert base_problem.history.fun == [14, 77]
    assert base_problem.history.task == [EvalTask.EXPLORATION, EvalTask.EXPLORATION]
    assert base_problem.history.batches == [0, 0]


def test_with_history_copy_constructor(base_problem):
    new = base_problem.with_new_history()
    new.fun(np.array([1, 2, 3]))

    assert len(new.history.params) == 1
    assert len(base_problem.history.params) == 0


# ======================================================================================
# test batch versions
# ======================================================================================


@pytest.mark.parametrize("n_cores", [1, 2])
def test_batch_fun(base_problem, n_cores):
    got = base_problem.batch_fun(
        [np.array([1, 2, 3]), np.array([4, 5, 6])], n_cores=n_cores
    )
    expected = [14, 77]
    assert got == expected


@pytest.mark.parametrize("n_cores", [1, 2])
def test_batch_jac(base_problem, n_cores):
    got = base_problem.batch_jac(
        [np.array([1, 2, 3]), np.array([4, 5, 6])], n_cores=n_cores
    )
    expected = [2 * np.array([1, 2, 3]), 2 * np.array([4, 5, 6])]
    aaae(got[0], expected[0])
    aaae(got[1], expected[1])


@pytest.mark.parametrize("n_cores", [1, 2])
def test_batch_fun_and_jac(base_problem, n_cores):
    res = base_problem.batch_fun_and_jac(
        [np.array([1, 2, 3]), np.array([4, 5, 6])], n_cores=n_cores
    )
    got_fun = [r[0] for r in res]
    got_jac = [r[1] for r in res]
    expected_fun = [14, 77]
    expected_jac = [2 * np.array([1, 2, 3]), 2 * np.array([4, 5, 6])]
    assert got_fun == expected_fun
    aaae(got_jac, expected_jac)


# ======================================================================================
# test sign flipping
# ======================================================================================


@pytest.fixture
def max_problem(base_problem):
    """Flip the sign of the functions.

    The sign should be flipped back by InternalOptimizationProblem such that in the end
    the same values for fun, jac, and fun_and_jac are returned as for the base_problem.

    """

    def fun(params):
        return ScalarFunctionValue(value=-params @ params)

    def jac(params):
        return -2 * params

    def fun_and_jac(params):
        return fun(params), jac(params)

    max_problem = copy(base_problem)
    max_problem._direction = Direction.MAXIMIZE
    max_problem._fun = fun
    max_problem._jac = jac
    max_problem._fun_and_jac = fun_and_jac

    return max_problem


def test_max_problem_fun(max_problem):
    got = max_problem.fun(np.array([1, 2, 3]))
    expected = 14
    assert got == expected


def test_max_problem_jac(max_problem):
    got = max_problem.jac(np.array([1, 2, 3]))
    expected = 2 * np.array([1, 2, 3])
    aaae(got, expected)


def test_max_problem_fun_and_jac(max_problem):
    got_fun, got_jac = max_problem.fun_and_jac(np.array([1, 2, 3]))
    expected_fun, expected_jac = (14, 2 * np.array([1, 2, 3]))
    assert got_fun == expected_fun
    aaae(got_jac, expected_jac)


def test_jac_via_numdiff(max_problem):
    max_problem._jac = None
    max_problem._fun_and_jac = None

    got = max_problem.jac(np.array([1, 2, 3]))
    expected = 2 * np.array([1, 2, 3])
    aaae(got, expected)


def test_fun_and_jac_via_numdiff(max_problem):
    max_problem._jac = None
    max_problem._fun_and_jac = None

    got_fun, got_jac = max_problem.fun_and_jac(np.array([1, 2, 3]))
    expected_fun, expected_jac = (14, 2 * np.array([1, 2, 3]))
    assert got_fun == expected_fun
    aaae(got_jac, expected_jac)


def test_max_problem_exploration_fun(max_problem):
    got = max_problem.exploration_fun(
        [np.array([1, 2, 3]), np.array([4, 5, 6])], n_cores=1
    )
    expected = [14, 77]
    assert got == expected


# ======================================================================================
# test pytree ls output and params
# ======================================================================================


@pytest.fixture
def pytree_problem(base_problem):
    def fun(params):
        assert isinstance(params, dict)
        return LeastSquaresFunctionValue(value=params)

    def jac(params):
        assert isinstance(params, dict)
        out = {}
        for outer_key in params:
            row = {}
            for inner_key in params:
                if inner_key == outer_key:
                    row[inner_key] = 1
                else:
                    row[inner_key] = 0
            out[outer_key] = row
        return out

    def fun_and_jac(params):
        assert isinstance(params, dict)
        return fun(params), jac(params)

    def derivative_flatten(tree, x):
        out = [list(row.values()) for row in tree.values()]
        return np.array(out)

    converter = Converter(
        params_to_internal=lambda x: np.array(list(x.values())),
        params_from_internal=lambda x: {
            k: v for k, v in zip(["a", "b", "c"], x, strict=False)
        },
        derivative_to_internal=derivative_flatten,
        has_transforming_constraints=False,
    )

    solver_type = AggregationLevel.LEAST_SQUARES

    direction = Direction.MINIMIZE

    bounds = InternalBounds(lower=None, upper=None)

    numdiff_options = NumdiffOptions()

    error_handling = ErrorHandling.RAISE

    batch_evaluator = process_batch_evaluator(batch_evaluator="joblib")

    linear_constraints = None

    nonlinear_constraints = None

    problem = InternalOptimizationProblem(
        fun=fun,
        jac=jac,
        fun_and_jac=fun_and_jac,
        converter=converter,
        solver_type=solver_type,
        direction=direction,
        bounds=bounds,
        numdiff_options=numdiff_options,
        error_handling=error_handling,
        error_penalty_func=None,
        batch_evaluator=batch_evaluator,
        linear_constraints=linear_constraints,
        nonlinear_constraints=nonlinear_constraints,
        logger=None,
    )

    return problem


def test_pytree_problem_fun(pytree_problem):
    got = pytree_problem.fun(np.array([1, 2, 3]))
    expected = np.array([1, 2, 3])
    aaae(got, expected)


def test_pytree_problem_fun_scalar_output(pytree_problem):
    pytree_problem._solver_type = AggregationLevel.SCALAR
    got = pytree_problem.fun(np.array([1, 2, 3]))
    expected = 14
    assert got == expected


def test_pytree_problem_jac(pytree_problem):
    got = pytree_problem.jac(np.array([1, 2, 3]))
    expected = np.eye(3)
    aaae(got, expected)


def test_pytree_problem_fun_and_jac(pytree_problem):
    got_fun, got_jac = pytree_problem.fun_and_jac(np.array([1, 2, 3]))
    expected_fun, expected_jac = np.array([1, 2, 3]), np.eye(3)
    aaae(got_jac, expected_jac)
    aaae(got_fun, expected_fun)


def test_pytree_problem_exploration_fun(pytree_problem):
    got = pytree_problem.exploration_fun(
        [np.array([1, 2, 3]), np.array([4, 5, 6])], n_cores=1
    )
    expected = [14, 77]
    assert got == expected


def test_numerical_jac_for_pytree_problem(pytree_problem):
    pytree_problem._jac = None
    pytree_problem._fun_and_jac = None

    got = pytree_problem.jac(np.array([1, 2, 3]))
    expected = np.eye(3)
    aaae(got, expected)


def test_numerical_fun_and_jac_for_pytree_problem(pytree_problem):
    pytree_problem._jac = None
    pytree_problem._fun_and_jac = None

    got_fun, got_jac = pytree_problem.fun_and_jac(np.array([1, 2, 3]))
    expected_fun, expected_jac = np.array([1, 2, 3]), np.eye(3)
    aaae(got_fun, expected_fun)
    aaae(got_jac, expected_jac)


# ======================================================================================
# test error penalty with minimize
# ======================================================================================


@pytest.fixture
def error_min_problem():
    """Set up a basic InternalOptimizationProblem that can be modified for tests."""

    def fun(params):
        raise ValueError("Test error")

    def jac(params):
        raise ValueError("Test error")

    def fun_and_jac(params):
        raise ValueError("Test error")

    converter = Converter(
        params_to_internal=lambda x: x,
        params_from_internal=lambda x: x,
        derivative_to_internal=lambda d, x: d,
        has_transforming_constraints=False,
    )

    solver_type = AggregationLevel.SCALAR

    direction = Direction.MINIMIZE

    bounds = InternalBounds(lower=None, upper=None)

    numdiff_options = NumdiffOptions()

    error_handling = ErrorHandling.CONTINUE

    batch_evaluator = process_batch_evaluator(batch_evaluator="joblib")

    linear_constraints = None

    nonlinear_constraints = None

    start_params = np.array([1, 2, 3])

    error_penalty_function = get_error_penalty_function(
        start_x=start_params,
        error_penalty=None,
        start_criterion=ScalarFunctionValue(14),
        direction=direction,
        solver_type=solver_type,
    )

    problem = InternalOptimizationProblem(
        fun=fun,
        jac=jac,
        fun_and_jac=fun_and_jac,
        converter=converter,
        solver_type=solver_type,
        direction=direction,
        bounds=bounds,
        numdiff_options=numdiff_options,
        error_handling=error_handling,
        error_penalty_func=error_penalty_function,
        batch_evaluator=batch_evaluator,
        linear_constraints=linear_constraints,
        nonlinear_constraints=nonlinear_constraints,
        logger=None,
    )

    return problem


def test_error_in_fun_minimize(error_min_problem):
    got = error_min_problem.fun(np.array([2, 3, 4]))
    expected = 28 + CRITERION_PENALTY_CONSTANT + np.sqrt(3) * CRITERION_PENALTY_SLOPE
    assert np.allclose(got, expected)


def test_error_in_jac_minimize(error_min_problem):
    got = error_min_problem.jac(np.array([2, 3, 4]))
    expected = np.full(3, CRITERION_PENALTY_SLOPE) / np.sqrt(3)
    aaae(got, expected)


def test_error_in_fun_and_jac_minimize(error_min_problem):
    got_fun, got_jac = error_min_problem.fun_and_jac(np.array([2, 3, 4]))
    expected_fun = (
        28 + CRITERION_PENALTY_CONSTANT + np.sqrt(3) * CRITERION_PENALTY_SLOPE
    )
    expected_jac = np.full(3, CRITERION_PENALTY_SLOPE) / np.sqrt(3)
    assert np.allclose(got_fun, expected_fun)
    aaae(got_jac, expected_jac)


def test_error_in_numerical_jac_minimize(error_min_problem):
    error_min_problem._jac = None
    error_min_problem._fun_and_jac = None

    got = error_min_problem.jac(np.array([2, 3, 4]))
    expected = np.full(3, CRITERION_PENALTY_SLOPE) / np.sqrt(3)
    aaae(got, expected)


def test_error_in_exploration_fun_minimize(error_min_problem):
    got = error_min_problem.exploration_fun(
        [np.array([2, 3, 4]), np.array([5, 6, 7])], n_cores=1
    )
    expected = [-np.inf, -np.inf]
    assert np.allclose(got, expected)


# ======================================================================================
# test error penalty with maximize
# ======================================================================================


@pytest.fixture
def error_max_problem(error_min_problem):
    problem = copy(error_min_problem)
    problem._direction = Direction.MAXIMIZE

    error_penalty_function = get_error_penalty_function(
        start_x=np.array([1, 2, 3]),
        error_penalty=None,
        start_criterion=ScalarFunctionValue(-14),
        direction=problem._direction,
        solver_type=problem._solver_type,
    )

    problem._error_penalty_func = error_penalty_function
    return problem


def test_error_in_fun_maximize(error_max_problem):
    got = error_max_problem.fun(np.array([2, 3, 4]))
    expected = 28 + CRITERION_PENALTY_CONSTANT + np.sqrt(3) * CRITERION_PENALTY_SLOPE
    assert np.allclose(got, expected)


def test_error_in_jac_maximize(error_max_problem):
    got = error_max_problem.jac(np.array([2, 3, 4]))
    expected = np.full(3, CRITERION_PENALTY_SLOPE) / np.sqrt(3)
    aaae(got, expected)


def test_error_in_fun_and_jac_maximize(error_max_problem):
    got_fun, got_jac = error_max_problem.fun_and_jac(np.array([2, 3, 4]))
    expected_fun = (
        28 + CRITERION_PENALTY_CONSTANT + np.sqrt(3) * CRITERION_PENALTY_SLOPE
    )
    expected_jac = np.full(3, CRITERION_PENALTY_SLOPE) / np.sqrt(3)
    assert np.allclose(got_fun, expected_fun)
    aaae(got_jac, expected_jac)


def test_error_in_numerical_jac_maximize(error_max_problem):
    error_max_problem._jac = None
    error_max_problem._fun_and_jac = None

    got = error_max_problem.jac(np.array([2, 3, 4]))
    expected = np.full(3, CRITERION_PENALTY_SLOPE) / np.sqrt(3)
    aaae(got, expected)


def test_error_in_exploration_fun_maximize(error_max_problem):
    got = error_max_problem.exploration_fun(
        [np.array([2, 3, 4]), np.array([5, 6, 7])], n_cores=1
    )
    expected = [-np.inf, -np.inf]
    assert np.allclose(got, expected)
