from dataclasses import dataclass

import numpy as np
import pytest

from optimagic import mark
from optimagic.exceptions import InvalidAlgoInfoError, InvalidAlgoOptionError
from optimagic.optimization.algorithm import AlgoInfo, Algorithm, InternalOptimizeResult
from optimagic.optimization.history import HistoryEntry
from optimagic.typing import (
    AggregationLevel,
    EvalTask,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
)

# ======================================================================================
# Test AlgoInfo does proper validation of arguments
# ======================================================================================

INVALID_ALGO_INFO_KWARGS = [
    {"name": 3},
    {"solver_type": "scalar"},
    {"is_available": "yes"},
    {"is_global": "no"},
    {"needs_jac": "yes"},
    {"needs_hess": "no"},
    {"needs_bounds": "no"},
    {"supports_parallelism": "yes"},
    {"supports_bounds": "no"},
    {"supports_infinite_bounds": "no"},
    {"supports_linear_constraints": "yes"},
    {"supports_nonlinear_constraints": "no"},
    {"disable_history": "no"},
]


@pytest.mark.parametrize("kwargs", INVALID_ALGO_INFO_KWARGS)
def test_algo_info_validation(kwargs):
    valid_kwargs = {
        "name": "test",
        "solver_type": AggregationLevel.LEAST_SQUARES,
        "is_available": True,
        "is_global": True,
        "needs_jac": True,
        "needs_hess": True,
        "needs_bounds": True,
        "supports_parallelism": True,
        "supports_bounds": True,
        "supports_infinite_bounds": True,
        "supports_linear_constraints": True,
        "supports_nonlinear_constraints": True,
        "disable_history": True,
    }

    combined_kwargs = {**valid_kwargs, **kwargs}
    msg = "The following arguments to AlgoInfo or `mark.minimizer` are invalid"
    with pytest.raises(InvalidAlgoInfoError, match=msg):
        AlgoInfo(**combined_kwargs)


# ======================================================================================
# Test InternalOptimizeResult does proper validation of arguments
# ======================================================================================


INVALID_INTERNAL_OPTIMIZE_RESULT_KWARGS = [
    {"x": 3},
    {"fun": [1, 2, 3]},
    {"success": "successful"},
    {"message": 3},
    {"n_fun_evals": "3"},
    {"n_jac_evals": "3"},
    {"n_hess_evals": "3"},
    {"n_iterations": "3"},
    {"status": "3"},
    {"jac": "3"},
    {"hess": "3"},
    {"hess_inv": "3"},
    {"max_constraint_violation": "3"},
]


@pytest.mark.parametrize("kwargs", INVALID_INTERNAL_OPTIMIZE_RESULT_KWARGS)
def test_internal_optimize_result_validation(kwargs):
    valid_kwargs = {
        "x": np.array([1, 2, 3]),
        "fun": 3.0,
        "success": True,
        "message": "success",
        "n_fun_evals": 3,
        "n_jac_evals": 3,
        "n_hess_evals": 3,
        "n_iterations": 3,
        "status": 3,
        "jac": np.array([1, 2, 3]),
        "hess": np.array([1, 2, 3]),
        "hess_inv": np.array([1, 2, 3]),
        "max_constraint_violation": 3.0,
    }

    combined_kwargs = {**valid_kwargs, **kwargs}
    msg = "The following arguments to InternalOptimizeResult are invalid"
    with pytest.raises(TypeError, match=msg):
        InternalOptimizeResult(**combined_kwargs)


# ======================================================================================
# Test the copy constructors of Algorithm
# ======================================================================================


@mark.minimizer(
    name="dummy_algorithm",
    solver_type=AggregationLevel.SCALAR,
    is_available=True,
    is_global=False,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=False,
    supports_parallelism=False,
    supports_bounds=False,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class DummyAlgorithm(Algorithm):
    initial_radius: PositiveFloat = 1.0
    max_radius: PositiveFloat = 10.0
    convergence_ftol_rel: NonNegativeFloat = 1e-6
    stopping_maxiter: PositiveInt = 1000

    def _solve_internal_problem(self, problem, x0):
        hist_entry = HistoryEntry(
            params=x0,
            fun=0.0,
            start_time=0.0,
            task=EvalTask.FUN,
        )
        problem.history.add_entry(hist_entry)
        return InternalOptimizeResult(x=x0, fun=0.0, success=True)


def test_with_option():
    algo = DummyAlgorithm()
    new_algo = algo.with_option(initial_radius=2.0, max_radius=20.0)
    assert new_algo is not algo
    assert new_algo.initial_radius == 2.0
    assert new_algo.max_radius == 20.0


def test_with_option_invalid_key():
    algo = DummyAlgorithm()
    with pytest.raises(InvalidAlgoOptionError):
        algo.with_option(invalid_key=2.0)


def test_with_stopping():
    algo = DummyAlgorithm()
    new_algo = algo.with_stopping(maxiter=2000)
    assert new_algo is not algo
    assert new_algo.stopping_maxiter == 2000


def test_with_stopping_with_full_option_name():
    algo = DummyAlgorithm()
    new_algo = algo.with_stopping(stopping_maxiter=2000)
    assert new_algo is not algo
    assert new_algo.stopping_maxiter == 2000


def test_with_stopping_invalid_key():
    algo = DummyAlgorithm()
    with pytest.raises(InvalidAlgoOptionError):
        algo.with_stopping(invalid_key=2000)


def test_with_convergence():
    algo = DummyAlgorithm()
    new_algo = algo.with_convergence(ftol_rel=1e-5)
    assert new_algo is not algo
    assert new_algo.convergence_ftol_rel == 1e-5


def test_with_convergence_with_full_option_name():
    algo = DummyAlgorithm()
    new_algo = algo.with_convergence(convergence_ftol_rel=1e-5)
    assert new_algo is not algo
    assert new_algo.convergence_ftol_rel == 1e-5


def test_with_convergence_invalid_key():
    algo = DummyAlgorithm()
    with pytest.raises(InvalidAlgoOptionError):
        algo.with_convergence(invalid_key=1e-5)


def test_with_option_if_applicable():
    algo = DummyAlgorithm()
    with pytest.warns(UserWarning):
        new_algo = algo.with_option_if_applicable(
            invalid=15,
            initial_radius=42,
        )
    assert new_algo is not algo
    assert new_algo.initial_radius == 42
