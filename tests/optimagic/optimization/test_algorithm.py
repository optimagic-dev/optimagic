from dataclasses import dataclass, fields

import numpy as np
import pytest

from optimagic.algorithms import ALL_ALGORITHMS
from optimagic.exceptions import InvalidAlgoInfoError, InvalidAlgoOptionError
from optimagic.optimization.algorithm import AlgoInfo, Algorithm, InternalOptimizeResult
from optimagic.optimization.history import HistoryEntry
from optimagic.typing import (
    AggregationLevel,
    EvalTask,
    NonNegativeFloat,
    NonNegativeInt,
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


# ======================================================================================
# Test the type conversions of algo options
# ======================================================================================


def test_field_types_are_type_objects():
    # Guard: this module must NOT use `from __future__ import annotations`,
    # otherwise the tests below no longer cover the type-object code path of the
    # option conversion. The stringified-annotations path is covered in
    # test_algorithm_future_annotations.py.
    field_types = {f.name: f.type for f in fields(DummyAlgorithm)}
    assert field_types["stopping_maxiter"] == PositiveInt
    assert field_types["initial_radius"] == PositiveFloat


def test_algorithm_does_type_conversion():
    algo = DummyAlgorithm(
        initial_radius="1.0",
        max_radius="10.0",
        convergence_ftol_rel="1e-6",
        stopping_maxiter="1000",
    )

    assert isinstance(algo.initial_radius, float)
    assert algo.initial_radius == 1.0
    assert isinstance(algo.max_radius, float)
    assert algo.max_radius == 10.0
    assert isinstance(algo.convergence_ftol_rel, float)
    assert algo.convergence_ftol_rel == 1e-6
    assert isinstance(algo.stopping_maxiter, int)
    assert algo.stopping_maxiter == 1000


def test_algorithm_does_type_conversion_in_with_option():
    algo = DummyAlgorithm()
    new_algo = algo.with_option(
        initial_radius="2.0",
        max_radius="20.0",
    )

    assert isinstance(new_algo.initial_radius, float)
    assert new_algo.initial_radius == 2.0
    assert isinstance(new_algo.max_radius, float)
    assert new_algo.max_radius == 20.0


def test_algorithm_converts_float_to_int():
    algo = DummyAlgorithm(stopping_maxiter=1000.0)
    assert isinstance(algo.stopping_maxiter, int)
    assert algo.stopping_maxiter == 1000


def test_error_with_negative_radius():
    with pytest.raises(Exception):  # noqa: B017
        DummyAlgorithm(initial_radius=-1.0)


# ======================================================================================
# Test type conversion works for all registered algorithms
# ======================================================================================

# Field types are type objects in modules without `from __future__ import
# annotations` and annotation strings in modules with it. Both must be coerced.
INT_ANNOTATIONS = (
    int,
    PositiveInt,
    NonNegativeInt,
    "int",
    "PositiveInt",
    "NonNegativeInt",
)


def _int_options_with_int_defaults(cls):
    out = {}
    for field in fields(cls):
        has_int_annotation = any(field.type == t for t in INT_ANNOTATIONS)
        has_int_default = isinstance(field.default, int) and not isinstance(
            field.default, bool
        )
        if has_int_annotation and has_int_default:
            out[field.name] = field.default
    return out


@pytest.mark.parametrize("cls", ALL_ALGORITHMS.values(), ids=ALL_ALGORITHMS.keys())
def test_int_options_are_coerced_for_all_algorithms(cls):
    """Passing floats for int-typed options must result in int attributes.

    This guards against the option conversion in Algorithm.__post_init__ being
    silently skipped, as happened for optimizer modules with postponed annotations
    where the field type is a string rather than a type object.

    """
    int_options = _int_options_with_int_defaults(cls)
    if not int_options:
        pytest.skip("Algorithm has no int-typed options with int defaults.")

    algo = cls(**{name: float(default) for name, default in int_options.items()})

    for name, default in int_options.items():
        value = getattr(algo, name)
        assert isinstance(value, int), f"Option {name} was not coerced to int."
        assert value == default
