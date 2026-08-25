"""Test validation and conversion of algo options under stringified annotations.

With ``from __future__ import annotations``, dataclass field types are stored as
strings. The old converter-based validation silently skipped such fields; the
pydantic-based validation resolves them at runtime and must behave exactly like in
modules without the future import.

This module must mirror test_algo_option_validation.py, which contains the same
tests for a module with regular annotations.

"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from optimagic import mark
from optimagic.exceptions import InvalidAlgoOptionError
from optimagic.optimization.algorithm import Algorithm, InternalOptimizeResult
from optimagic.typing import AggregationLevel, PositiveFloat, PositiveInt


@mark.minimizer(
    name="dummy_stringified_algorithm",
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
    stopping_maxiter: PositiveInt = 1000
    n_points: PositiveInt | None = None

    def _solve_internal_problem(self, problem, x0):
        return InternalOptimizeResult(x=x0, fun=0.0, success=True)


def test_field_types_are_annotation_strings():
    # Guard: if this fails, the module no longer covers the code path for
    # stringified annotations and the mirrored test setup must be adapted.
    field_type = DummyAlgorithm.__dataclass_fields__["stopping_maxiter"].type
    assert isinstance(field_type, str)


def test_str_option_is_converted():
    algo = DummyAlgorithm(initial_radius="2.0")
    assert isinstance(algo.initial_radius, float)
    assert algo.initial_radius == 2.0


def test_float_option_is_converted_to_int():
    algo = DummyAlgorithm(stopping_maxiter=500.0)
    assert isinstance(algo.stopping_maxiter, int)
    assert algo.stopping_maxiter == 500


def test_optional_option_is_converted():
    algo = DummyAlgorithm(n_points=3.0)
    assert isinstance(algo.n_points, int)
    assert algo.n_points == 3
    assert DummyAlgorithm(n_points=None).n_points is None


def test_invalid_option_value_raises_error():
    with pytest.raises(InvalidAlgoOptionError):
        DummyAlgorithm(initial_radius=-1.0)


def test_invalid_option_name_raises_error():
    with pytest.raises(InvalidAlgoOptionError):
        DummyAlgorithm(this_is_not_an_option=1)


def test_conversion_works_in_with_option():
    algo = DummyAlgorithm().with_option(stopping_maxiter="500")
    assert isinstance(algo.stopping_maxiter, int)
    assert algo.stopping_maxiter == 500


def test_validation_works_in_with_option():
    with pytest.raises(InvalidAlgoOptionError):
        DummyAlgorithm().with_option(stopping_maxiter=-1)
