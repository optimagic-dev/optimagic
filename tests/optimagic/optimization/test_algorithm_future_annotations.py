"""Test option validation in modules with postponed annotation evaluation.

With ``from __future__ import annotations``, dataclass field types are stored as
strings. The old converter-based validation silently skipped such fields; the
pydantic-based validation must resolve them at runtime and behave exactly like in
modules without the future import.

"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from optimagic import mark
from optimagic.exceptions import InvalidAlgoOptionError
from optimagic.optimization.algorithm import Algorithm, InternalOptimizeResult
from optimagic.typing import AggregationLevel, PositiveFloat, PositiveInt


@mark.minimizer(
    name="dummy_postponed_algorithm",
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
class DummyPostponedAlgorithm(Algorithm):
    initial_radius: PositiveFloat = 1.0
    stopping_maxiter: PositiveInt = 1000
    n_points: PositiveInt | None = None

    def _solve_internal_problem(self, problem, x0):
        return InternalOptimizeResult(x=x0, fun=0.0, success=True)


def test_field_types_are_annotation_strings():
    # Guard: if this fails, the module no longer covers the postponed annotations
    # code path and the test setup must be adapted.
    field_type = DummyPostponedAlgorithm.__dataclass_fields__["stopping_maxiter"].type
    assert isinstance(field_type, str)


def test_type_conversion_works_with_postponed_annotations():
    algo = DummyPostponedAlgorithm(initial_radius="2.0", stopping_maxiter=500.0)
    assert isinstance(algo.initial_radius, float)
    assert algo.initial_radius == 2.0
    assert isinstance(algo.stopping_maxiter, int)
    assert algo.stopping_maxiter == 500


def test_optional_option_is_converted_with_postponed_annotations():
    algo = DummyPostponedAlgorithm(n_points=3.0)
    assert isinstance(algo.n_points, int)
    assert algo.n_points == 3
    assert DummyPostponedAlgorithm(n_points=None).n_points is None


def test_validation_works_with_postponed_annotations():
    with pytest.raises(InvalidAlgoOptionError):
        DummyPostponedAlgorithm(initial_radius=-1.0)


def test_validation_works_in_with_option_with_postponed_annotations():
    algo = DummyPostponedAlgorithm()
    with pytest.raises(InvalidAlgoOptionError):
        algo.with_option(stopping_maxiter=-1)
