"""Test algo option type conversion in modules with postponed annotations.

With `from __future__ import annotations`, dataclass field types are annotation
strings rather than type objects. This silently disabled the type conversion in
`Algorithm.__post_init__` for all optimizer modules using the import. The tests in
this module mirror the type conversion tests in test_algorithm.py for a dummy
algorithm defined under postponed annotations; test_algorithm.py covers the case
without the import.

"""

from __future__ import annotations

from dataclasses import dataclass, fields

import pytest

from optimagic.exceptions import InvalidAlgoOptionError
from optimagic.optimization.algorithm import Algorithm, InternalOptimizeResult
from optimagic.optimization.history import HistoryEntry
from optimagic.typing import (
    EvalTask,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
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


def test_field_types_are_annotation_strings():
    # Guard: this module must keep `from __future__ import annotations`, otherwise
    # the tests below no longer cover the stringified-annotations code path.
    field_types = {f.name: f.type for f in fields(DummyAlgorithm)}
    assert field_types["stopping_maxiter"] == "PositiveInt"
    assert field_types["initial_radius"] == "PositiveFloat"


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


def test_algorithm_converts_float_to_int():
    algo = DummyAlgorithm(stopping_maxiter=1000.0)
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


def test_error_with_negative_radius():
    with pytest.raises(InvalidAlgoOptionError):
        DummyAlgorithm(initial_radius=-1.0)


def test_error_with_negative_maxiter():
    with pytest.raises(InvalidAlgoOptionError):
        DummyAlgorithm(stopping_maxiter=-1)
