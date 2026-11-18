import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

import optimagic as om
from optimagic import mark
from optimagic.exceptions import InvalidAlgoOptionError
from optimagic.optimization.algo_option_tools import (
    DeprecatedOption,
    option_group,
    replace_deprecated_options,
    resolve_changed_default,
)
from optimagic.optimization.algorithm import Algorithm, InternalOptimizeResult
from optimagic.typing import AggregationLevel, PositiveInt


@option_group
@dataclass(frozen=True)
class RestartOptions:
    use_restarts: bool = False
    max_restarts: PositiveInt = 10


def _translate_old_restarts(value):
    return {
        "restart_options.use_restarts": value["on"],
        "restart_options.max_restarts": value["n"],
    }


DEPRECATED = (
    DeprecatedOption("old_maxiter", "stopping_maxiter", removed_in="9.9.0"),
    DeprecatedOption("n_restarts", "restart_options.max_restarts", removed_in="9.9.0"),
    DeprecatedOption("restarts", _translate_old_restarts, removed_in="9.9.0"),
    DeprecatedOption("obsolete", lambda value: {}, removed_in="9.9.0"),
    DeprecatedOption("fast", lambda value: {"speed": 2}, removed_in="9.9.0"),
    DeprecatedOption("faster", lambda value: {"speed": 3}, removed_in="9.9.0"),
)


@mark.minimizer(
    name="dummy",
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
    deprecated_options=DEPRECATED,
)
@dataclass(frozen=True)
class Dummy(Algorithm):
    stopping_maxiter: PositiveInt = 100
    speed: int = 1
    restart_options: RestartOptions = RestartOptions()

    def _solve_internal_problem(self, problem, x0):
        return InternalOptimizeResult(x=x0, fun=problem.fun(x0))


def make_dummy(**options: Any) -> Dummy:
    """Create a Dummy from options that static type checkers cannot verify."""
    return Dummy(**options)


# ======================================================================================
# option groups
# ======================================================================================


def test_option_group_is_validated_on_creation():
    with pytest.raises(InvalidAlgoOptionError, match="RestartOptions"):
        RestartOptions(max_restarts=-1)


def test_algorithm_accepts_option_group_as_dict():
    algo = make_dummy(restart_options={"use_restarts": True})
    assert algo.restart_options == RestartOptions(use_restarts=True)


def test_algorithm_accepts_option_group_as_instance():
    group = RestartOptions(max_restarts=3)
    assert make_dummy(restart_options=group).restart_options == group


@pytest.mark.parametrize("group", [{"max_restarts": -1}, {"unknown": 1}])
def test_invalid_option_group_dict_raises(group):
    with pytest.raises(InvalidAlgoOptionError, match="dummy"):
        make_dummy(restart_options=group)


# ======================================================================================
# deprecated options
# ======================================================================================


def test_renamed_option_in_constructor():
    with pytest.warns(FutureWarning, match="Use `stopping_maxiter=5` instead"):
        algo = make_dummy(old_maxiter=5)
    assert algo.stopping_maxiter == 5


def test_deprecation_warning_names_algorithm_and_removal_version():
    with pytest.warns(FutureWarning, match="`old_maxiter` of the algorithm dummy") as w:
        make_dummy(old_maxiter=5)
    assert "removed in optimagic 9.9.0" in str(w[0].message)


def test_renamed_option_in_with_option():
    with pytest.warns(FutureWarning, match="old_maxiter"):
        algo = Dummy().with_option(old_maxiter=5)
    assert algo.stopping_maxiter == 5


def test_renamed_option_in_algo_options():
    with pytest.warns(FutureWarning, match="old_maxiter"):
        res = om.minimize(
            lambda x: x @ x,
            np.arange(3.0),
            algorithm=Dummy,
            algo_options={"old_maxiter": 5},
        )
    assert res.algorithm == "dummy"


def test_option_moved_into_group():
    with pytest.warns(FutureWarning, match=r"restart_options=\{'max_restarts': 3\}"):
        algo = make_dummy(n_restarts=3)
    assert algo.restart_options == RestartOptions(max_restarts=3)


def test_option_moved_into_group_keeps_other_entries_of_configured_algorithm():
    algo = make_dummy(restart_options={"use_restarts": True})
    with pytest.warns(FutureWarning):
        algo = algo.with_option(n_restarts=3)
    assert algo.restart_options == RestartOptions(use_restarts=True, max_restarts=3)


def test_option_moved_into_group_is_merged_with_new_group_dict():
    with pytest.warns(FutureWarning):
        algo = make_dummy(n_restarts=3, restart_options={"use_restarts": True})
    assert algo.restart_options == RestartOptions(use_restarts=True, max_restarts=3)


def test_converter_can_set_several_options():
    with pytest.warns(FutureWarning, match="'use_restarts': True, 'max_restarts': 4"):
        algo = make_dummy(restarts={"on": True, "n": 4})
    assert algo.restart_options == RestartOptions(use_restarts=True, max_restarts=4)


def test_option_without_replacement():
    with pytest.warns(FutureWarning, match="has no effect anymore"):
        algo = make_dummy(obsolete=1)
    assert algo == Dummy()


def test_new_spelling_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        make_dummy(stopping_maxiter=5).with_option(restart_options={"max_restarts": 2})


@pytest.mark.parametrize(
    "options",
    [
        {"old_maxiter": 5, "stopping_maxiter": 6},
        {"n_restarts": 3, "restart_options": {"max_restarts": 4}},
        {"n_restarts": 3, "restart_options": RestartOptions()},
    ],
)
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_old_and_new_spelling_together_raise(options):
    with pytest.raises(InvalidAlgoOptionError, match="cannot be combined"):
        make_dummy(**options)


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_contradicting_deprecated_options_raise():
    with pytest.raises(InvalidAlgoOptionError, match="contradicts"):
        make_dummy(fast=True, faster=True)
    with pytest.raises(InvalidAlgoOptionError, match="contradicts"):
        make_dummy(n_restarts=3, restarts={"on": True, "n": 4})


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_agreeing_deprecated_options_are_accepted():
    algo = make_dummy(n_restarts=4, restarts={"on": True, "n": 4})
    assert algo.restart_options.max_restarts == 4


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_converted_option_accepts_the_same_explicit_value_of_its_side_effect():
    assert make_dummy(fast=True, speed=2).speed == 2
    with pytest.raises(InvalidAlgoOptionError, match="cannot be combined"):
        make_dummy(fast=True, speed=5)


def test_unknown_option_still_raises():
    with pytest.raises(InvalidAlgoOptionError):
        make_dummy(not_an_option=1)
    with pytest.raises(InvalidAlgoOptionError):
        Dummy().with_option(not_an_option=1)


def test_replace_deprecated_options_does_not_change_its_input():
    options = {"old_maxiter": 5}
    with pytest.warns(FutureWarning):
        out = replace_deprecated_options(options, DEPRECATED, "dummy")
    assert options == {"old_maxiter": 5}
    assert out == {"stopping_maxiter": 5}


# ======================================================================================
# changed defaults
# ======================================================================================


def test_resolve_changed_default_warns_if_option_was_not_set():
    with pytest.warns(FutureWarning, match="from 'full' to '2D' in optimagic 9.9.0"):
        value = resolve_changed_default(
            None,
            option="subspace_solver",
            old_default="full",
            new_default="2D",
            algo_name="dummy",
            changed_in="9.9.0",
        )
    assert value == "full"


@pytest.mark.parametrize("value", ["full", "2D"])
def test_resolve_changed_default_is_silent_if_option_was_set(value):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = resolve_changed_default(
            value,
            option="subspace_solver",
            old_default="full",
            new_default="2D",
            algo_name="dummy",
            changed_in="9.9.0",
        )
    assert out == value
