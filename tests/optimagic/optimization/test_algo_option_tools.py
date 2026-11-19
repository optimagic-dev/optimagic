import warnings
from dataclasses import dataclass, field
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


def _translate_old_restarts(value, options):
    return {
        "restart_options.use_restarts": value["on"],
        "restart_options.max_restarts": value["n"],
    }


@option_group
@dataclass(frozen=True)
class NoiseOptions:
    noise_level: float | None = None
    scale_factor: float = 1.0


def _translate_old_noise_level(value, options):
    # A noise level used to be the only way to declare the function as noisy.
    return {"noisy.noise_level": value} if value else {"noisy": False}


def _translate_old_noise_scale(value, options):
    # The scale only ever had an effect together with a noise level.
    return {"noisy.scale_factor": value} if options.get("old_noise_level") else {}


DEPRECATED = (
    DeprecatedOption("old_maxiter", "stopping_maxiter", removed_in="9.9.0"),
    DeprecatedOption("n_restarts", "restart_options.max_restarts", removed_in="9.9.0"),
    DeprecatedOption("restarts", _translate_old_restarts, removed_in="9.9.0"),
    DeprecatedOption("obsolete", lambda value, options: {}, removed_in="9.9.0"),
    DeprecatedOption("fast", lambda value, options: {"speed": 2}, removed_in="9.9.0"),
    DeprecatedOption("faster", lambda value, options: {"speed": 3}, removed_in="9.9.0"),
    DeprecatedOption("old_noise_level", _translate_old_noise_level, removed_in="9.9.0"),
    DeprecatedOption("old_noise_scale", _translate_old_noise_scale, removed_in="9.9.0"),
    DeprecatedOption("not_noisy", lambda value, options: {"noisy": False}, "9.9.0"),
)


@mark.deprecated_options(DEPRECATED)
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
)
@dataclass(frozen=True)
class Dummy(Algorithm):
    stopping_maxiter: PositiveInt = 100
    speed: int = 1
    restart_options: RestartOptions = field(default_factory=RestartOptions)
    noisy: bool | NoiseOptions = False

    def _solve_internal_problem(self, problem, x0):
        return InternalOptimizeResult(x=x0, fun=problem.fun(x0))


def make_dummy(**options: Any) -> Dummy:
    """Create a Dummy from options that static type checkers cannot verify."""
    return Dummy(**options)


def make_group(**entries: Any) -> RestartOptions:
    """Create RestartOptions from entries that static type checkers cannot verify."""
    return RestartOptions(**entries)


# ======================================================================================
# option groups
# ======================================================================================


def test_option_group_is_validated_on_creation():
    with pytest.raises(InvalidAlgoOptionError, match="RestartOptions"):
        RestartOptions(max_restarts=-1)


def test_option_group_converts_values_like_algorithms_do():
    group = make_group(max_restarts=3.0)
    assert group.max_restarts == 3
    assert isinstance(group.max_restarts, int)


def test_option_group_rejects_unknown_entries():
    with pytest.raises(InvalidAlgoOptionError, match="unknown"):
        make_group(unknown=1)


def test_algorithm_accepts_option_group_as_instance():
    group = RestartOptions(max_restarts=3)
    assert make_dummy(restart_options=group).restart_options == group


@pytest.mark.parametrize("group", [{"use_restarts": True}, {}, None, 1])
def test_algorithm_rejects_everything_but_instances_of_the_option_group(group):
    with pytest.raises(InvalidAlgoOptionError, match="instance of RestartOptions"):
        make_dummy(restart_options=group)
    with pytest.raises(InvalidAlgoOptionError, match="instance of RestartOptions"):
        Dummy().with_option(restart_options=group)


def test_option_group_is_rejected_as_dict_in_algo_options():
    with pytest.raises(InvalidAlgoOptionError, match="instance of RestartOptions"):
        om.minimize(
            lambda x: x @ x,
            np.arange(3.0),
            algorithm=Dummy,
            algo_options={"restart_options": {"use_restarts": True}},
        )


def test_deprecated_options_decorator_has_to_be_placed_above_minimizer():
    @dataclass(frozen=True)
    class Unmarked(Algorithm):
        def _solve_internal_problem(self, problem, x0):
            return InternalOptimizeResult(x=x0, fun=problem.fun(x0))

    with pytest.raises(TypeError, match="above the `mark.minimizer` decorator"):
        mark.deprecated_options(DEPRECATED)(Unmarked)


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
    with pytest.warns(
        FutureWarning, match=r"restart_options=RestartOptions\(max_restarts=3\)"
    ):
        algo = make_dummy(n_restarts=3)
    assert algo.restart_options == RestartOptions(max_restarts=3)


def test_option_moved_into_group_keeps_other_entries_of_configured_algorithm():
    algo = Dummy(restart_options=RestartOptions(use_restarts=True))
    with pytest.warns(FutureWarning):
        algo = algo.with_option(n_restarts=3)
    assert algo.restart_options == RestartOptions(use_restarts=True, max_restarts=3)


def test_converter_can_set_several_options():
    with pytest.warns(
        FutureWarning, match=r"RestartOptions\(use_restarts=True, max_restarts=4\)"
    ):
        algo = make_dummy(restarts={"on": True, "n": 4})
    assert algo.restart_options == RestartOptions(use_restarts=True, max_restarts=4)


def test_option_without_replacement():
    with pytest.warns(FutureWarning, match="The value you passed has no effect"):
        algo = make_dummy(obsolete=1)
    assert algo == Dummy()


def test_new_spelling_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        algo = Dummy(stopping_maxiter=5)
        algo.with_option(restart_options=RestartOptions(max_restarts=2))


@pytest.mark.parametrize(
    "options",
    [
        {"old_maxiter": 5, "stopping_maxiter": 6},
        {"n_restarts": 3, "restart_options": RestartOptions(use_restarts=True)},
    ],
)
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_old_and_new_spelling_together_raise(options):
    with pytest.raises(InvalidAlgoOptionError, match="cannot be combined"):
        make_dummy(**options)
    with pytest.raises(InvalidAlgoOptionError, match="cannot be combined"):
        Dummy().with_option(**options)


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
def test_converted_option_cannot_be_combined_with_its_target():
    with pytest.raises(InvalidAlgoOptionError, match="cannot be combined"):
        make_dummy(fast=True, speed=2)


# ======================================================================================
# deprecated options that move into an option that is a boolean or an option group
# ======================================================================================


@pytest.mark.parametrize("noisy", [True, False, NoiseOptions(scale_factor=2.0)])
def test_boolean_or_option_group_is_accepted(noisy):
    assert Dummy(noisy=noisy).noisy == noisy


def test_dictionary_is_rejected_for_boolean_or_option_group():
    with pytest.raises(InvalidAlgoOptionError, match="noisy"):
        make_dummy(noisy={"scale_factor": 2.0})


def test_entry_of_deprecated_option_switches_the_feature_on():
    new_spelling = r"noisy=NoiseOptions\(noise_level=0.1\)"
    with pytest.warns(FutureWarning, match=new_spelling):
        algo = make_dummy(old_noise_level=0.1)
    assert algo.noisy == NoiseOptions(noise_level=0.1)


@pytest.mark.parametrize("noisy", [True, False])
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_entry_of_deprecated_option_replaces_boolean_of_configured_algorithm(noisy):
    algo = Dummy(noisy=noisy).with_option(old_noise_level=0.1)
    assert algo.noisy == NoiseOptions(noise_level=0.1)


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_entry_of_deprecated_option_keeps_entries_of_configured_algorithm():
    algo = Dummy(noisy=NoiseOptions(scale_factor=2.0)).with_option(old_noise_level=0.1)
    assert algo.noisy == NoiseOptions(noise_level=0.1, scale_factor=2.0)


def test_deprecated_option_can_switch_the_feature_off():
    with pytest.warns(FutureWarning, match="Use `noisy=False` instead"):
        algo = make_dummy(old_noise_level=None)
    assert algo.noisy is False


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_converter_can_depend_on_other_options():
    alone = make_dummy(old_noise_scale=3.0)
    together = make_dummy(old_noise_scale=3.0, old_noise_level=0.1)

    assert alone.noisy is False
    assert together.noisy == NoiseOptions(noise_level=0.1, scale_factor=3.0)


@pytest.mark.parametrize("noisy", [True, NoiseOptions()])
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_deprecated_entry_cannot_be_combined_with_boolean_or_option_group(noisy):
    with pytest.raises(InvalidAlgoOptionError, match="cannot be combined"):
        make_dummy(old_noise_level=0.1, noisy=noisy)


@pytest.mark.parametrize(
    "options",
    [
        {"old_noise_level": 0.1, "not_noisy": True},
        {"not_noisy": True, "old_noise_level": 0.1},
    ],
)
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_switching_off_contradicts_setting_an_entry(options):
    with pytest.raises(InvalidAlgoOptionError, match="contradicts"):
        make_dummy(**options)


def test_unknown_option_still_raises():
    with pytest.raises(InvalidAlgoOptionError):
        make_dummy(not_an_option=1)
    with pytest.raises(InvalidAlgoOptionError):
        Dummy().with_option(not_an_option=1)


def test_replace_deprecated_options_does_not_change_its_input():
    options = {"old_maxiter": 5}
    with pytest.warns(FutureWarning):
        out = replace_deprecated_options(options, Dummy)
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
