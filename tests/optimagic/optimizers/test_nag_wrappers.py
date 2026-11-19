"""Wrapper tests for the optimizers from DFO-LS and Py-BOBYQA.

The tests check four things:

- contract: every argument and option of the upstream solvers is either mapped to an
  optimagic option or listed as unsupported, and the wrapper defaults agree with the
  upstream defaults. These tests fail if the upstream libraries change.
- forwarding: every option reaches the upstream solver under the right name.
- behavior: options have the documented effect on tiny problems.
- deprecations: every old option name still works, warns and translates correctly.

Run them with ``pytest -m nag_wrappers``.

"""

import inspect
import itertools
import warnings
from dataclasses import fields, is_dataclass, replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import optimagic as om
from optimagic.exceptions import InvalidAlgoOptionError
from optimagic.optimization.algo_options import STOPPING_MAXFUN
from optimagic.optimizers import nag_optimizers as nag
from optimagic.optimizers.nag_optimizers import NagDFOLS, NagPyBOBYQA

dfols = pytest.importorskip("dfols")
pybobyqa = pytest.importorskip("pybobyqa")

from dfols.params import ParameterList as DfolsParameterList  # noqa: E402
from pybobyqa.params import ParameterList as PybobyqaParameterList  # noqa: E402

pytestmark = pytest.mark.nag_wrappers

# Typed as Any because the tests use deprecated options and build options from
# tables, which type checkers cannot verify.
ALGORITHMS: dict[str, Any] = {"nag_dfols": NagDFOLS, "nag_pybobyqa": NagPyBOBYQA}
UPSTREAM = {"nag_dfols": dfols, "nag_pybobyqa": pybobyqa}

SOLVE_ARGUMENTS = {
    "nag_dfols": nag.DFOLS_SOLVE_ARGUMENTS,
    "nag_pybobyqa": nag.PYBOBYQA_SOLVE_ARGUMENTS,
}
ADVANCED_OPTIONS = {
    "nag_dfols": nag.DFOLS_ADVANCED_OPTIONS,
    "nag_pybobyqa": nag.PYBOBYQA_ADVANCED_OPTIONS,
}
UNSUPPORTED_SOLVE_ARGUMENTS = {
    "nag_dfols": nag.DFOLS_UNSUPPORTED_SOLVE_ARGUMENTS,
    "nag_pybobyqa": nag.PYBOBYQA_UNSUPPORTED_SOLVE_ARGUMENTS,
}
UNSUPPORTED_ADVANCED_OPTIONS = {
    "nag_dfols": nag.DFOLS_UNSUPPORTED_ADVANCED_OPTIONS,
    "nag_pybobyqa": nag.PYBOBYQA_UNSUPPORTED_ADVANCED_OPTIONS,
}
DEPRECATED_OPTIONS = {
    "nag_dfols": nag.DFOLS_DEPRECATED_OPTIONS,
    "nag_pybobyqa": nag.PYBOBYQA_DEPRECATED_OPTIONS,
}

# Options that are a boolean or an option group, with the class of the group. The
# boolean switches a feature on or off, the group switches it on with settings.
SWITCHABLE_GROUPS: dict[str, dict[str, Any]] = {
    "nag_dfols": {
        "noisy": nag.NagNoiseOptions,
        "restarts": nag.DfolsRestartOptions,
    },
    "nag_pybobyqa": {
        "noisy": nag.NagNoiseOptions,
        "restarts": nag.PybobyqaRestartOptions,
    },
}

# Arguments of the upstream solve functions that the wrapper fills itself.
FILLED_BY_WRAPPER = {"objfun", "x0", "bounds", "user_params"}

# Defaults where optimagic deliberately differs from upstream, with the upstream
# default. The reasons are given in the docstrings of the options.
DELIBERATE_DEFAULT_DEVIATIONS = {
    "nag_dfols": {
        "maxfun": None,
        "slow.thresh_for_slow": 1e-4,
        "model.rel_tol": 1e-20,
    },
    "nag_pybobyqa": {
        "maxfun": None,
        "model.abs_tol": -1e20,
    },
}

# Options where the solver replaces its static default while it runs. The wrapper
# default is None, such that the solver can do so.
DEFAULTS_SET_BY_SOLVER_AT_RUNTIME = {
    "nag_dfols": {
        "growing.full_rank.use_full_rank_interp",
        "growing.perturb_trust_region_step",
        "growing.delta_scale_new_dirns",
    },
    "nag_pybobyqa": set(),
}


# ======================================================================================
# Test problems and helpers
# ======================================================================================

OPTIMUM = np.array([1.0, 2.0, 3.0])
START = np.zeros(3)
ROSENBROCK_START = np.array([-1.2, 1.0])


def _residuals(x):
    """Residuals with optimum at OPTIMUM and a sum of squares of 1 at the optimum."""
    return np.concatenate([x - OPTIMUM, [1.0]])


def _rosenbrock_residuals(x):
    return np.array([10 * (x[1] - x[0] ** 2), 1 - x[0]])


def _as_fun(residuals, algo_name, noise_scale=0.0, seed=0):
    """Create the function type the algorithm needs, optionally with noise."""
    rng = np.random.default_rng(seed)

    def _noisy_residuals(x):
        out = residuals(x)
        return out + noise_scale * rng.normal(size=len(out))

    if algo_name == "nag_dfols":
        return om.mark.least_squares(_noisy_residuals)

    def _scalar(x):
        out = _noisy_residuals(x)
        return float(out @ out)

    return _scalar


def _group_class(algo_name, option):
    """Get the class of the option group of an option, or None if it has no group.

    The classes of some groups differ between the algorithms. For options that are
    always a group, the class is taken from the default value of the option.

    """
    if option in SWITCHABLE_GROUPS[algo_name]:
        return SWITCHABLE_GROUPS[algo_name][option]
    default = getattr(ALGORITHMS[algo_name](), option, None)
    return type(default) if is_dataclass(default) else None


def _group(algo_name, option, **entries):
    """Create an option group of an algorithm, e.g. for its ``restarts`` option."""
    return _group_class(algo_name, option)(**entries)


def _with_groups(algo_name, options):
    """Replace dictionaries with the entries of option groups by the groups.

    The tables in this module spell option groups as dictionaries of their entries,
    because the class of a group depends on the algorithm. Algorithms only accept
    instances of the group classes. Values of deprecated options stay dictionaries.

    """
    return {
        name: _group(algo_name, name, **value)
        if isinstance(value, dict) and _group_class(algo_name, name) is not None
        else value
        for name, value in options.items()
    }


def _nest(algo_name, options):
    """Convert {"group.entry": value} to keyword arguments of the algorithm."""
    out = {}
    for path, value in options.items():
        group, _, entry = path.partition(".")
        if entry:
            out.setdefault(group, {})[entry] = value
        else:
            out[path] = value
    return _with_groups(algo_name, out)


def _default(algo_name, path):
    """Get the default of an option or of an entry of its option group."""
    option, _, entry = path.partition(".")
    if entry:
        return getattr(_group(algo_name, option), entry)
    return getattr(ALGORITHMS[algo_name](), option)


def _option_paths(algo_name):
    """Paths of all options of an algorithm, with entries of groups spelled out.

    An option that is a boolean or an option group has a path for the option itself
    and one for each entry of the group.

    """
    out = []
    for field in fields(ALGORITHMS[algo_name]()):
        group_cls = _group_class(algo_name, field.name)
        if group_cls is None or field.name in SWITCHABLE_GROUPS[algo_name]:
            out.append(field.name)
        if group_cls is not None:
            out += [f"{field.name}.{entry.name}" for entry in fields(group_cls)]
    return out


def _is_entry_of_switchable_group(algo_name, path):
    option, _, entry = path.partition(".")
    return bool(entry) and option in SWITCHABLE_GROUPS[algo_name]


def _minimize(algo_name, residuals=_residuals, x0=START, noise_scale=0.0, **options):
    algo = ALGORITHMS[algo_name](**options)
    fun = _as_fun(residuals, algo_name, noise_scale=noise_scale)
    return om.minimize(fun, x0, algorithm=algo)


def _n_leading_axis_points(res, x0, radius):
    """Count evaluations after x0 that change one parameter of x0 by the radius."""
    count = 0
    for params in res.history.params[1:]:
        diff = np.asarray(params) - x0
        is_axis_point = np.sum(~np.isclose(diff, 0)) == 1
        if is_axis_point and np.isclose(np.abs(diff).max(), radius):
            count += 1
        else:
            break
    return count


@pytest.fixture()
def solve_calls(monkeypatch):
    """Record the keyword arguments with which the upstream solvers are called."""
    calls = {}
    for name, module in UPSTREAM.items():
        original = module.solve

        def spy(*args, _original=original, _name=name, **kwargs):
            calls[_name] = kwargs
            return _original(*args, **kwargs)

        monkeypatch.setattr(module, "solve", spy)
    return calls


@pytest.fixture()
def raw_results(monkeypatch):
    """Record the result objects that the upstream solvers return."""
    results = {}
    for name, module in UPSTREAM.items():
        original = module.solve

        def spy(*args, _original=original, _name=name, **kwargs):
            results[_name] = _original(*args, **kwargs)
            return results[_name]

        monkeypatch.setattr(module, "solve", spy)
    return results


# ======================================================================================
# Contract with the upstream libraries
# ======================================================================================


def _upstream_parameter_lists(algo_name, only_noisy=False):
    """Default advanced options of the solver in several contexts.

    The contexts differ in the number of parameters, interpolation points and function
    evaluations, in whether the function is noisy and in whether a global optimum is
    searched.

    Args:
        algo_name: The name of the algorithm.
        only_noisy: Whether only contexts with a noisy function are returned. These
            are the contexts in which the entries of ``noisy`` are passed.

    """
    if algo_name == "nag_dfols":
        contexts = [
            (False, DfolsParameterList(3, 4, 100, objfun_has_noise=False).params),
            (True, DfolsParameterList(3, 4, 100, objfun_has_noise=True).params),
            (True, DfolsParameterList(5, 9, 500, objfun_has_noise=True).params),
            (False, DfolsParameterList(5, 9, 500, objfun_has_noise=False).params),
        ]
    else:
        contexts = [
            (
                noisy,
                PybobyqaParameterList(
                    3, 7, 100, objfun_has_noise=noisy, seek_global_minimum=seek_global
                ).params,
            )
            for noisy, seek_global in itertools.product([False, True], repeat=2)
        ]
        contexts.append((False, PybobyqaParameterList(5, 11, 500).params))
    return [params for noisy, params in contexts if noisy or not only_noisy]


@pytest.mark.parametrize("algo_name", ALGORITHMS)
def test_all_solve_arguments_are_mapped_or_unsupported(algo_name):
    upstream = set(inspect.signature(UPSTREAM[algo_name].solve).parameters)
    mapped = set(SOLVE_ARGUMENTS[algo_name].values())
    unsupported = set(UNSUPPORTED_SOLVE_ARGUMENTS[algo_name])

    assert not mapped & unsupported
    assert upstream == mapped | unsupported | FILLED_BY_WRAPPER


@pytest.mark.parametrize("algo_name", ALGORITHMS)
def test_all_advanced_options_are_mapped_or_unsupported(algo_name):
    mapped = set(ADVANCED_OPTIONS[algo_name].values())
    unsupported = set(UNSUPPORTED_ADVANCED_OPTIONS[algo_name])

    assert not mapped & unsupported
    for params in _upstream_parameter_lists(algo_name):
        assert set(params) == mapped | unsupported


@pytest.mark.parametrize("algo_name", ALGORITHMS)
def test_all_options_of_the_wrapper_are_mapped(algo_name):
    mapped = set(SOLVE_ARGUMENTS[algo_name]) | set(ADVANCED_OPTIONS[algo_name])
    assert set(_option_paths(algo_name)) == mapped


@pytest.mark.parametrize(
    ("algo_name", "path", "upstream_name"),
    [
        (algo_name, path, upstream_name)
        for algo_name in ALGORITHMS
        for path, upstream_name in SOLVE_ARGUMENTS[algo_name].items()
    ],
)
def test_defaults_of_solve_arguments_match_upstream(algo_name, path, upstream_name):
    signature = inspect.signature(UPSTREAM[algo_name].solve)
    upstream_default = signature.parameters[upstream_name].default
    wrapper_default = _default(algo_name, path)

    deviations = DELIBERATE_DEFAULT_DEVIATIONS[algo_name]
    if upstream_name in deviations:
        assert upstream_default == deviations[upstream_name]
        assert wrapper_default != upstream_default
    else:
        assert wrapper_default == upstream_default


@pytest.mark.parametrize("algo_name", ALGORITHMS)
def test_flags_of_switchable_groups_are_decided_by_the_solver(algo_name):
    """The contexts in which the solver restarts are the documented ones."""
    use_restarts = [
        params["restarts.use_restarts"]
        for params in _upstream_parameter_lists(algo_name)
    ]
    noisy_only = [
        params["restarts.use_restarts"]
        for params in _upstream_parameter_lists(algo_name, only_noisy=True)
    ]

    assert _default(algo_name, "restarts") is None
    assert set(use_restarts) == {True, False}
    assert all(noisy_only)
    assert _default(algo_name, "noisy") is False


def test_stopping_maxfun_uses_the_harmonized_default():
    assert NagDFOLS().stopping_maxfun == STOPPING_MAXFUN
    assert NagPyBOBYQA().stopping_maxfun == STOPPING_MAXFUN


@pytest.mark.parametrize(
    ("algo_name", "path", "upstream_name"),
    [
        (algo_name, path, upstream_name)
        for algo_name in ALGORITHMS
        for path, upstream_name in ADVANCED_OPTIONS[algo_name].items()
    ],
)
def test_defaults_of_advanced_options_match_upstream(algo_name, path, upstream_name):
    """Constant upstream defaults are copied, all others are None in the wrapper.

    The entries of ``noisy`` are only passed for noisy functions, so their defaults
    are compared with the upstream defaults for noisy functions.

    """
    wrapper_default = _default(algo_name, path)
    only_noisy = path.startswith("noisy.")
    upstream_defaults = [
        params[upstream_name]
        for params in _upstream_parameter_lists(algo_name, only_noisy=only_noisy)
    ]
    is_constant = all(default == upstream_defaults[0] for default in upstream_defaults)

    deviations = DELIBERATE_DEFAULT_DEVIATIONS[algo_name]
    if upstream_name in deviations:
        assert is_constant
        assert upstream_defaults[0] == deviations[upstream_name]
        assert wrapper_default != upstream_defaults[0]
    elif upstream_name in DEFAULTS_SET_BY_SOLVER_AT_RUNTIME[algo_name]:
        assert wrapper_default is None
    elif is_constant:
        assert wrapper_default == upstream_defaults[0]
        assert type(wrapper_default) is type(upstream_defaults[0])
    else:
        assert wrapper_default is None


@pytest.mark.parametrize("algo_name", ALGORITHMS)
def test_unsupported_arguments_have_the_values_the_wrapper_relies_on(algo_name):
    signature = inspect.signature(UPSTREAM[algo_name].solve)
    assert signature.parameters["scaling_within_bounds"].default is False
    assert signature.parameters["projections"].default in (None, [])
    for params in _upstream_parameter_lists(algo_name):
        assert params["init.run_in_parallel"] is False


@pytest.mark.parametrize("algo_name", ALGORITHMS)
def test_diagnostic_options_set_by_the_wrapper_exist_upstream(algo_name):
    # The wrapper sets them to count iterations, so users cannot set them.
    assert set(nag._DIAGNOSTIC_OPTIONS) <= set(UNSUPPORTED_ADVANCED_OPTIONS[algo_name])
    for params in _upstream_parameter_lists(algo_name):
        assert set(nag._DIAGNOSTIC_OPTIONS) <= set(params)
        # Not saving the diagnostic information is the default of the solvers. The
        # result then has no diagnostic information at all.
        assert params["logging.save_diagnostic_info"] is False


# ======================================================================================
# Forwarding of options
# ======================================================================================

_COMMON_TEST_VALUES = {
    "stopping_maxfun": 15,
    "convergence_min_trustregion_radius": 1e-4,
    "trustregion_initial_radius": 0.2,
    "n_interpolation_points": 6,
    "noisy": True,
    "rounding_error_constant": 0.2,
    "safety_step_threshold": 0.4,
    "init_options.random_initial_directions": True,
    "init_options.random_directions_make_orthogonal": False,
    "trustregion_options.shrinking_threshold": 0.2,
    "trustregion_options.expansion_threshold": 0.8,
    "trustregion_options.shrinking_factor": 0.6,
    "trustregion_options.expansion_factor": 3.0,
    "trustregion_options.step_expansion_factor": 5.0,
    "trustregion_options.min_radius_shrinking_factor": 0.2,
    "trustregion_options.radius_factor_after_min_radius_shrinking": 0.6,
    "slow_progress_options.history_for_slow": 6,
    "slow_progress_options.threshold_for_slow": 1e-6,
    "slow_progress_options.max_slow_iterations": 7,
    "noisy.quit_on_noise_level": False,
    "noisy.scale_factor_for_quit": 2.0,
    "noisy.multiplicative_noise_level": 0.1,
    "noisy.additive_noise_level": 0.1,
    "interpolation_options.precondition": False,
    "restarts": True,
    "restarts.max_unsuccessful_restarts": 2,
    "restarts.convergence_min_radius_scale": 0.5,
    "restarts.use_soft_restarts": False,
    "restarts.soft_n_geometry_steps": 2,
    "restarts.soft_move_iterate": False,
    "restarts.soft_max_fake_successful_steps": 9,
    "restarts.hard_use_old_fun_value": False,
    "restarts.auto_detect": False,
    "restarts.auto_detect_history": 20,
    "restarts.auto_detect_min_model_change_slope": 0.02,
    "restarts.auto_detect_min_correlation": 0.2,
}

TEST_VALUES = {
    "nag_dfols": {
        **_COMMON_TEST_VALUES,
        "convergence_target_value": 1e-6,
        "convergence_ftol_scaled": 1e-3,
        "check_fun_for_overflow": False,
        "interpolation_options.throw_error_on_nans": True,
        "regression_options.n_extra_steps": 1,
        "regression_options.increase_n_extra_steps_with_restart": 1,
        "regression_options.momentum_extra_steps": True,
        "restarts.increase_n_interpolation_points": True,
        "restarts.increase_n_interpolation_points_amount": 2,
        "restarts.hard_increase_n_initial_directions_amount": 2,
        "restarts.max_n_interpolation_points": 6,
        "restarts.throw_error_on_nans": True,
        "growing_options.n_initial_directions": 2,
        "growing_options.n_new_directions_each_iteration": 1,
        "growing_options.radius_scale_new_directions": 0.5,
        "growing_options.do_geometry_steps": True,
        "growing_options.reset_radius": True,
        "growing_options.reset_min_radius": True,
        "growing_options.shrinking_factor": 0.7,
        "growing_options.do_safety_step": False,
        "growing_options.safety_reduce_radius": True,
        "growing_options.safety_full_geometry_step": True,
        "growing_options.use_full_rank_interpolation": False,
        "growing_options.perturb_trust_region_step": True,
        "growing_options.full_rank_scale_factor": 0.05,
        "growing_options.full_rank_svd_scale_factor": 0.5,
        "growing_options.full_rank_min_singular_value": 1e-5,
        "growing_options.full_rank_max_jacobian_condition_number": 1e6,
    },
    "nag_pybobyqa": {
        **_COMMON_TEST_VALUES,
        "convergence_target_value": 0.5,
        "seek_global_optimum": True,
        "interpolation_options.minimum_change_hessian": False,
        "restarts.max_unsuccessful_restarts_total": 5,
        "restarts.initial_radius_scale_after_unsuccessful_restart": 1.2,
    },
}

# Options that are not forwarded unchanged and have their own tests below.
CONVERTED_BY_WRAPPER = {"n_evals_per_point"}

# Options that the solvers only accept in combination with other options.
REQUIRED_COMPANIONS = {
    "growing_options.reset_min_radius": {"growing_options.reset_radius": True},
    "growing_options.perturb_trust_region_step": {
        "growing_options.use_full_rank_interpolation": False
    },
}

FORWARDING_CASES = [
    (algo_name, path, value)
    for algo_name in ALGORITHMS
    for path, value in TEST_VALUES[algo_name].items()
]


@pytest.mark.parametrize("algo_name", ALGORITHMS)
def test_there_is_a_forwarding_test_for_every_option(algo_name):
    tested = set(TEST_VALUES[algo_name]) | CONVERTED_BY_WRAPPER
    assert tested == set(_option_paths(algo_name))


@pytest.mark.parametrize(("algo_name", "path", "value"), FORWARDING_CASES)
def test_option_is_forwarded_to_solver(algo_name, path, value, solve_calls):
    assert value != _default(algo_name, path)

    options = {"stopping_maxfun": 12, **REQUIRED_COMPANIONS.get(path, {}), path: value}
    fun = _as_fun(_residuals, algo_name)
    with warnings.catch_warnings():
        # Some values make no sense for such a short run and the solvers say so.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        om.minimize(
            fun,
            START,
            algorithm=ALGORITHMS[algo_name](**_nest(algo_name, options)),
            bounds=om.Bounds(lower=np.full(3, -10.0), upper=np.full(3, 10.0)),
        )

    call = solve_calls[algo_name]
    if path in SOLVE_ARGUMENTS[algo_name]:
        forwarded = call[SOLVE_ARGUMENTS[algo_name][path]]
    else:
        forwarded = call["user_params"][ADVANCED_OPTIONS[algo_name][path]]

    assert forwarded == value
    assert type(forwarded) is type(value)
    if _is_entry_of_switchable_group(algo_name, path):
        # Setting an entry switches the feature on.
        option = path.partition(".")[0]
        flags = {
            "noisy": call["objfun_has_noise"],
            "restarts": call["user_params"].get("restarts.use_restarts"),
        }
        assert flags[option] is True


@pytest.mark.parametrize("algo_name", ALGORITHMS)
def test_options_that_are_none_are_not_passed_to_solver(algo_name, solve_calls):
    _minimize(algo_name)
    call = solve_calls[algo_name]

    def _is_passed_at_defaults(path):
        # The entries of `noisy` and `restarts` are only passed for an option group.
        if _is_entry_of_switchable_group(algo_name, path):
            return False
        return _default(algo_name, path) is not None

    for path, upstream_name in SOLVE_ARGUMENTS[algo_name].items():
        assert (upstream_name in call) is _is_passed_at_defaults(path)
    for path, upstream_name in ADVANCED_OPTIONS[algo_name].items():
        assert (upstream_name in call["user_params"]) is _is_passed_at_defaults(path)


def _forwarded_group(call, prefix):
    """The advanced options of one upstream group that the solver was called with."""
    return {
        key: value
        for key, value in call["user_params"].items()
        if key.startswith(prefix)
    }


@pytest.mark.parametrize("algo_name", ALGORITHMS)
def test_restarts_as_none_or_boolean_only_forwards_the_flag(algo_name, solve_calls):
    expected: dict[Any, dict[str, bool]] = {
        None: {},
        False: {"restarts.use_restarts": False},
        True: {"restarts.use_restarts": True},
    }
    for restarts, forwarded in expected.items():
        _minimize(algo_name, restarts=restarts, stopping_maxfun=12)
        assert _forwarded_group(solve_calls[algo_name], "restarts.") == forwarded


@pytest.mark.parametrize("algo_name", ALGORITHMS)
def test_restarts_as_option_group_forwards_flag_and_entries(algo_name, solve_calls):
    group = _group(algo_name, "restarts", max_unsuccessful_restarts=2)
    _minimize(algo_name, restarts=group, stopping_maxfun=12)
    forwarded = _forwarded_group(solve_calls[algo_name], "restarts.")

    expected = {"restarts.use_restarts": True}
    for path, upstream_name in ADVANCED_OPTIONS[algo_name].items():
        option, _, entry = path.partition(".")
        if option == "restarts" and entry and getattr(group, entry) is not None:
            expected[upstream_name] = getattr(group, entry)

    assert forwarded == expected
    assert forwarded["restarts.max_unsuccessful_restarts"] == 2
    assert len(forwarded) > 10


@pytest.mark.parametrize("algo_name", ALGORITHMS)
def test_noisy_as_boolean_only_forwards_the_flag(algo_name, solve_calls):
    for noisy in (False, True):
        _minimize(algo_name, noisy=noisy, stopping_maxfun=12)
        call = solve_calls[algo_name]

        assert call["objfun_has_noise"] is noisy
        assert _forwarded_group(call, "noise.") == {}


@pytest.mark.parametrize("algo_name", ALGORITHMS)
def test_noisy_as_option_group_forwards_flag_and_entries(algo_name, solve_calls):
    group = nag.NagNoiseOptions(additive_noise_level=0.1)
    _minimize(algo_name, noisy=group, stopping_maxfun=12)
    call = solve_calls[algo_name]

    assert call["objfun_has_noise"] is True
    assert _forwarded_group(call, "noise.") == {
        "noise.quit_on_noise_level": True,
        "noise.scale_factor_for_quit": 1.0,
        "noise.additive_noise_level": 0.1,
    }


@pytest.mark.parametrize("algo_name", ALGORITHMS)
def test_wrapper_switches_off_upstream_logging_and_scaling(algo_name, solve_calls):
    _minimize(algo_name)
    call = solve_calls[algo_name]

    assert call["do_logging"] is False
    assert call["print_progress"] is False
    assert call["scaling_within_bounds"] is False


@pytest.mark.parametrize("algo_name", ALGORITHMS)
def test_wrapper_only_requests_the_diagnostics_it_needs(algo_name, solve_calls):
    _minimize(algo_name)
    user_params = solve_calls[algo_name]["user_params"]
    logging_options = {
        key: value for key, value in user_params.items() if key.startswith("logging.")
    }

    # One row per iteration is needed to count iterations. The poisedness is expensive
    # to compute and not needed.
    assert logging_options == {
        "logging.save_diagnostic_info": True,
        "logging.save_poisedness": False,
    }


@pytest.mark.parametrize("algo_name", ALGORITHMS)
def test_integers_are_converted_for_options_that_need_floats(algo_name, solve_calls):
    # The solvers reject integers where they expect floats.
    res = _minimize(
        algo_name,
        trustregion_options=_group(
            algo_name, "trustregion_options", expansion_factor=3
        ),
        restarts=_group(algo_name, "restarts", convergence_min_radius_scale=1),
    )
    user_params = solve_calls[algo_name]["user_params"]

    assert res.success
    assert isinstance(user_params["tr_radius.gamma_inc"], float)
    assert isinstance(user_params["restarts.rhoend_scale"], float)


# ======================================================================================
# Results
# ======================================================================================


@pytest.mark.parametrize("algo_name", ALGORITHMS)
def test_result_at_defaults(algo_name):
    res = _minimize(algo_name)

    np.testing.assert_allclose(res.params, OPTIMUM, atol=1e-6)
    np.testing.assert_allclose(res.fun, 1.0)
    assert res.success
    assert "rho has reached rhoend" in res.message
    assert res.n_fun_evals == len(res.history.fun)
    assert isinstance(res.n_iterations, int)
    assert res.n_iterations > 0
    assert res.algorithm_output == {
        "n_runs": 1,
        "n_evaluated_points": res.n_fun_evals,
    }
    assert res.hess is None
    assert res.hess_inv is None


def test_result_of_dfols_contains_the_estimated_jacobian_of_the_residuals():
    res = _minimize("nag_dfols")
    # `_residuals` has one residual per parameter and a constant one.
    expected = np.vstack([np.eye(3), np.zeros(3)])

    np.testing.assert_allclose(res.jac, expected, atol=1e-6)


def test_result_of_pybobyqa_contains_the_estimated_gradient():
    res: Any = om.minimize(
        _as_fun(_residuals, "nag_pybobyqa"),
        START,
        algorithm=NagPyBOBYQA(),
        bounds=om.Bounds(upper=np.array([0.5, np.inf, np.inf])),
    )
    # The first parameter ends at its bound, where the gradient is not zero.
    expected = np.array([2 * (0.5 - OPTIMUM[0]), 0.0, 0.0])

    np.testing.assert_allclose(res.params, [0.5, 2.0, 3.0], atol=1e-6)
    np.testing.assert_allclose(res.jac, expected, atol=1e-4)


@pytest.mark.parametrize("algo_name", ALGORITHMS)
def test_n_iterations_is_the_total_over_all_runs(algo_name, raw_results):
    res = _minimize(
        algo_name, restarts=_group(algo_name, "restarts", max_unsuccessful_restarts=2)
    )
    info = raw_results[algo_name].diagnostic_info
    rows_per_run = info.groupby("nruns").size()
    last_iteration_per_run = info.groupby("nruns")["iter_this_run"].max()

    assert res.algorithm_output["n_runs"] == len(rows_per_run) == 3
    # The solvers number the iterations of each run from zero.
    assert (rows_per_run == last_iteration_per_run + 1).all()
    assert res.n_iterations == rows_per_run.sum()
    assert res.n_iterations > rows_per_run.max()


@pytest.mark.parametrize("algo_name", ALGORITHMS)
@pytest.mark.filterwarnings("ignore:maxfun <= npt:RuntimeWarning")
def test_n_iterations_is_zero_if_solver_stops_before_the_first_iteration(algo_name):
    res = _minimize(algo_name, stopping_maxfun=2)

    assert not res.success
    assert res.n_iterations == 0
    # The solver did not build a model from which it could estimate derivatives.
    assert res.jac is None


def test_n_iterations_is_none_without_diagnostic_information():
    assert nag._count_iterations(SimpleNamespace(diagnostic_info=None)) is None


@pytest.mark.parametrize("algo_name", ALGORITHMS)
def test_options_rejected_by_solver_raise(algo_name):
    with pytest.raises(InvalidAlgoOptionError, match="npt must be >= n"):
        _minimize(algo_name, n_interpolation_points=1)


def test_contradicting_growing_options_raise():
    with pytest.raises(InvalidAlgoOptionError, match="Safety step while growing"):
        _minimize(
            "nag_dfols",
            growing_options=nag.DfolsGrowingOptions(
                safety_reduce_radius=True, safety_full_geometry_step=True
            ),
        )


@pytest.mark.parametrize("algo_name", ALGORITHMS)
def test_invalid_entry_of_option_group_raises(algo_name):
    with pytest.raises(InvalidAlgoOptionError, match="RestartOptions"):
        _group(algo_name, "restarts", use_resets=True)
    with pytest.raises(InvalidAlgoOptionError, match="RestartOptions"):
        # The flag is the option itself and not an entry of the group anymore.
        _group(algo_name, "restarts", use_restarts=True)
    with pytest.raises(InvalidAlgoOptionError, match="NagTrustregionOptions"):
        nag.NagTrustregionOptions(shrinking_threshold=1.5)


@pytest.mark.parametrize("algo_name", ALGORITHMS)
def test_option_group_as_dictionary_is_rejected(algo_name):
    entries = {"expansion_factor": 3.0}
    match = "instance of NagTrustregionOptions"

    with pytest.raises(InvalidAlgoOptionError, match=match):
        ALGORITHMS[algo_name](trustregion_options=entries)
    with pytest.raises(InvalidAlgoOptionError, match=match):
        ALGORITHMS[algo_name]().with_option(trustregion_options=entries)
    with pytest.raises(InvalidAlgoOptionError, match=match):
        om.minimize(
            _as_fun(_residuals, algo_name),
            START,
            algorithm=algo_name,
            algo_options={"trustregion_options": entries},
        )


@pytest.mark.parametrize("algo_name", ALGORITHMS)
@pytest.mark.parametrize("option", ["restarts", "noisy"])
def test_dictionary_is_rejected_for_boolean_or_option_group(algo_name, option):
    group_cls = SWITCHABLE_GROUPS[algo_name][option]
    entries = {
        field.name: getattr(group_cls(), field.name) for field in fields(group_cls)
    }
    match = f"instance of {group_cls.__name__}"

    with pytest.raises(InvalidAlgoOptionError, match=match):
        ALGORITHMS[algo_name](**{option: entries})
    with pytest.raises(InvalidAlgoOptionError, match=match):
        ALGORITHMS[algo_name]().with_option(**{option: entries})
    with pytest.raises(InvalidAlgoOptionError, match=match):
        om.minimize(
            _as_fun(_residuals, algo_name),
            START,
            algorithm=algo_name,
            algo_options={option: entries},
        )


@pytest.mark.parametrize("algo_name", ALGORITHMS)
def test_restarts_and_noisy_work_in_algo_options(algo_name, solve_calls):
    om.minimize(
        _as_fun(_residuals, algo_name),
        START,
        algorithm=algo_name,
        algo_options={
            "noisy": True,
            "restarts": _group(algo_name, "restarts", max_unsuccessful_restarts=1),
            "stopping_maxfun": 30,
        },
    )
    call = solve_calls[algo_name]

    assert call["objfun_has_noise"] is True
    assert call["user_params"]["restarts.use_restarts"] is True
    assert call["user_params"]["restarts.max_unsuccessful_restarts"] == 1


@pytest.mark.parametrize("algo_name", ALGORITHMS)
def test_option_group_instance_works_in_algo_options(algo_name, solve_calls):
    om.minimize(
        _as_fun(_residuals, algo_name),
        START,
        algorithm=algo_name,
        algo_options={
            "trustregion_options": nag.NagTrustregionOptions(expansion_factor=3.0)
        },
    )

    assert solve_calls[algo_name]["user_params"]["tr_radius.gamma_inc"] == 3.0


@pytest.mark.parametrize("algo_name", ALGORITHMS)
def test_stopping_maxfun_works_in_algo_options(algo_name, solve_calls):
    res = om.minimize(
        _as_fun(_residuals, algo_name),
        START,
        algorithm=algo_name,
        algo_options={"stopping_maxfun": 8},
    )

    assert solve_calls[algo_name]["maxfun"] == 8
    assert res.n_fun_evals == 8


# ======================================================================================
# Behavior: stopping and convergence
# ======================================================================================


@pytest.mark.parametrize("algo_name", ALGORITHMS)
def test_stopping_maxfun(algo_name):
    res = _minimize(algo_name, stopping_maxfun=8)

    assert res.n_fun_evals == 8
    assert not res.success
    assert "MAXFUN" in res.message


@pytest.mark.parametrize("algo_name", ALGORITHMS)
def test_convergence_min_trustregion_radius(algo_name):
    precise = _minimize(algo_name)
    rough = _minimize(algo_name, convergence_min_trustregion_radius=1e-3)

    assert rough.success
    assert "rho has reached rhoend" in rough.message
    assert rough.n_fun_evals < precise.n_fun_evals


@pytest.mark.parametrize("algo_name", ALGORITHMS)
def test_convergence_target_value(algo_name):
    default = _minimize(algo_name)
    res = _minimize(algo_name, convergence_target_value=1.5)

    assert res.success
    assert "Objective is sufficiently small" in res.message
    assert res.fun <= 1.5
    assert res.n_fun_evals < default.n_fun_evals


def test_convergence_target_value_of_dfols_stops_at_zero_residuals_by_default():
    res = _minimize("nag_dfols", _rosenbrock_residuals, ROSENBROCK_START)

    assert res.success
    assert "Objective is sufficiently small" in res.message
    assert res.fun <= 1e-12


def test_convergence_ftol_scaled():
    start_value = float(np.sum(_rosenbrock_residuals(ROSENBROCK_START) ** 2))
    res = _minimize(
        "nag_dfols",
        _rosenbrock_residuals,
        ROSENBROCK_START,
        convergence_ftol_scaled=1e-2,
    )

    assert res.success
    assert "Objective is sufficiently small" in res.message
    assert 1e-6 * start_value < res.fun <= 1e-2 * start_value


@pytest.mark.parametrize("algo_name", ALGORITHMS)
def test_slow_progress_stops_optimization(algo_name):
    kwargs = {"residuals": _rosenbrock_residuals, "x0": ROSENBROCK_START}
    default = _minimize(algo_name, **kwargs)
    impatient = _minimize(
        algo_name,
        slow_progress_options=nag.NagSlowProgressOptions(
            threshold_for_slow=10.0, max_slow_iterations=3
        ),
        **kwargs,
    )
    patient = _minimize(
        algo_name,
        slow_progress_options=nag.NagSlowProgressOptions(
            threshold_for_slow=10.0, max_slow_iterations=8
        ),
        **kwargs,
    )

    assert default.success
    assert "slow" not in default.message
    for res in (impatient, patient):
        assert not res.success
        assert "Maximum slow iterations reached" in res.message
    assert impatient.n_fun_evals < patient.n_fun_evals < default.n_fun_evals


# ======================================================================================
# Behavior: trust region, interpolation points and initialization
# ======================================================================================


@pytest.mark.parametrize("algo_name", ALGORITHMS)
def test_trustregion_initial_radius(algo_name):
    default = _minimize(algo_name)
    res = _minimize(algo_name, trustregion_initial_radius=0.3)

    # the default radius is 0.1 * max(max(abs(x0)), 1)
    np.testing.assert_allclose(default.history.params[1] - START, [0.1, 0, 0])
    np.testing.assert_allclose(res.history.params[1] - START, [0.3, 0, 0])


@pytest.mark.parametrize(
    ("algo_name", "n_default"), [("nag_dfols", 4), ("nag_pybobyqa", 7)]
)
def test_n_interpolation_points(algo_name, n_default):
    default = _minimize(algo_name)
    res = _minimize(algo_name, n_interpolation_points=6)

    # All points except the start parameters are first placed along the axes.
    assert _n_leading_axis_points(default, START, radius=0.1) == n_default - 1
    assert _n_leading_axis_points(res, START, radius=0.1) == 5


@pytest.mark.parametrize("algo_name", ALGORITHMS)
def test_random_initial_directions(algo_name):
    np.random.seed(0)
    orthogonal = _minimize(
        algo_name, init_options=nag.NagInitOptions(random_initial_directions=True)
    )
    np.random.seed(0)
    not_orthogonal = _minimize(
        algo_name,
        init_options=nag.NagInitOptions(
            random_initial_directions=True, random_directions_make_orthogonal=False
        ),
    )

    assert _n_leading_axis_points(orthogonal, START, radius=0.1) == 0
    for res, expect_orthogonal in [(orthogonal, True), (not_orthogonal, False)]:
        directions = np.array(res.history.params[1:4]) - START
        gram = directions @ directions.T
        np.testing.assert_allclose(np.diag(gram), 0.1**2)
        off_diagonal = gram[~np.eye(3, dtype=bool)]
        assert np.allclose(off_diagonal, 0, atol=1e-12) is expect_orthogonal


def test_growing_n_initial_directions():
    res = _minimize(
        "nag_dfols", growing_options=nag.DfolsGrowingOptions(n_initial_directions=1)
    )

    assert _n_leading_axis_points(res, START, radius=0.1) == 1
    assert res.success
    np.testing.assert_allclose(res.params, OPTIMUM, atol=1e-6)


def test_interpolation_throw_error_on_nans():
    def _residuals_with_nans(x):
        out = _rosenbrock_residuals(x)
        return out * np.nan if x[0] > -1.15 else out

    graceful = _minimize("nag_dfols", _residuals_with_nans, ROSENBROCK_START)
    assert not graceful.success
    assert "linear algebra" in graceful.message

    with pytest.raises(np.linalg.LinAlgError, match="NaN encountered"):
        _minimize(
            "nag_dfols",
            _residuals_with_nans,
            ROSENBROCK_START,
            interpolation_options=nag.DfolsInterpolationOptions(
                throw_error_on_nans=True
            ),
        )


# ======================================================================================
# Behavior: noise
# ======================================================================================


@pytest.mark.parametrize("algo_name", ALGORITHMS)
def test_n_evals_per_point_as_number(algo_name):
    res = _minimize(algo_name, n_evals_per_point=3)

    assert res.success
    assert res.n_fun_evals == 3 * res.algorithm_output["n_evaluated_points"]
    assert res.n_fun_evals == len(res.history.fun)


@pytest.mark.parametrize("algo_name", ALGORITHMS)
def test_n_evals_per_point_as_function(algo_name):
    calls = []

    def n_evals(radius, min_radius, n_iterations, n_restarts):
        calls.append((radius, min_radius, n_iterations, n_restarts))
        return 2 if radius < 0.05 else 1

    res = _minimize(algo_name, n_evals_per_point=n_evals)
    n_points = res.algorithm_output["n_evaluated_points"]

    assert res.success
    assert n_points < res.n_fun_evals < 2 * n_points
    assert calls[0] == (0.1, 0.1, 0, 0)
    assert min(radius for radius, *_ in calls) < 0.05


@pytest.mark.parametrize("algo_name", ALGORITHMS)
def test_noisy_function_is_restarted(algo_name, solve_calls):
    kwargs = {
        "residuals": _rosenbrock_residuals,
        "x0": ROSENBROCK_START,
        "noise_scale": 0.01,
        "stopping_maxfun": 150,
    }
    not_noisy = _minimize(algo_name, noisy=False, **kwargs)
    assert solve_calls[algo_name]["objfun_has_noise"] is False
    noisy = _minimize(algo_name, noisy=True, **kwargs)
    assert solve_calls[algo_name]["objfun_has_noise"] is True
    # Describing the noise also declares the function as noisy.
    described = _minimize(algo_name, noisy=nag.NagNoiseOptions(), **kwargs)
    assert solve_calls[algo_name]["objfun_has_noise"] is True
    without_restarts = _minimize(algo_name, noisy=True, restarts=False, **kwargs)

    assert not_noisy.algorithm_output["n_runs"] == 1
    assert noisy.algorithm_output["n_runs"] > 1
    assert described.algorithm_output["n_runs"] > 1
    assert without_restarts.algorithm_output["n_runs"] == 1


@pytest.mark.parametrize("algo_name", ALGORITHMS)
@pytest.mark.parametrize(
    "level", ["additive_noise_level", "multiplicative_noise_level"]
)
def test_quit_on_noise_level(algo_name, level):
    kwargs = {
        "residuals": _rosenbrock_residuals,
        "x0": ROSENBROCK_START,
        "noise_scale": 0.01,
        "stopping_maxfun": 300,
        "restarts": False,
    }

    def _noise(**entries):
        return _group(algo_name, "noisy", **entries)

    unknown_level = _minimize(algo_name, noisy=True, **kwargs)
    large_level = _minimize(algo_name, noisy=_noise(**{level: 5.0}), **kwargs)
    tiny_level = _minimize(algo_name, noisy=_noise(**{level: 1e-12}), **kwargs)
    switched_off = _minimize(
        algo_name,
        noisy=_noise(**{level: 5.0}, quit_on_noise_level=False),
        **kwargs,
    )

    assert large_level.success
    assert "All points within noise level" in large_level.message
    for res in (unknown_level, tiny_level, switched_off):
        assert "noise level" not in res.message
        assert res.n_fun_evals > large_level.n_fun_evals
        assert res.fun < large_level.fun


@pytest.mark.parametrize("algo_name", ALGORITHMS)
def test_only_one_noise_level_can_be_set(algo_name):
    with pytest.raises(InvalidAlgoOptionError, match="exactly one of additive"):
        _minimize(
            algo_name,
            noisy=nag.NagNoiseOptions(
                additive_noise_level=0.1, multiplicative_noise_level=0.1
            ),
        )


# ======================================================================================
# Behavior: restarts
# ======================================================================================


@pytest.mark.parametrize("algo_name", ALGORITHMS)
def test_restarts(algo_name):
    default = _minimize(algo_name)
    switched_off = _minimize(algo_name, restarts=False)
    switched_on = _minimize(algo_name, restarts=True)
    with_settings = _minimize(
        algo_name, restarts=_group(algo_name, "restarts", max_unsuccessful_restarts=2)
    )

    # Without noise the solver does not restart by default.
    for res in (default, switched_off):
        assert res.algorithm_output["n_runs"] == 1
        assert "rho has reached rhoend" in res.message
    assert switched_off.n_fun_evals == default.n_fun_evals

    # The solver stops after 10 restarts in a row that did not improve the result,
    # or after 2 with the settings. A restart can improve the result marginally,
    # which starts the count again.
    assert switched_on.algorithm_output["n_runs"] >= 11
    assert with_settings.algorithm_output["n_runs"] == 3
    for res in (switched_on, with_settings):
        assert res.success
        assert "unsuccessful restarts" in res.message
    assert switched_on.n_fun_evals > with_settings.n_fun_evals > default.n_fun_evals


@pytest.mark.parametrize("algo_name", ALGORITHMS)
def test_use_soft_restarts(algo_name):
    radius = 0.25

    def _n_axis_points_around_optimum(res):
        diff = np.array(res.history.params) - OPTIMUM
        is_axis_point = np.sum(~np.isclose(diff, 0, atol=1e-6), axis=1) == 1
        has_radius = np.isclose(np.abs(diff).max(axis=1), radius, atol=1e-6)
        return int(np.sum(is_axis_point & has_radius))

    results = {}
    for soft in (True, False):
        results[soft] = _minimize(
            algo_name,
            trustregion_initial_radius=radius,
            restarts=_group(
                algo_name,
                "restarts",
                use_soft_restarts=soft,
                max_unsuccessful_restarts=2,
            ),
        )

    # A hard restart builds a new set of points along the axes around the best
    # point. A soft restart only moves a few of the existing points.
    assert _n_axis_points_around_optimum(results[True]) == 0
    assert _n_axis_points_around_optimum(results[False]) >= 2 * 3


def test_seek_global_optimum_restarts_and_needs_bounds():
    def multimodal(x):
        return float(np.sum(x**2) + 3 * np.sum(1 - np.cos(3 * x)))

    algo_cls = ALGORITHMS["nag_pybobyqa"]
    kwargs: dict[str, Any] = {
        "fun": multimodal,
        "params": np.array([2.2, 2.2]),
        "bounds": om.Bounds(lower=np.full(2, -5.0), upper=np.full(2, 5.0)),
    }
    local: Any = om.minimize(algorithm=algo_cls(), **kwargs)
    searching: Any = om.minimize(
        algorithm=algo_cls(seek_global_optimum=True, stopping_maxfun=400), **kwargs
    )

    assert local.algorithm_output["n_runs"] == 1
    assert searching.algorithm_output["n_runs"] > 1
    assert searching.fun <= local.fun

    with pytest.raises(InvalidAlgoOptionError, match="must specify upper and lower"):
        om.minimize(
            multimodal,
            np.array([2.2, 2.2]),
            algorithm=algo_cls(seek_global_optimum=True),
        )


# ======================================================================================
# Deprecated options
# ======================================================================================

_OLD_SLOW_PROGRESS = {
    "threshold_to_characterize_as_slow": 1e-6,
    "max_insufficient_improvements": 7,
    "comparison_period": 6,
}
_NEW_SLOW_PROGRESS = {
    "threshold_for_slow": 1e-6,
    "max_slow_iterations": 7,
    "history_for_slow": 6,
}

_OLD_RESET_OPTIONS = {
    "use_resets": True,
    "minimal_trustregion_radius_tolerance_scaling_at_reset": 0.5,
    "reset_type": "hard",
    "move_center_at_soft_reset": False,
    "reuse_criterion_value_at_hard_reset": False,
    "max_iterations_without_new_best_after_soft_reset": 9,
    "auto_detect": False,
    "auto_detect_history": 20,
    "auto_detect_min_jacobian_increase": 0.02,
    "auto_detect_min_correlations": 0.2,
    "points_to_replace_at_soft_reset": 2,
    "max_consecutive_unsuccessful_resets": 4,
}
_NEW_RESTART_OPTIONS = {
    "convergence_min_radius_scale": 0.5,
    "use_soft_restarts": False,
    "soft_move_iterate": False,
    "hard_use_old_fun_value": False,
    "soft_max_fake_successful_steps": 9,
    "auto_detect": False,
    "auto_detect_history": 20,
    "auto_detect_min_model_change_slope": 0.02,
    "auto_detect_min_correlation": 0.2,
    "soft_n_geometry_steps": 2,
    "max_unsuccessful_restarts": 4,
}

_OLD_FAST_START_OPTIONS = {
    "min_inital_points": 2,
    "method": "trustregion",
    "scale_of_trustregion_step_perturbation": 0.5,
    "scale_of_jacobian_components_perturbation": 0.05,
    "floor_of_jacobian_singular_values": 0.5,
    "jacobian_max_condition_number": 1e6,
    "geometry_improving_steps": True,
    "safety_steps": False,
    "shrink_upper_radius_in_safety_steps": True,
    "full_geometry_improving_step": True,
    "reset_trustregion_radius_after_fast_start": True,
    "reset_min_trustregion_radius_after_fast_start": True,
    "shrinking_factor_not_successful": 0.7,
    "n_extra_search_directions_per_iteration": 1,
}
_NEW_GROWING_OPTIONS = {
    "n_initial_directions": 2,
    "use_full_rank_interpolation": False,
    "perturb_trust_region_step": True,
    "radius_scale_new_directions": 0.5,
    "full_rank_scale_factor": 0.05,
    "full_rank_svd_scale_factor": 0.5,
    "full_rank_max_jacobian_condition_number": 1e6,
    "do_geometry_steps": True,
    "do_safety_step": False,
    "safety_reduce_radius": True,
    "safety_full_geometry_step": True,
    "reset_radius": True,
    "reset_min_radius": True,
    "shrinking_factor": 0.7,
    "n_new_directions_each_iteration": 1,
}

_OLD_MINIMUM_CHANGE_HESSIAN = (
    "trustregion_minimum_change_hession_for_underdetermined_interpolation"
)

# Each case is a tuple of old keyword arguments and their new spelling. In the new
# spelling, option groups are given as dictionaries of their entries and converted
# to the group classes of the algorithm with `_with_groups`.
_COMMON_DEPRECATION_CASES = [
    (
        {"convergence_minimal_trustregion_radius_tolerance": 1e-5},
        {"convergence_min_trustregion_radius": 1e-5},
    ),
    ({"interpolation_rounding_error": 0.2}, {"rounding_error_constant": 0.2}),
    ({"threshold_for_safety_step": 0.4}, {"safety_step_threshold": 0.4}),
    ({"trustregion_n_interpolation_points": 6}, {"n_interpolation_points": 6}),
    ({"noise_n_evals_per_point": 2}, {"n_evals_per_point": 2}),
    (
        {"initial_directions": "random"},
        {"init_options": {"random_initial_directions": True}},
    ),
    (
        {"initial_directions": "coordinate"},
        {"init_options": {"random_initial_directions": False}},
    ),
    (
        {"random_directions_orthogonal": False},
        {"init_options": {"random_directions_make_orthogonal": False}},
    ),
    (
        {"trustregion_threshold_successful": 0.2},
        {"trustregion_options": {"shrinking_threshold": 0.2}},
    ),
    (
        {"trustregion_threshold_very_successful": 0.8},
        {"trustregion_options": {"expansion_threshold": 0.8}},
    ),
    (
        {"trustregion_shrinking_factor_not_successful": 0.6},
        {"trustregion_options": {"shrinking_factor": 0.6}},
    ),
    (
        {"trustregion_expansion_factor_successful": 3.0},
        {"trustregion_options": {"expansion_factor": 3.0}},
    ),
    (
        {"trustregion_expansion_factor_very_successful": 5.0},
        {"trustregion_options": {"step_expansion_factor": 5.0}},
    ),
    (
        {"trustregion_shrinking_factor_lower_radius": 0.2},
        {"trustregion_options": {"min_radius_shrinking_factor": 0.2}},
    ),
    (
        {"trustregion_shrinking_factor_upper_radius": 0.6},
        {"trustregion_options": {"radius_factor_after_min_radius_shrinking": 0.6}},
    ),
    (
        {"trustregion_precondition_interpolation": False},
        {"interpolation_options": {"precondition": False}},
    ),
    (
        {"convergence_slow_progress": _OLD_SLOW_PROGRESS},
        {"slow_progress_options": _NEW_SLOW_PROGRESS},
    ),
    (
        {"convergence_slow_progress": {"comparison_period": 6}},
        {"slow_progress_options": {"history_for_slow": 6}},
    ),
    # A noise level used to be the only way to declare the function as noisy.
    ({"noise_additive_level": 0.1}, {"noisy": {"additive_noise_level": 0.1}}),
    (
        {"noise_multiplicative_level": 0.1},
        {"noisy": {"multiplicative_noise_level": 0.1}},
    ),
    (
        {"noise_additive_level": 0.1, "noise_multiplicative_level": 0.2},
        {"noisy": {"additive_noise_level": 0.1, "multiplicative_noise_level": 0.2}},
    ),
    # The tolerance only had an effect together with a noise level.
    (
        {
            "convergence_noise_corrected_criterion_tolerance": 2.0,
            "noise_additive_level": 0.1,
        },
        {"noisy": {"additive_noise_level": 0.1, "scale_factor_for_quit": 2.0}},
    ),
    (
        {
            "convergence_noise_corrected_criterion_tolerance": 0.0,
            "noise_multiplicative_level": 0.1,
        },
        {"noisy": {"multiplicative_noise_level": 0.1, "quit_on_noise_level": False}},
    ),
    # `use_resets` used to switch restarts on and off.
    ({"trustregion_reset_options": {"use_resets": True}}, {"restarts": True}),
    ({"trustregion_reset_options": {"use_resets": False}}, {"restarts": False}),
    (
        {"trustregion_reset_options": {"use_resets": False, "reset_type": "hard"}},
        {"restarts": False},
    ),
    (
        {"trustregion_reset_options": {"use_resets": True, "reset_type": "soft"}},
        {"restarts": {"use_soft_restarts": True}},
    ),
    # Without `use_resets` the solver decided. It restarts for noisy functions.
    (
        {
            "noise_additive_level": 0.1,
            "trustregion_reset_options": {"reset_type": "hard"},
        },
        {
            "noisy": {"additive_noise_level": 0.1},
            "restarts": {"use_soft_restarts": False},
        },
    ),
]

# Old keyword arguments that are accepted with a warning but have no effect.
_COMMON_CASES_WITHOUT_EFFECT = [
    # A noise level of None or zero did not declare the function as noisy.
    {"noise_additive_level": None},
    {"noise_multiplicative_level": 0.0},
    # The tolerance only had an effect together with a noise level.
    {"convergence_noise_corrected_criterion_tolerance": 2.0},
    # Without `use_resets` the solver decided, and it only restarts for noisy
    # functions. Without restarts the other entries had no effect.
    {"trustregion_reset_options": {"max_consecutive_unsuccessful_resets": 4}},
    {"trustregion_reset_options": {"use_resets": None, "reset_type": "hard"}},
    {"trustregion_reset_options": None},
]

DEPRECATION_CASES = {
    "nag_dfols": [
        *_COMMON_DEPRECATION_CASES,
        ({"clip_criterion_if_overflowing": False}, {"check_fun_for_overflow": False}),
        (
            {"trustregion_n_extra_points_to_replace_successful": 2},
            {"regression_options": {"n_extra_steps": 2}},
        ),
        (
            {"trustregion_method_to_replace_extra_points": "momentum"},
            {"regression_options": {"momentum_extra_steps": True}},
        ),
        (
            {"trustregion_method_to_replace_extra_points": "geometry_improving"},
            {"regression_options": {"momentum_extra_steps": False}},
        ),
        (
            {
                "trustregion_reset_options": {
                    **_OLD_RESET_OPTIONS,
                    "max_interpolation_points": 8,
                    "n_extra_interpolation_points_per_soft_reset": 2,
                    "n_extra_interpolation_points_per_hard_reset": 3,
                    "n_additional_extra_points_to_replace_per_reset": 1,
                }
            },
            {
                "restarts": {
                    **_NEW_RESTART_OPTIONS,
                    "max_n_interpolation_points": 8,
                    "increase_n_interpolation_points": True,
                    "increase_n_interpolation_points_amount": 2,
                    "hard_increase_n_initial_directions_amount": 1,
                },
                "regression_options": {"increase_n_extra_steps_with_restart": 1},
            },
        ),
        (
            {
                "trustregion_reset_options": {
                    "use_resets": True,
                    "n_extra_interpolation_points_per_soft_reset": 0,
                }
            },
            {"restarts": {"increase_n_interpolation_points": False}},
        ),
        # An entry that belongs to another option group does not switch restarts on
        # by itself.
        (
            {
                "trustregion_reset_options": {
                    "use_resets": True,
                    "n_additional_extra_points_to_replace_per_reset": 1,
                }
            },
            {
                "restarts": True,
                "regression_options": {"increase_n_extra_steps_with_restart": 1},
            },
        ),
        (
            {"trustregion_fast_start_options": _OLD_FAST_START_OPTIONS},
            {"growing_options": _NEW_GROWING_OPTIONS},
        ),
        (
            {"trustregion_fast_start_options": {"method": "jacobian"}},
            {
                "growing_options": {
                    "use_full_rank_interpolation": True,
                    "perturb_trust_region_step": False,
                }
            },
        ),
    ],
    "nag_pybobyqa": [
        *_COMMON_DEPRECATION_CASES,
        ({"stopping_max_criterion_evaluations": 50}, {"stopping_maxfun": 50}),
        ({"convergence_criterion_value": 0.5}, {"convergence_target_value": 0.5}),
        (
            {_OLD_MINIMUM_CHANGE_HESSIAN: False},
            {"interpolation_options": {"minimum_change_hessian": False}},
        ),
        (
            {
                "trustregion_reset_options": {
                    **_OLD_RESET_OPTIONS,
                    "max_unsuccessful_resets": 5,
                    "trust_region_scaling_at_unsuccessful_reset": 1.2,
                }
            },
            {
                "restarts": {
                    **_NEW_RESTART_OPTIONS,
                    "max_unsuccessful_restarts_total": 5,
                    "initial_radius_scale_after_unsuccessful_restart": 1.2,
                }
            },
        ),
    ],
}

DEPRECATION_PARAMETRIZATION = [
    pytest.param(algo_name, old, new, id=f"{algo_name}-{'-'.join(old)}-{i}")
    for algo_name in ALGORITHMS
    for i, (old, new) in enumerate(DEPRECATION_CASES[algo_name])
]

CASES_WITHOUT_EFFECT = {
    "nag_dfols": [
        *_COMMON_CASES_WITHOUT_EFFECT,
        {
            "trustregion_reset_options": {
                "n_additional_extra_points_to_replace_per_reset": 1
            }
        },
    ],
    "nag_pybobyqa": [
        *_COMMON_CASES_WITHOUT_EFFECT,
        # Py-BOBYQA only catches an OverflowError of the function, which optimagic
        # handles before the solver sees it.
        {"clip_criterion_if_overflowing": False},
    ],
}

WITHOUT_EFFECT_PARAMETRIZATION = [
    pytest.param(algo_name, old, id=f"{algo_name}-{'-'.join(old)}-{i}")
    for algo_name in ALGORITHMS
    for i, old in enumerate(CASES_WITHOUT_EFFECT[algo_name])
]


@pytest.mark.parametrize("algo_name", ALGORITHMS)
def test_there_is_a_test_for_every_deprecated_option(algo_name):
    tested = set()
    for old, _ in DEPRECATION_CASES[algo_name]:
        tested |= set(old)
    for old in CASES_WITHOUT_EFFECT[algo_name]:
        tested |= set(old)

    assert tested == {dep.name for dep in DEPRECATED_OPTIONS[algo_name]}


@pytest.mark.parametrize("algo_name", ALGORITHMS)
def test_deprecated_options_are_no_fields_anymore(algo_name):
    deprecated = {dep.name for dep in DEPRECATED_OPTIONS[algo_name]}
    current = {field.name for field in fields(ALGORITHMS[algo_name]())}

    assert not deprecated & current
    assert all(dep.removed_in == "0.7.0" for dep in DEPRECATED_OPTIONS[algo_name])


@pytest.mark.parametrize(("algo_name", "old", "new"), DEPRECATION_PARAMETRIZATION)
def test_deprecated_option_in_constructor(algo_name, old, new):
    algo_cls = ALGORITHMS[algo_name]
    with pytest.warns(FutureWarning) as record:
        algo = algo_cls(**old)

    assert algo == algo_cls(**_with_groups(algo_name, new))
    messages = [str(warning.message) for warning in record]
    for name in old:
        assert any(f"`{name}` of the algorithm {algo_name}" in m for m in messages)
    assert all("removed in optimagic 0.7.0" in m for m in messages)


@pytest.mark.parametrize(("algo_name", "old", "new"), DEPRECATION_PARAMETRIZATION)
def test_deprecated_option_in_with_option(algo_name, old, new):
    # Start from a configured algorithm to check that other entries are kept.
    start = ALGORITHMS[algo_name](
        stopping_maxfun=77,
        restarts=_group(algo_name, "restarts", auto_detect_history=25),
    )
    with pytest.warns(FutureWarning):
        algo = start.with_option(**old)

    expected_options = {}
    for name, value in new.items():
        if not isinstance(value, dict):
            expected_options[name] = value
        elif is_dataclass(getattr(start, name)):
            # The entries are merged into the group of the configured algorithm.
            expected_options[name] = replace(getattr(start, name), **value)
        else:
            # A boolean is replaced by a group with default entries.
            expected_options[name] = _group(algo_name, name, **value)

    assert algo == start.with_option(**expected_options)
    new_restarts = new.get("restarts", {})
    if isinstance(new_restarts, dict) and "auto_detect_history" not in new_restarts:
        assert algo.restarts.auto_detect_history == 25


@pytest.mark.parametrize(("algo_name", "old", "new"), DEPRECATION_PARAMETRIZATION)
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_deprecated_option_together_with_new_spelling_raises(algo_name, old, new):
    with pytest.raises(InvalidAlgoOptionError, match="cannot be combined"):
        ALGORITHMS[algo_name](**old, **_with_groups(algo_name, new))


@pytest.mark.parametrize("algo_name", ALGORITHMS)
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_deprecated_option_together_with_its_group_raises(algo_name):
    # The group is an instance that has a value for every entry. It is impossible to
    # tell which entries were set on purpose, so this also holds for other entries.
    group = nag.NagTrustregionOptions(expansion_factor=3.0)
    with pytest.raises(InvalidAlgoOptionError, match="cannot be combined"):
        ALGORITHMS[algo_name](
            trustregion_threshold_successful=0.2, trustregion_options=group
        )
    with pytest.raises(InvalidAlgoOptionError, match="cannot be combined"):
        ALGORITHMS[algo_name]().with_option(
            trustregion_threshold_successful=0.2, trustregion_options=group
        )


@pytest.mark.parametrize(
    ("algo_name", "group_cls"),
    [("nag_dfols", "DfolsRestartOptions"), ("nag_pybobyqa", "PybobyqaRestartOptions")],
)
def test_deprecation_warning_shows_the_option_group_class(algo_name, group_cls):
    match = rf"Use `restarts={group_cls}\(use_soft_restarts=True\)` instead"
    old = {"use_resets": True, "reset_type": "soft"}
    with pytest.warns(FutureWarning, match=match):
        ALGORITHMS[algo_name](trustregion_reset_options=old)


@pytest.mark.parametrize("algo_name", ALGORITHMS)
def test_deprecated_options_work_in_algo_options(algo_name, solve_calls):
    fun = _as_fun(_residuals, algo_name)
    with pytest.warns(FutureWarning) as record:
        om.minimize(
            fun,
            START,
            algorithm=algo_name,
            algo_options={
                "trustregion_threshold_successful": 0.2,
                "trustregion_reset_options": {"use_resets": False},
            },
        )
    user_params = solve_calls[algo_name]["user_params"]

    assert len(record) == 2
    assert user_params["tr_radius.eta1"] == 0.2
    assert user_params["restarts.use_restarts"] is False


@pytest.mark.parametrize(("algo_name", "old"), WITHOUT_EFFECT_PARAMETRIZATION)
def test_deprecated_option_without_effect_warns(algo_name, old):
    with pytest.warns(FutureWarning, match="The value you passed has no effect"):
        algo = ALGORITHMS[algo_name](**old)
    assert algo == ALGORITHMS[algo_name]()

    # Applied to a configured algorithm, nothing changes either.
    start = ALGORITHMS[algo_name](
        noisy=True, restarts=_group(algo_name, "restarts", auto_detect_history=25)
    )
    with pytest.warns(FutureWarning, match="The value you passed has no effect"):
        assert start.with_option(**old) == start


def test_deprecated_reset_options_apply_if_pybobyqa_seeks_a_global_optimum():
    # Without `use_resets` the solver decided. Py-BOBYQA also restarts if it searches
    # for a global optimum.
    with pytest.warns(FutureWarning):
        algo = ALGORITHMS["nag_pybobyqa"](
            seek_global_optimum=True,
            trustregion_reset_options={"max_unsuccessful_resets": 5},
        )
    assert algo == NagPyBOBYQA(
        seek_global_optimum=True,
        restarts=nag.PybobyqaRestartOptions(max_unsuccessful_restarts_total=5),
    )


@pytest.mark.parametrize("algo_name", ALGORITHMS)
def test_deprecated_reset_options_apply_if_noisy_is_passed_with_them(algo_name):
    with pytest.warns(FutureWarning):
        algo = ALGORITHMS[algo_name](
            noisy=True, trustregion_reset_options={"reset_type": "hard"}
        )
    assert algo == ALGORITHMS[algo_name](
        noisy=True, restarts=_group(algo_name, "restarts", use_soft_restarts=False)
    )


@pytest.mark.parametrize("algo_name", ALGORITHMS)
def test_deprecated_noise_options_behave_like_before(algo_name, solve_calls):
    """The old wrappers declared noise and its level with one option."""
    kwargs = {"stopping_maxfun": 30}
    with pytest.warns(FutureWarning):
        _minimize(algo_name, noise_additive_level=0.1, **kwargs)
    call = solve_calls[algo_name]

    assert call["objfun_has_noise"] is True
    assert call["user_params"]["noise.additive_noise_level"] == 0.1
    # Restarts are left to the solver, which uses them for noisy functions.
    assert "restarts.use_restarts" not in call["user_params"]

    with pytest.warns(FutureWarning):
        _minimize(
            algo_name, convergence_noise_corrected_criterion_tolerance=2.0, **kwargs
        )
    call = solve_calls[algo_name]

    assert call["objfun_has_noise"] is False
    assert not any(key.startswith("noise.") for key in call["user_params"])


@pytest.mark.parametrize(
    ("algo_name", "option"),
    [
        ("nag_dfols", "convergence_slow_progress"),
        ("nag_dfols", "trustregion_reset_options"),
        ("nag_dfols", "trustregion_fast_start_options"),
        ("nag_pybobyqa", "convergence_slow_progress"),
        ("nag_pybobyqa", "trustregion_reset_options"),
    ],
)
def test_unknown_entry_in_deprecated_dict_raises(algo_name, option):
    with pytest.raises(InvalidAlgoOptionError, match="has no entries"):
        ALGORITHMS[algo_name](**{option: {"not_an_entry": 1}})


@pytest.mark.parametrize(
    "old",
    [
        {"initial_directions": "diagonal"},
        {"trustregion_method_to_replace_extra_points": "other"},
        {"trustregion_reset_options": {"reset_type": "medium"}},
        {"trustregion_fast_start_options": {"method": "other"}},
        {
            "trustregion_reset_options": {
                "n_extra_interpolation_points_per_soft_reset": 2,
                "n_extra_interpolation_points_per_hard_reset": 1,
            }
        },
    ],
)
def test_invalid_value_of_deprecated_option_raises(old):
    with pytest.raises(InvalidAlgoOptionError):
        ALGORITHMS["nag_dfols"](**old)


@pytest.mark.parametrize("algo_name", ALGORITHMS)
@pytest.mark.parametrize(
    "old",
    [
        {"noise_additive_level": 0.1},
        {"convergence_noise_corrected_criterion_tolerance": 2.0},
    ],
)
@pytest.mark.parametrize("noisy", [True, False, nag.NagNoiseOptions()])
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_deprecated_noise_options_cannot_be_combined_with_noisy(algo_name, old, noisy):
    with pytest.raises(InvalidAlgoOptionError, match="cannot be combined"):
        ALGORITHMS[algo_name](noisy=noisy, **old)


@pytest.mark.parametrize("algo_name", ALGORITHMS)
def test_new_options_do_not_warn(algo_name):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _minimize(
            algo_name,
            noisy=nag.NagNoiseOptions(additive_noise_level=0.1),
            stopping_maxfun=30,
            restarts=_group(algo_name, "restarts", max_unsuccessful_restarts=1),
        )
