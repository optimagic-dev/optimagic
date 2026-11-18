"""Wrapper tests for fides.

The tests check that every option of the wrapper has the intended effect and that the
wrapper still matches the fides package. Run them with ``-m fides_wrappers``.

"""

import inspect
import time
import warnings
from dataclasses import fields
from typing import Any, get_args

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

import optimagic as om
from optimagic.config import IS_FIDES_INSTALLED
from optimagic.exceptions import InvalidAlgoOptionError
from optimagic.optimizers.fides import (
    FIDES_DEPRECATED_OPTIONS,
    FIDES_DEVIATING_DEFAULTS,
    FIDES_EXIT_MESSAGES,
    FIDES_OPTION_NAMES,
    FIDES_OPTIONS_NOT_EXPOSED,
    Fides,
    StepbackStrategy,
    SubspaceSolver,
)

pytestmark = [
    pytest.mark.fides_wrappers,
    pytest.mark.skipif(not IS_FIDES_INSTALLED, reason="fides not installed."),
]

if IS_FIDES_INSTALLED:
    import fides
    from fides import constants as fides_constants
    from fides import hessian_approximation

# Set explicitly in most tests so that the warning about its changing default does
# not show up.
BASE: dict[str, Any] = {"stepback_strategy": "reflect"}


def rosenbrock(x):
    return float((100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2).sum())


def rosenbrock_and_gradient(x):
    grad = np.zeros_like(x)
    grad[:-1] = -400 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2 * (1 - x[:-1])
    grad[1:] += 200 * (x[1:] - x[:-1] ** 2)
    return rosenbrock(x), grad


def shifted_sphere(x, target):
    return float(((x - target) ** 2).sum())


def shifted_sphere_and_gradient(x, target):
    return shifted_sphere(x, target), 2 * (x - target)


def minimize_rosenbrock(n_params: int = 4, **options: Any) -> Any:
    return om.minimize(
        fun=rosenbrock,
        fun_and_jac=rosenbrock_and_gradient,
        params=np.full(n_params, -1.0),
        algorithm=Fides(**{**BASE, **options}),
    )


def minimize_sphere_with_active_bound(**options: Any) -> Any:
    """The first step from [1, 1] hits the lower bound of the first parameter."""
    kwargs = {"target": np.array([-10.0, 3.0])}
    all_options: dict[str, Any] = {
        "trustregion_initial_radius": 10.0,
        **BASE,
        **options,
    }
    return om.minimize(
        fun=shifted_sphere,
        fun_and_jac=shifted_sphere_and_gradient,
        fun_kwargs=kwargs,
        fun_and_jac_kwargs=kwargs,
        params=np.array([1.0, 1.0]),
        bounds=om.Bounds(lower=np.zeros(2)),
        algorithm=Fides(**all_options),
    )


@pytest.fixture()
def fides_calls(monkeypatch):
    """Record the arguments with which the wrapper creates the fides Optimizer."""
    calls = []
    original = fides.Optimizer

    class SpyOptimizer(original):
        def __init__(self, *args, **kwargs):
            calls.append(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(fides, "Optimizer", SpyOptimizer)
    return calls


# ======================================================================================
# Contract with the fides package
# ======================================================================================


def test_every_fides_option_is_mapped_or_listed_as_not_exposed():
    upstream = {option.value for option in fides_constants.Options}
    assert upstream == set(FIDES_OPTION_NAMES) | set(FIDES_OPTIONS_NOT_EXPOSED)
    assert not set(FIDES_OPTION_NAMES) & set(FIDES_OPTIONS_NOT_EXPOSED)


def test_mapped_options_are_fields_of_the_wrapper():
    field_names = {field.name for field in fields(Fides)}
    assert set(FIDES_OPTION_NAMES.values()) | {"hessian_update"} == field_names


def test_defaults_equal_fides_defaults_unless_listed_as_deviating():
    defaults = {field.name: field.default for field in fields(Fides)}
    upstream = {
        option.value: value for option, value in fides_constants.DEFAULT_OPTIONS.items()
    }
    for fides_name, optimagic_name in FIDES_OPTION_NAMES.items():
        is_equal = defaults[optimagic_name] == upstream[fides_name]
        assert is_equal != (optimagic_name in FIDES_DEVIATING_DEFAULTS), optimagic_name


def test_signature_of_fides_optimizer_is_unchanged():
    init_args = set(inspect.signature(fides.Optimizer.__init__).parameters)
    assert init_args == {
        "self",
        "fun",
        "ub",
        "lb",
        "verbose",
        "options",
        "funargs",
        "hessian_update",
        "resfun",
        "x_names",
    }
    minimize_args = set(inspect.signature(fides.Optimizer.minimize).parameters)
    assert minimize_args == {"self", "x0", "start_id", "hess0"}


def test_choices_equal_fides_choices():
    subspace_solvers = {member.value for member in fides_constants.SubSpaceDim}
    stepback_strategies = {member.value for member in fides_constants.StepBackStrategy}
    assert set(get_args(SubspaceSolver)) == subspace_solvers
    assert set(get_args(StepbackStrategy)) == stepback_strategies


def test_every_exit_flag_has_a_message():
    assert {flag.name for flag in fides_constants.ExitFlag} == set(FIDES_EXIT_MESSAGES)


def test_hessian_approximations_of_fides_are_known():
    base = hessian_approximation.HessianApproximation
    classes = {
        name
        for name, obj in vars(hessian_approximation).items()
        if inspect.isclass(obj) and issubclass(obj, base) and not name.startswith("_")
    }
    supported = {"BFGS", "SR1", "DFP", "BG", "BB", "Broyden"}
    not_supported = {"HybridFixed", "HybridFraction", "FX", "SSM", "TSSM", "GNSBFGS"}
    abstract = {
        "HessianApproximation",
        "IterativeHessianApproximation",
        "HybridApproximation",
        "HybridSwitchApproximation",
        "StructuredApproximation",
    }
    assert classes == supported | not_supported | abstract


# ======================================================================================
# Forwarding of options
# ======================================================================================

FORWARDED_VALUES: dict[str, Any] = {
    "stopping_maxiter": 77,
    "stopping_maxtime": 123.0,
    "convergence_ftol_abs": 1e-3,
    "convergence_ftol_rel": 1e-4,
    "convergence_xtol_abs": 1e-5,
    "convergence_gtol_abs": 1e-6,
    "convergence_gtol_rel": 1e-7,
    "subspace_solver": "scg",
    "stepback_strategy": "mixed",
    "max_stepback_fraction": 0.8,
    "trustregion_initial_radius": 0.3,
    "trustregion_shrinking_threshold": 0.15,
    "trustregion_expansion_threshold": 0.85,
    "trustregion_shrinking_factor": 0.4,
    "trustregion_expansion_factor": 3.0,
}


def test_forwarded_values_cover_all_options():
    assert set(FORWARDED_VALUES) == set(FIDES_OPTION_NAMES.values())


@pytest.mark.parametrize("fides_name", list(FIDES_OPTION_NAMES))
def test_option_is_forwarded_under_its_fides_name(fides_name, fides_calls):
    optimagic_name = FIDES_OPTION_NAMES[fides_name]
    value = FORWARDED_VALUES[optimagic_name]
    minimize_rosenbrock(**{"stopping_maxiter": 2, optimagic_name: value})
    assert fides_calls[0]["options"][fides_name] == value


def test_only_fides_options_are_forwarded(fides_calls):
    minimize_rosenbrock(stopping_maxiter=2)
    assert set(fides_calls[0]["options"]) == set(FIDES_OPTION_NAMES)
    assert fides_calls[0]["resfun"] is False


# ======================================================================================
# Hessian approximation
# ======================================================================================


@pytest.mark.parametrize("name", ["bfgs", "sr1", "dfp", "bg", "bb", "BFGS", "Sr1"])
def test_hessian_update_names_select_the_fides_class(name, fides_calls):
    minimize_rosenbrock(hessian_update=name, stopping_maxiter=2)
    expected = getattr(hessian_approximation, name.upper())
    assert type(fides_calls[0]["hessian_update"]) is expected


def test_hessian_update_instance_is_passed_to_fides(fides_calls):
    instance = hessian_approximation.BFGS(enforce_curv_cond=False)
    minimize_rosenbrock(hessian_update=instance, stopping_maxiter=2)
    assert fides_calls[0]["hessian_update"] is instance


@pytest.mark.parametrize(("phi", "name"), [(0.0, "bfgs"), (1.0, "dfp")])
def test_broyden_instance_reproduces_named_update(phi, name):
    """The Broyden class interpolates between BFGS (phi=0) and DFP (phi=1)."""
    from_instance = minimize_rosenbrock(
        hessian_update=hessian_approximation.Broyden(phi=phi), stopping_maxiter=30
    )
    from_name = minimize_rosenbrock(hessian_update=name, stopping_maxiter=30)
    aaae(from_instance.params, from_name.params)


def test_hessian_update_changes_the_optimization_path():
    bfgs = minimize_rosenbrock(hessian_update="bfgs", stopping_maxiter=15)
    sr1 = minimize_rosenbrock(hessian_update="sr1", stopping_maxiter=15)
    assert not np.allclose(bfgs.params, sr1.params)


@pytest.mark.parametrize("hessian_update", ["bfgs", "sr1"])
def test_recommended_hessian_updates_solve_rosenbrock(hessian_update):
    res = minimize_rosenbrock(hessian_update=hessian_update)
    assert res.success
    aaae(res.params, np.ones(4), decimal=4)


def test_bfgs_approximation_is_positive_definite_and_returned():
    res = minimize_rosenbrock(hessian_update="bfgs")
    assert res.hess.shape == (4, 4)
    assert np.linalg.eigvalsh(res.hess).min() > 0


UNSUPPORTED_HESSIAN_UPDATES = [
    "FX",
    "SSM",
    "TSSM",
    "GNSBFGS",
    "HybridFixed",
    "HybridFraction",
]


@pytest.mark.parametrize("class_name", UNSUPPORTED_HESSIAN_UPDATES)
def test_hessian_update_instances_that_need_hessians_or_residuals_raise(class_name):
    instance = getattr(hessian_approximation, class_name)()
    with pytest.raises(NotImplementedError, match="not supported yet"):
        minimize_rosenbrock(hessian_update=instance)


@pytest.mark.parametrize("class_name", UNSUPPORTED_HESSIAN_UPDATES)
def test_hessian_update_names_that_need_hessians_or_residuals_raise(class_name):
    with pytest.raises(NotImplementedError, match="not supported yet"):
        minimize_rosenbrock(hessian_update=class_name.lower())


def test_broyden_by_name_raises():
    with pytest.raises(ValueError, match="phi"):
        minimize_rosenbrock(hessian_update="broyden")


def test_unknown_hessian_update_name_raises():
    with pytest.raises(ValueError, match="Invalid hessian_update"):
        minimize_rosenbrock(hessian_update="newton")


def test_hessian_update_of_wrong_type_raises():
    with pytest.raises(TypeError, match="hessian_update must be"):
        minimize_rosenbrock(hessian_update=3)


# ======================================================================================
# Convergence and stopping
# ======================================================================================

ONLY_GTOL_ABS: dict[str, Any] = {
    "convergence_ftol_abs": 0,
    "convergence_ftol_rel": 0,
    "convergence_xtol_abs": 0,
    "convergence_gtol_rel": 0,
}
NO_GTOL: dict[str, Any] = {"convergence_gtol_abs": 0, "convergence_gtol_rel": 0}


@pytest.mark.parametrize("gtol", [1e-2, 1e-8])
def test_gtol_abs_bounds_the_final_gradient_norm(gtol):
    res = minimize_rosenbrock(convergence_gtol_abs=gtol, **ONLY_GTOL_ABS)
    assert res.success
    assert res.message == FIDES_EXIT_MESSAGES["GTOL"]
    assert np.linalg.norm(res.jac) <= gtol


def test_smaller_gtol_abs_needs_more_iterations_and_is_more_precise():
    loose = minimize_rosenbrock(convergence_gtol_abs=1e-2, **ONLY_GTOL_ABS)
    strict = minimize_rosenbrock(convergence_gtol_abs=1e-8, **ONLY_GTOL_ABS)
    assert strict.n_iterations > loose.n_iterations
    assert np.abs(strict.params - 1).max() < np.abs(loose.params - 1).max()


def test_gtol_rel_scales_the_gradient_criterion_with_the_function_value():
    offset = 1000.0
    gtol_rel = 1e-4
    res = om.minimize(
        fun=lambda x: rosenbrock(x) + offset,
        fun_and_jac=lambda x: (rosenbrock(x) + offset, rosenbrock_and_gradient(x)[1]),
        params=np.full(4, -1.0),
        algorithm=Fides(
            convergence_gtol_rel=gtol_rel,
            convergence_gtol_abs=0,
            convergence_ftol_rel=0,
            **BASE,
        ),
    )
    assert res.message == FIDES_EXIT_MESSAGES["GTOL"]
    gradient_norm = np.linalg.norm(np.asarray(res.jac))
    assert gradient_norm <= gtol_rel * res.fun
    # The criterion is much looser than the absolute default of 1e-5
    assert gradient_norm > 1e-5


def test_ftol_abs_stops_when_function_value_changes_little():
    res = minimize_rosenbrock(
        convergence_ftol_abs=1e-2, convergence_ftol_rel=0, **NO_GTOL
    )
    strict = minimize_rosenbrock(
        convergence_ftol_abs=1e-10, convergence_ftol_rel=0, **NO_GTOL
    )
    assert res.message == FIDES_EXIT_MESSAGES["FTOL"]
    assert strict.message == FIDES_EXIT_MESSAGES["FTOL"]
    assert res.n_iterations < strict.n_iterations


def test_ftol_rel_stops_when_function_value_changes_little():
    offset = 1000.0

    def minimize_with_offset(ftol_rel):
        return om.minimize(
            fun=lambda x: rosenbrock(x) + offset,
            fun_and_jac=lambda x: (
                rosenbrock(x) + offset,
                rosenbrock_and_gradient(x)[1],
            ),
            params=np.full(4, -1.0),
            algorithm=Fides(convergence_ftol_rel=ftol_rel, **NO_GTOL, **BASE),
        )

    loose = minimize_with_offset(1e-4)
    strict = minimize_with_offset(1e-12)
    assert loose.message == FIDES_EXIT_MESSAGES["FTOL"]
    assert loose.n_iterations < strict.n_iterations
    assert loose.fun > strict.fun


def test_xtol_abs_stops_when_steps_become_short():
    res = minimize_rosenbrock(
        convergence_xtol_abs=1e-3, convergence_ftol_rel=0, **NO_GTOL
    )
    assert res.success
    assert res.message == FIDES_EXIT_MESSAGES["XTOL"]


def test_default_tolerances_end_rosenbrock_with_the_gradient_criterion():
    res = minimize_rosenbrock()
    assert res.message == FIDES_EXIT_MESSAGES["GTOL"]


def test_stopping_maxiter_limits_iterations():
    res = minimize_rosenbrock(stopping_maxiter=3)
    assert not res.success
    assert res.n_iterations == 3
    assert res.message == FIDES_EXIT_MESSAGES["MAXITER"]


def test_stopping_maxtime_limits_runtime():
    def slow_rosenbrock_and_gradient(x):
        time.sleep(0.02)
        return rosenbrock_and_gradient(x)

    start = time.perf_counter()
    res = om.minimize(
        fun=rosenbrock,
        fun_and_jac=slow_rosenbrock_and_gradient,
        params=np.full(4, -1.0),
        algorithm=Fides(stopping_maxtime=0.2, **BASE),
    )
    runtime = time.perf_counter() - start
    assert not res.success
    assert res.message == FIDES_EXIT_MESSAGES["MAXTIME"]
    assert runtime < 2


# ======================================================================================
# Steps: trust region, subspace solver and bounds
# ======================================================================================


@pytest.mark.parametrize("radius", [1.0, 0.1])
def test_initial_radius_is_the_length_of_the_first_step(radius):
    start = np.array([10.0, 0.0])
    kwargs = {"target": np.zeros(2)}
    res = om.minimize(
        fun=shifted_sphere,
        fun_and_jac=shifted_sphere_and_gradient,
        fun_kwargs=kwargs,
        fun_and_jac_kwargs=kwargs,
        params=start,
        algorithm=Fides(trustregion_initial_radius=radius, stopping_maxiter=1, **BASE),
    )
    assert np.linalg.norm(res.params - start) == pytest.approx(radius)


def test_small_initial_radius_keeps_first_steps_in_the_region_where_fun_is_defined():
    """With the default radius the first step leaves the domain of the function."""

    def fun_and_jac(x):
        if x[0] <= 0:
            raise ValueError("The function is only defined for x[0] > 0.")
        value = float(4 * x[0] - np.log(x[0]) + x[1] ** 2)
        return value, np.array([4 - 1 / x[0], 2 * x[1]])

    def minimize_with_radius(radius):
        return om.minimize(
            fun=lambda x: fun_and_jac(x)[0],
            fun_and_jac=fun_and_jac,
            params=np.array([0.6, 1.0]),
            algorithm=Fides(trustregion_initial_radius=radius, **BASE),
        )

    with pytest.raises(Exception, match="error occurred when evaluating"):
        minimize_with_radius(1.0)

    res = minimize_with_radius(0.1)
    aaae(res.params, np.array([0.25, 0.0]), decimal=4)


@pytest.mark.parametrize("subspace_solver", get_args(SubspaceSolver))
def test_every_subspace_solver_solves_rosenbrock(subspace_solver):
    res = minimize_rosenbrock(n_params=8, subspace_solver=subspace_solver)
    assert res.success
    aaae(res.params, np.ones(8), decimal=4)


def test_subspace_solver_changes_the_optimization_path():
    params = {
        solver: minimize_rosenbrock(
            n_params=8, subspace_solver=solver, stopping_maxiter=10
        ).params
        for solver in get_args(SubspaceSolver)
    }
    assert not np.allclose(params["2D"], params["full"])
    assert not np.allclose(params["2D"], params["scg"])


@pytest.mark.parametrize("stepback_strategy", get_args(StepbackStrategy))
def test_every_stepback_strategy_finds_the_optimum_at_the_bound(stepback_strategy):
    res = minimize_sphere_with_active_bound(stepback_strategy=stepback_strategy)
    assert res.success
    aaae(res.params, np.array([0.0, 3.0]), decimal=4)


def test_truncate_stops_at_the_bound_and_reflect_bounces_off():
    truncated = minimize_sphere_with_active_bound(
        stepback_strategy="truncate", stopping_maxiter=1
    )
    reflected = minimize_sphere_with_active_bound(
        stepback_strategy="reflect", stopping_maxiter=1
    )
    # After truncation, the second parameter still takes its full step. The reflected
    # step loses some of its length in both directions.
    aaae(truncated.params, np.array([0.05, 5.0]))
    assert reflected.params[1] < 5.0 - 1e-3


@pytest.mark.parametrize("fraction", [0.95, 0.5])
def test_max_stepback_fraction_limits_the_step_towards_a_bound(fraction):
    res = minimize_sphere_with_active_bound(
        max_stepback_fraction=fraction,
        stepback_strategy="truncate",
        stopping_maxiter=1,
    )
    distance_to_bound_at_start = 1.0
    assert res.params[0] == pytest.approx(distance_to_bound_at_start * (1 - fraction))


def test_fides_runs_without_bounds_and_with_partial_bounds():
    kwargs = {"target": np.array([-10.0, 3.0])}
    common: dict[str, Any] = {
        "fun": shifted_sphere,
        "fun_and_jac": shifted_sphere_and_gradient,
        "fun_kwargs": kwargs,
        "fun_and_jac_kwargs": kwargs,
        "params": np.array([1.0, 1.0]),
        "algorithm": Fides(**BASE),
    }
    unbounded = om.minimize(**common)
    aaae(unbounded.params, kwargs["target"], decimal=4)
    partially_bounded = om.minimize(
        **common, bounds=om.Bounds(lower=np.array([0.0, -np.inf]))
    )
    aaae(partially_bounded.params, np.array([0.0, 3.0]), decimal=4)


# ======================================================================================
# Trust-region management
# ======================================================================================


def test_radius_options_change_the_optimization_path():
    default = minimize_rosenbrock(stopping_maxiter=15).params
    changed_options: dict[str, Any] = {
        "trustregion_shrinking_threshold": 0.6,
        "trustregion_expansion_threshold": 0.3,
        "trustregion_shrinking_factor": 0.05,
        "trustregion_expansion_factor": 10.0,
    }
    for option, value in changed_options.items():
        changed = minimize_rosenbrock(stopping_maxiter=15, **{option: value}).params
        assert not np.allclose(default, changed), option


@pytest.mark.parametrize(
    "options",
    [
        {"max_stepback_fraction": 0.0},
        {"max_stepback_fraction": 1.1},
        {"trustregion_initial_radius": 0.0},
        {"trustregion_shrinking_threshold": 1.0},
        {"trustregion_expansion_threshold": -0.1},
        {"trustregion_shrinking_factor": 1.0},
        {"trustregion_expansion_factor": 1.0},
        {"subspace_solver": "3D"},
        {"stepback_strategy": "bounce"},
        {"stopping_maxtime": 0.0},
        {"convergence_gtol_abs": -1.0},
    ],
)
def test_invalid_option_values_raise_on_creation(options: dict[str, Any]):
    with pytest.raises(InvalidAlgoOptionError):
        Fides(**options)


# ======================================================================================
# Changing defaults
# ======================================================================================


def test_changing_default_of_stepback_strategy_warns_and_keeps_current_behavior(
    fides_calls,
):
    match = "`stepback_strategy` of the algorithm fides will change from 'truncate'"
    with pytest.warns(FutureWarning, match=match) as record:
        om.minimize(
            fun=rosenbrock,
            fun_and_jac=rosenbrock_and_gradient,
            params=np.full(4, -1.0),
            bounds=om.Bounds(lower=np.full(4, -5.0)),
            algorithm=Fides(stopping_maxiter=2),
        )
    assert "to 'reflect' in optimagic 0.7.0" in str(record[0].message)
    assert fides_calls[0]["options"]["stepback_strategy"] == "truncate"


def test_default_stepback_strategy_does_not_warn_without_bounds():
    """Without bounds the stepback strategy is never used."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        om.minimize(
            fun=rosenbrock,
            fun_and_jac=rosenbrock_and_gradient,
            params=np.full(4, -1.0),
            algorithm=Fides(stopping_maxiter=2),
        )


def test_explicit_options_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        minimize_rosenbrock(stopping_maxiter=2)


# ======================================================================================
# Deprecated option names
# ======================================================================================

DEPRECATED_VALUES: dict[str, Any] = {
    "hessian_update_strategy": "sr1",
    "stopping_max_seconds": 100.0,
    "trustregion_subspace_dimension": "scg",
    "trustregion_stepback_strategy": "mixed",
    "trustregion_max_stepback_fraction": 0.8,
    "trustregion_decrease_threshold": 0.15,
    "trustregion_increase_threshold": 0.85,
    "trustregion_decrease_factor": 0.4,
    "trustregion_increase_factor": 3.0,
}


def test_deprecated_values_cover_all_deprecated_options():
    assert set(DEPRECATED_VALUES) == {dep.name for dep in FIDES_DEPRECATED_OPTIONS}


@pytest.mark.parametrize("deprecated", FIDES_DEPRECATED_OPTIONS, ids=lambda d: d.name)
def test_deprecated_option_warns_and_equals_new_spelling(deprecated):
    value = DEPRECATED_VALUES[deprecated.name]
    expected = Fides(**{deprecated.replacement: value})

    match = f"`{deprecated.name}` of the algorithm fides is deprecated"
    with pytest.warns(FutureWarning, match=match) as record:
        from_constructor = Fides(**{deprecated.name: value})
    assert "removed in optimagic 0.7.0" in str(record[0].message)
    assert f"{deprecated.replacement}={value!r}" in str(record[0].message)

    with pytest.warns(FutureWarning, match=match):
        from_with_option = Fides().with_option(**{deprecated.name: value})

    assert from_constructor == expected
    assert from_with_option == expected


def test_deprecated_option_works_in_algo_options(fides_calls):
    with pytest.warns(FutureWarning, match="trustregion_subspace_dimension"):
        om.minimize(
            fun=rosenbrock,
            fun_and_jac=rosenbrock_and_gradient,
            params=np.full(4, -1.0),
            algorithm="fides",
            algo_options={
                "trustregion_subspace_dimension": "scg",
                "stepback_strategy": "reflect",
                "stopping_maxiter": 2,
            },
        )
    assert fides_calls[0]["options"]["subspace_solver"] == "scg"


@pytest.mark.parametrize("deprecated", FIDES_DEPRECATED_OPTIONS, ids=lambda d: d.name)
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_deprecated_option_together_with_new_spelling_raises(deprecated):
    value = DEPRECATED_VALUES[deprecated.name]
    with pytest.raises(InvalidAlgoOptionError, match="cannot be combined"):
        Fides(**{deprecated.name: value, deprecated.replacement: value})
