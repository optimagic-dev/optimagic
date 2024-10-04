"""Test that our deprecations work.

This also serves as an internal overview of deprecated functions.

"""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_almost_equal as aaae

import estimagic as em
import optimagic as om
from estimagic import (
    OptimizeLogReader,
    OptimizeResult,
    batch_evaluators,
    check_constraints,
    convergence_plot,
    convergence_report,
    count_free_params,
    criterion_plot,
    first_derivative,
    get_benchmark_problems,
    maximize,
    minimize,
    params_plot,
    profile_plot,
    rank_report,
    run_benchmark,
    second_derivative,
    slice_plot,
    traceback_report,
    utilities,
)
from optimagic.deprecations import (
    convert_dict_to_function_value,
    handle_log_options_throw_deprecated_warning,
    infer_problem_type_from_dict_output,
    is_dict_output,
    pre_process_constraints,
)
from optimagic.differentiation.derivatives import NumdiffResult
from optimagic.exceptions import InvalidConstraintError
from optimagic.logging.logger import SQLiteLogOptions
from optimagic.optimization.fun_value import (
    LeastSquaresFunctionValue,
    LikelihoodFunctionValue,
    ScalarFunctionValue,
)
from optimagic.parameters.bounds import Bounds
from optimagic.typing import AggregationLevel

# ======================================================================================
# Deprecated in 0.5.0, remove in 0.6.0
# ======================================================================================


def test_estimagic_minimize_is_deprecated():
    with pytest.warns(FutureWarning, match="estimagic.minimize has been deprecated"):
        minimize(lambda x: x @ x, np.arange(3), algorithm="scipy_lbfgsb")


def test_estimagic_maximize_is_deprecated():
    with pytest.warns(FutureWarning, match="estimagic.maximize has been deprecated"):
        maximize(lambda x: -x @ x, np.arange(3), algorithm="scipy_lbfgsb")


def test_estimagic_first_derivative_is_deprecated():
    msg = "estimagic.first_derivative has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        first_derivative(lambda x: x @ x, np.arange(3))


def test_estimagic_second_derivative_is_deprecated():
    msg = "estimagic.second_derivative has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        second_derivative(lambda x: x @ x, np.arange(3))


def test_estimagic_benchmarking_functions_are_deprecated():
    msg = "estimagic.get_benchmark_problems has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        problems = get_benchmark_problems("example")

    msg = "estimagic.run_benchmark has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        results = run_benchmark(
            problems, optimize_options={"test": {"algorithm": "scipy_lbfgsb"}}
        )

    msg = "estimagic.convergence_report has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        convergence_report(problems, results)

    msg = "estimagic.rank_report has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        rank_report(problems, results)

    msg = "estimagic.traceback_report has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        traceback_report(problems, results)

    msg = "estimagic.profile_plot has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        profile_plot(problems, results)

    msg = "estimagic.convergence_plot has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        convergence_plot(problems, results)


def test_estimagic_slice_plot_is_deprecated():
    msg = "estimagic.slice_plot has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        slice_plot(
            func=lambda x: x @ x,
            params=np.arange(3),
            bounds=Bounds(lower=np.zeros(3), upper=np.ones(3) * 5),
        )


def test_estimagic_check_constraints_is_deprecated():
    msg = "estimagic.check_constraints has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        check_constraints(
            params=np.arange(3),
            constraints=om.FixedConstraint(lambda x: x[0]),
        )


def test_estimagic_count_free_params_is_deprecated():
    msg = "estimagic.count_free_params has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        count_free_params(
            params=np.arange(3),
            constraints=om.FixedConstraint(lambda x: x[0]),
        )


@pytest.fixture()
def example_db(tmp_path):
    path = tmp_path / "test.db"

    def _crit(params):
        x = np.array(list(params.values()))
        return x @ x

    om.minimize(
        fun=_crit,
        params={"a": 1, "b": 2, "c": 3},
        algorithm="scipy_lbfgsb",
        logging=path,
    )
    return path


def test_estimagic_log_reader_is_deprecated(example_db):
    msg = "OptimizeLogReader is deprecated and will be removed in a future "
    "version. Please use optimagic.logging.SQLiteLogger instead."
    with pytest.warns(FutureWarning, match=msg):
        OptimizeLogReader(example_db)


def test_estimagic_optimize_result_is_deprecated():
    res = om.minimize(lambda x: x @ x, np.arange(3), algorithm="scipy_lbfgsb")

    msg = "estimagic.OptimizeResult has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        OptimizeResult(
            params=res.params,
            fun=res.fun,
            start_fun=res.start_fun,
            start_params=res.start_params,
            algorithm=res.algorithm,
            direction=res.direction,
            n_free=res.n_free,
        )


def test_estimagic_chol_params_to_lower_triangular_matrix_is_deprecated():
    msg = "estimagic.utilities.chol_params_to_lower_triangular_matrix has been deprecat"
    with pytest.warns(FutureWarning, match=msg):
        utilities.chol_params_to_lower_triangular_matrix(np.arange(6))


def test_estimagic_cov_params_to_matrix_is_deprecated():
    msg = "estimagic.utilities.cov_params_to_matrix has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        utilities.cov_params_to_matrix(np.arange(6))


def test_estimagic_cov_matrix_to_params_is_deprecated():
    msg = "estimagic.utilities.cov_matrix_to_params has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        utilities.cov_matrix_to_params(np.eye(3))


def test_estimagic_sdcorr_params_to_sds_and_corr_is_deprecated():
    msg = "estimagic.utilities.sdcorr_params_to_sds_and_corr has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        utilities.sdcorr_params_to_sds_and_corr(np.arange(6))


def test_estimagic_sds_and_corr_to_cov_is_deprecated():
    msg = "estimagic.utilities.sds_and_corr_to_cov has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        utilities.sds_and_corr_to_cov(np.arange(3), np.eye(3))


def test_estimagic_cov_to_sds_and_corr_is_deprecated():
    msg = "estimagic.utilities.cov_to_sds_and_corr has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        utilities.cov_to_sds_and_corr(np.eye(3))


def test_estimagic_sdcorr_params_to_matrix_is_deprecated():
    msg = "estimagic.utilities.sdcorr_params_to_matrix has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        utilities.sdcorr_params_to_matrix(np.arange(6))


def test_estimagic_cov_matrix_to_sdcorr_params_is_deprecated():
    msg = "estimagic.utilities.cov_matrix_to_sdcorr_params has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        utilities.cov_matrix_to_sdcorr_params(np.eye(3))


def test_estimagic_number_of_triangular_elements_to_dimension_is_deprecated():
    msg = "estimagic.utilities.number_of_triangular_elements_to_dimension has been"
    with pytest.warns(FutureWarning, match=msg):
        utilities.number_of_triangular_elements_to_dimension(6)


def test_estimagic_dimension_to_number_of_triangular_elements_is_deprecated():
    msg = "estimagic.utilities.dimension_to_number_of_triangular_elements has been"
    with pytest.warns(FutureWarning, match=msg):
        utilities.dimension_to_number_of_triangular_elements(3)


def test_estimagic_propose_alternatives_is_deprecated():
    msg = "estimagic.utilities.propose_alternatives has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        utilities.propose_alternatives("estimagic", list("abcdefg"))


def test_estimagic_robust_cholesky_is_deprecated():
    msg = "estimagic.utilities.robust_cholesky has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        utilities.robust_cholesky(np.eye(3))


def test_estimagic_robust_inverse_is_deprecated():
    msg = "estimagic.utilities.robust_inverse has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        utilities.robust_inverse(np.eye(3))


def test_estimagic_hash_array_is_deprecated():
    msg = "estimagic.utilities.hash_array has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        utilities.hash_array(np.arange(3))


def test_estimagic_calculate_trustregion_initial_radius_is_deprecated():
    msg = "estimagic.utilities.calculate_trustregion_initial_radius has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        utilities.calculate_trustregion_initial_radius(np.arange(3))


def test_estimagic_pickle_functions_are_deprecated(tmp_path):
    msg = "estimagic.utilities.to_pickle has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        utilities.to_pickle(np.arange(3), tmp_path / "test.pkl")

    msg = "estimagic.utilities.read_pickle has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        utilities.read_pickle(tmp_path / "test.pkl")


def test_estimagic_isscalar_is_deprecated():
    msg = "estimagic.utilities.isscalar has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        utilities.isscalar(1)


def test_estimagic_get_rng_is_deprecated():
    msg = "estimagic.utilities.get_rng has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        utilities.get_rng(42)


def test_estimagic_criterion_plot_is_deprecated():
    msg = "estimagic.criterion_plot has been deprecated"
    res = om.minimize(lambda x: x @ x, np.arange(3), algorithm="scipy_lbfgsb")
    with pytest.warns(FutureWarning, match=msg):
        criterion_plot(res)


def test_estimagic_params_plot_is_deprecated():
    msg = "estimagic.params_plot has been deprecated"
    res = om.minimize(lambda x: x @ x, np.arange(3), algorithm="scipy_lbfgsb")
    with pytest.warns(FutureWarning, match=msg):
        params_plot(res)


def test_criterion_is_depracated():
    msg = "the `criterion` argument has been renamed"
    with pytest.warns(FutureWarning, match=msg):
        om.minimize(
            criterion=lambda x: x @ x,
            params=np.arange(3),
            algorithm="scipy_lbfgsb",
        )


def test_criterion_kwargs_is_deprecated():
    msg = "the `criterion_kwargs` argument has been renamed"
    with pytest.warns(FutureWarning, match=msg):
        om.minimize(
            lambda x, a: x @ x,
            params=np.arange(3),
            algorithm="scipy_lbfgsb",
            criterion_kwargs={"a": 1},
        )


def test_derivative_is_deprecated():
    msg = "the `derivative` argument has been renamed"
    with pytest.warns(FutureWarning, match=msg):
        om.minimize(
            lambda x: x @ x,
            params=np.arange(3),
            algorithm="scipy_lbfgsb",
            derivative=lambda x: 2 * x,
        )


def test_derivative_kwargs_is_deprecated():
    msg = "the `derivative_kwargs` argument has been renamed"
    with pytest.warns(FutureWarning, match=msg):
        om.minimize(
            lambda x: x @ x,
            params=np.arange(3),
            algorithm="scipy_lbfgsb",
            jac=lambda x, a: 2 * x,
            derivative_kwargs={"a": 1},
        )


def test_criterion_and_derivative_is_deprecated():
    msg = "the `criterion_and_derivative` argument has been renamed"
    with pytest.warns(FutureWarning, match=msg):
        om.minimize(
            lambda x: x @ x,
            params=np.arange(3),
            algorithm="scipy_lbfgsb",
            criterion_and_derivative=lambda x: (x @ x, 2 * x),
        )


def test_criterion_and_derivative_kwargs_is_deprecated():
    msg = "the `criterion_and_derivative_kwargs` argument has been renamed"
    with pytest.warns(FutureWarning, match=msg):
        om.minimize(
            lambda x: x @ x,
            params=np.arange(3),
            algorithm="scipy_lbfgsb",
            fun_and_jac=lambda x, a: (x @ x, 2 * x),
            criterion_and_derivative_kwargs={"a": 1},
        )


ALGO_OPTIONS = [
    {"convergence_absolute_criterion_tolerance": 1e-8},
    {"convergence_relative_criterion_tolerance": 1e-8},
    {"convergence_absolute_params_tolerance": 1e-8},
    {"convergence_relative_params_tolerance": 1e-8},
    {"convergence_absolute_gradient_tolerance": 1e-8},
    {"convergence_relative_gradient_tolerance": 1e-8},
    {"convergence_scaled_gradient_tolerance": 1e-8},
    {"stopping_max_iterations": 1_000},
    {"stopping_max_criterion_evaluations": 1_000},
]


@pytest.mark.parametrize("algo_option", ALGO_OPTIONS)
def test_old_convergence_criteria_are_deprecated(algo_option):
    msg = "The following keys in `algo_options` are deprecated"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        with pytest.warns(FutureWarning, match=msg):
            om.minimize(
                lambda x: x @ x,
                params=np.arange(3),
                algorithm="scipy_lbfgsb",
                algo_options=algo_option,
            )


def test_deprecated_attributes_of_optimize_result():
    res = om.minimize(lambda x: x @ x, np.arange(3), algorithm="scipy_lbfgsb")

    msg = "attribute is deprecated"

    with pytest.warns(FutureWarning, match=msg):
        _ = res.n_criterion_evaluations

    with pytest.warns(FutureWarning, match=msg):
        _ = res.n_derivative_evaluations

    with pytest.warns(FutureWarning, match=msg):
        _ = res.criterion

    with pytest.warns(FutureWarning, match=msg):
        _ = res.start_criterion


BOUNDS_KWARGS = [
    {"lower_bounds": np.full(3, -1)},
    {"upper_bounds": np.full(3, 2)},
]

SOFT_BOUNDS_KWARGS = [
    {"soft_lower_bounds": np.full(3, -1)},
    {"soft_upper_bounds": np.full(3, 1)},
]


@pytest.mark.parametrize("bounds_kwargs", BOUNDS_KWARGS + SOFT_BOUNDS_KWARGS)
def test_old_bounds_are_deprecated_in_minimize(bounds_kwargs):
    msg = "Specifying bounds via the arguments"
    with pytest.warns(FutureWarning, match=msg):
        om.minimize(
            lambda x: x @ x,
            np.arange(3),
            algorithm="scipy_lbfgsb",
            **bounds_kwargs,
        )


@pytest.mark.parametrize("bounds_kwargs", BOUNDS_KWARGS + SOFT_BOUNDS_KWARGS)
def test_old_bounds_are_deprecated_in_maximize(bounds_kwargs):
    msg = "Specifying bounds via the arguments"
    with pytest.warns(FutureWarning, match=msg):
        om.maximize(
            lambda x: -x @ x,
            np.arange(3),
            algorithm="scipy_lbfgsb",
            **bounds_kwargs,
        )


@pytest.mark.parametrize("bounds_kwargs", BOUNDS_KWARGS)
def test_old_bounds_are_deprecated_in_first_derivative(bounds_kwargs):
    msg = "Specifying bounds via the arguments"
    with pytest.warns(FutureWarning, match=msg):
        om.first_derivative(
            lambda x: x @ x,
            np.arange(3),
            **bounds_kwargs,
        )


@pytest.mark.parametrize("bounds_kwargs", BOUNDS_KWARGS)
def test_old_bounds_are_deprecated_in_second_derivative(bounds_kwargs):
    msg = "Specifying bounds via the arguments"
    with pytest.warns(FutureWarning, match=msg):
        om.second_derivative(
            lambda x: x @ x,
            np.arange(3),
            **bounds_kwargs,
        )


@pytest.mark.parametrize("bounds_kwargs", BOUNDS_KWARGS)
def test_old_bounds_are_deprecated_in_estimate_ml(bounds_kwargs):
    msg = "Specifying bounds via the arguments"
    with pytest.warns(FutureWarning, match=msg):

        @om.mark.likelihood
        def loglike(x):
            return -(x**2)

        em.estimate_ml(
            loglike=loglike,
            params=np.arange(3),
            optimize_options={"algorithm": "scipy_lbfgsb"},
            **bounds_kwargs,
        )


def test_numdiff_options_is_deprecated_in_estimate_ml():
    msg = "The argument `numdiff_options` is deprecated"
    with pytest.warns(FutureWarning, match=msg):

        @om.mark.likelihood
        def loglike(x):
            return -(x**2)

        em.estimate_ml(
            loglike=loglike,
            params=np.arange(3),
            optimize_options={"algorithm": "scipy_lbfgsb"},
            numdiff_options={"method": "forward"},
        )


@pytest.mark.parametrize("bounds_kwargs", BOUNDS_KWARGS)
def test_old_bounds_are_deprecated_in_estimate_msm(bounds_kwargs):
    msg = "Specifying bounds via the arguments"
    with pytest.warns(FutureWarning, match=msg):
        em.estimate_msm(
            simulate_moments=lambda x: x,
            empirical_moments=np.zeros(3),
            moments_cov=np.eye(3),
            params=np.arange(3),
            optimize_options={"algorithm": "scipy_lbfgsb"},
            **bounds_kwargs,
        )


def test_numdiff_options_is_deprecated_in_estimate_msm():
    msg = "The argument `numdiff_options` is deprecated"
    with pytest.warns(FutureWarning, match=msg):
        em.estimate_msm(
            simulate_moments=lambda x: x,
            empirical_moments=np.zeros(3),
            moments_cov=np.eye(3),
            params=np.arange(3),
            optimize_options={"algorithm": "scipy_lbfgsb"},
            numdiff_options={"method": "forward"},
        )


@pytest.mark.parametrize("bounds_kwargs", BOUNDS_KWARGS)
def test_old_bounds_are_deprecated_in_count_free_params(bounds_kwargs):
    msg = "Specifying bounds via the arguments"
    with pytest.warns(FutureWarning, match=msg):
        om.count_free_params(
            np.arange(3),
            constraints=om.FixedConstraint(lambda x: x[0]),
            **bounds_kwargs,
        )


@pytest.mark.parametrize("bounds_kwargs", BOUNDS_KWARGS)
def test_old_bounds_are_deprecated_in_check_constraints(bounds_kwargs):
    msg = "Specifying bounds via the arguments"
    with pytest.warns(FutureWarning, match=msg):
        om.check_constraints(
            np.arange(3),
            constraints=om.FixedConstraint(lambda x: x[0]),
            **bounds_kwargs,
        )


def test_old_bounds_are_deprecated_in_slice_plot():
    msg = "Specifying bounds via the arguments"
    with pytest.warns(FutureWarning, match=msg):
        om.slice_plot(
            lambda x: x @ x,
            np.arange(3),
            lower_bounds=np.full(3, -1),
            upper_bounds=np.full(3, 2),
        )


def test_is_dict_output():
    assert is_dict_output({"value": 1})
    assert not is_dict_output(1)


def test_infer_problem_type_from_dict_output():
    assert infer_problem_type_from_dict_output({"value": 1}) == AggregationLevel.SCALAR
    assert (
        infer_problem_type_from_dict_output({"value": 1, "root_contributions": 2})
        == AggregationLevel.LEAST_SQUARES
    )
    assert (
        infer_problem_type_from_dict_output({"value": 1, "contributions": 2})
        == AggregationLevel.LIKELIHOOD
    )


def test_convert_value_dict_to_function_value():
    got = convert_dict_to_function_value({"value": 1})
    assert isinstance(got, ScalarFunctionValue)
    assert got.value == 1


def test_convert_root_contributions_dict_to_function_value():
    got = convert_dict_to_function_value({"value": 5, "root_contributions": [1, 2]})
    assert isinstance(got, LeastSquaresFunctionValue)
    assert got.value == [1, 2]


def test_convert_contributions_dict_to_function_value():
    got = convert_dict_to_function_value({"value": 5, "contributions": [1, 4]})
    assert isinstance(got, LikelihoodFunctionValue)
    assert got.value == [1, 4]


def test_old_scaling_options_are_deprecated_in_minimize():
    msg = "Specifying scaling options via the argument `scaling_options` is deprecated"
    with pytest.warns(FutureWarning, match=msg):
        om.minimize(
            lambda x: x @ x,
            np.arange(3),
            algorithm="scipy_lbfgsb",
            scaling_options={"method": "start_values", "magnitude": 1},
        )


def test_old_scaling_options_are_deprecated_in_maximize():
    msg = "Specifying scaling options via the argument `scaling_options` is deprecated"
    with pytest.warns(FutureWarning, match=msg):
        om.maximize(
            lambda x: -x @ x,
            np.arange(3),
            algorithm="scipy_lbfgsb",
            scaling_options={"method": "start_values", "magnitude": 1},
        )


def test_old_multistart_options_are_deprecated_in_minimize():
    msg = "Specifying multistart options via the argument `multistart_options` is"
    with pytest.warns(FutureWarning, match=msg):
        om.minimize(
            lambda x: x @ x,
            np.arange(3),
            algorithm="scipy_lbfgsb",
            multistart_options={"n_samples": 10},
        )


def test_old_multistart_options_are_deprecated_in_maximize():
    msg = "Specifying multistart options via the argument `multistart_options` is"
    with pytest.warns(FutureWarning, match=msg):
        om.maximize(
            lambda x: -x @ x,
            np.arange(3),
            algorithm="scipy_lbfgsb",
            multistart_options={"n_samples": 10},
        )


def test_multistart_option_share_optimization_option_is_deprecated():
    msg = "The `share_optimization` option is deprecated and will be removed in"
    with pytest.warns(FutureWarning, match=msg):
        om.minimize(
            lambda x: x @ x,
            np.arange(3),
            algorithm="scipy_lbfgsb",
            bounds=om.Bounds(lower=np.full(3, -1), upper=np.full(3, 2)),
            multistart={"share_optimization": 0.1},
        )


def test_multistart_option_convergence_relative_params_tolerance_option_is_deprecated():
    msg = "The `convergence_relative_params_tolerance` option is deprecated and will"
    with pytest.warns(FutureWarning, match=msg):
        om.minimize(
            lambda x: x @ x,
            np.arange(3),
            algorithm="scipy_lbfgsb",
            bounds=om.Bounds(lower=np.full(3, -1), upper=np.full(3, 2)),
            multistart={"convergence_relative_params_tolerance": 0.01},
        )


def test_multistart_option_optimization_error_handling_option_is_deprecated():
    msg = "The `optimization_error_handling` option is deprecated and will be removed"
    with pytest.warns(FutureWarning, match=msg):
        om.minimize(
            lambda x: x @ x,
            np.arange(3),
            algorithm="scipy_lbfgsb",
            bounds=om.Bounds(lower=np.full(3, -1), upper=np.full(3, 2)),
            multistart={"optimization_error_handling": "continue"},
        )


def test_multistart_option_exploration_error_handling_option_is_deprecated():
    msg = "The `exploration_error_handling` option is deprecated and will be removed"
    with pytest.warns(FutureWarning, match=msg):
        om.minimize(
            lambda x: x @ x,
            np.arange(3),
            algorithm="scipy_lbfgsb",
            bounds=om.Bounds(lower=np.full(3, -1), upper=np.full(3, 2)),
            multistart={"exploration_error_handling": "continue"},
        )


def test_deprecated_dict_access_of_multistart_info():
    res = om.minimize(
        lambda x: x @ x,
        np.arange(3),
        algorithm="scipy_lbfgsb",
        multistart=True,
        bounds=om.Bounds(lower=np.full(3, -1), upper=np.full(3, 2)),
    )
    msg = "The dictionary access for 'local_optima' is deprecated and will be removed"
    with pytest.warns(FutureWarning, match=msg):
        _ = res.multistart_info["local_optima"]


def test_base_steps_in_first_derivatives_is_deprecated():
    msg = "The `base_steps` argument is deprecated and will be removed alongside"
    with pytest.warns(FutureWarning, match=msg):
        om.first_derivative(lambda x: x @ x, np.arange(3), base_steps=1e-3)


def test_step_ratio_in_first_derivatives_is_deprecated():
    msg = "The `step_ratio` argument is deprecated and will be removed alongside"
    with pytest.warns(FutureWarning, match=msg):
        om.first_derivative(lambda x: x @ x, np.arange(3), step_ratio=2)


def test_n_steps_in_first_derivatives_is_deprecated():
    msg = "The `n_steps` argument is deprecated and will be removed alongside"
    with pytest.warns(FutureWarning, match=msg):
        om.first_derivative(lambda x: x @ x, np.arange(3), n_steps=2)


def test_return_info_in_first_derivatives_is_deprecated():
    msg = "The `return_info` argument is deprecated and will be removed alongside"
    with pytest.warns(FutureWarning, match=msg):
        om.first_derivative(lambda x: x @ x, np.arange(3), return_info=True)


def test_return_func_value_in_first_derivatives_is_deprecated():
    msg = "The `return_func_value` argument is deprecated and will be removed in"
    with pytest.warns(FutureWarning, match=msg):
        om.first_derivative(lambda x: x @ x, np.arange(3), return_func_value=True)


def test_base_steps_in_second_derivatives_is_deprecated():
    msg = "The `base_steps` argument is deprecated and will be removed alongside"
    with pytest.warns(FutureWarning, match=msg):
        om.second_derivative(lambda x: x @ x, np.arange(3), base_steps=1e-3)


def test_step_ratio_in_second_derivatives_is_deprecated():
    msg = "The `step_ratio` argument is deprecated and will be removed alongside"
    with pytest.warns(FutureWarning, match=msg):
        om.second_derivative(lambda x: x @ x, np.arange(3), step_ratio=2)


def test_n_steps_in_second_derivatives_is_deprecated():
    msg = "The `n_steps` argument is deprecated and will be removed alongside"
    with pytest.warns(FutureWarning, match=msg):
        om.second_derivative(lambda x: x @ x, np.arange(3), n_steps=1)


def test_return_func_value_in_second_derivatives_is_deprecated():
    msg = "The `return_func_value` argument is deprecated and will be removed in"
    with pytest.warns(FutureWarning, match=msg):
        om.second_derivative(lambda x: x @ x, np.arange(3), return_func_value=True)


def test_return_info_in_second_derivatives_is_deprecated():
    msg = "The `return_info` argument is deprecated and will be removed alongside"
    with pytest.warns(FutureWarning, match=msg):
        om.second_derivative(lambda x: x @ x, np.arange(3), return_info=True)


def test_numdiff_result_func_evals_is_deprecated():
    msg = "The `func_evals` attribute is deprecated and will be removed in optimagic"
    res = NumdiffResult(derivative=1)
    with pytest.warns(FutureWarning, match=msg):
        _ = res.func_evals


def test_numdiff_result_derivative_candidates_is_deprecated():
    msg = "The `derivative_candidates` attribute is deprecated and will be removed"
    res = NumdiffResult(derivative=1)
    with pytest.warns(FutureWarning, match=msg):
        _ = res.derivative_candidates


def test_numdiff_result_dict_access_is_deprecated():
    msg = "The dictionary access for 'derivative' is deprecated and will be removed"
    res = NumdiffResult(derivative=1)
    with pytest.warns(FutureWarning, match=msg):
        _ = res["derivative"]


def test_key_argument_is_deprecated_in_first_derivative():
    with pytest.warns(FutureWarning, match="The `key` argument in"):
        om.first_derivative(lambda x: {"value": x @ x}, np.arange(3), key="value")


def test_key_argument_is_deprecated_in_second_derivative():
    with pytest.warns(FutureWarning, match="The `key` argument in"):
        om.second_derivative(lambda x: {"value": x @ x}, np.arange(3), key="value")


def test_jac_dicts_are_deprecated_in_minimize():
    msg = "Specifying a dictionary of jac functions is deprecated"
    with pytest.warns(FutureWarning, match=msg):
        res = om.minimize(
            lambda x: x @ x,
            np.arange(3),
            algorithm="scipy_lbfgsb",
            jac={"value": lambda x: 2 * x},
        )
        aaae(res.params, np.zeros(3))


def test_jac_dicts_are_deprecated_in_maximize():
    msg = "Specifying a dictionary of jac functions is deprecated"
    with pytest.warns(FutureWarning, match=msg):
        res = om.maximize(
            lambda x: -x @ x,
            np.arange(3),
            algorithm="scipy_lbfgsb",
            jac={"value": lambda x: -2 * x},
        )
        aaae(res.params, np.zeros(3))


def test_fun_and_jac_dicts_are_deprecated_in_minimize():
    msg = "Specifying a dictionary of fun_and_jac functions is deprecated"
    with pytest.warns(FutureWarning, match=msg):
        res = om.minimize(
            lambda x: x @ x,
            np.arange(3),
            algorithm="scipy_lbfgsb",
            fun_and_jac={"value": lambda x: (x @ x, 2 * x)},
        )
        aaae(res.params, np.zeros(3))


def test_fun_and_jac_dicts_are_deprecated_in_maximize():
    msg = "Specifying a dictionary of fun_and_jac functions is deprecated"
    with pytest.warns(FutureWarning, match=msg):
        res = om.maximize(
            lambda x: -x @ x,
            np.arange(3),
            algorithm="scipy_lbfgsb",
            fun_and_jac={"value": lambda x: (-x @ x, -2 * x)},
        )
        aaae(res.params, np.zeros(3))


def test_fun_with_dict_return_is_deprecated_in_minimize():
    msg = "Returning a dictionary with the special keys"
    with pytest.warns(FutureWarning, match=msg):
        res = om.minimize(
            lambda x: {"value": x @ x},
            np.arange(3),
            algorithm="scipy_lbfgsb",
        )
        aaae(res.params, np.zeros(3))


def test_fun_with_dict_return_is_deprecated_in_slice_plot():
    msg = "Functions that return dictionaries"
    with pytest.warns(FutureWarning, match=msg):
        om.slice_plot(
            lambda x: {"value": x @ x},
            np.arange(3),
            bounds=om.Bounds(lower=np.zeros(3), upper=np.ones(3) * 5),
        )


def test_handle_log_options():
    msg = (
        "Usage of the parameter log_options is deprecated "
        "and will be removed in a future version. "
        "Provide a LogOptions instance for the parameter `logging`, if you need to "
        "configure the logging."
    )
    log_options = {"fast_logging": True}
    with pytest.warns(FutureWarning, match=msg):
        logger = None
        handled_logger = handle_log_options_throw_deprecated_warning(
            log_options, logger
        )
        assert handled_logger is None

    creation_warning = (
        f"\nUsing {log_options=} to create an instance of SQLiteLogOptions. "
        f"This mechanism will be removed in the future."
    )

    with pytest.warns(match=creation_warning):
        handled_logger = handle_log_options_throw_deprecated_warning(
            log_options, ":memory:"
        )
        assert isinstance(handled_logger, SQLiteLogOptions)

    incompatibility_msg = "Found string or path for logger argument, but parameter"
    f" {log_options=} is not compatible "
    log_options_typo = {"fast_lugging": False}

    with pytest.raises(ValueError, match=incompatibility_msg):
        handled_logger = handle_log_options_throw_deprecated_warning(
            log_options_typo, ":memory:"
        )
        assert handled_logger == ":memory:"


def test_log_options_are_deprecated_in_estimate_ml(tmp_path):
    with pytest.warns(FutureWarning, match="LogOptions"):

        @om.mark.likelihood
        def loglike(x):
            return -(x**2)

        em.estimate_ml(
            loglike=loglike,
            params=np.arange(3),
            optimize_options={"algorithm": "scipy_lbfgsb"},
            logging=tmp_path / "log.db",
            log_options={"fast_logging": True, "if_database_exists": "replace"},
        )

    with pytest.warns(FutureWarning, match="if_table_exists"):

        @om.mark.likelihood
        def loglike(x):
            return -(x**2)

        em.estimate_ml(
            loglike=loglike,
            params=np.arange(3),
            optimize_options={"algorithm": "scipy_lbfgsb"},
            logging=tmp_path / "log_1.db",
            log_options={"fast_logging": True, "if_table_exists": "replace"},
        )


def test_log_options_are_deprecated_in_estimate_msm(tmp_path):
    with pytest.warns(FutureWarning, match="LogOptions"):

        @om.mark.likelihood
        def loglike(x):
            return -(x**2)

        em.estimate_msm(
            simulate_moments=lambda x: x,
            empirical_moments=np.zeros(3),
            moments_cov=np.eye(3),
            params=np.arange(3),
            optimize_options={"algorithm": "scipy_lbfgsb"},
            logging=tmp_path / "log.db",
            log_options={"fast_logging": True, "if_database_exists": "replace"},
        )

    with pytest.warns(FutureWarning, match="if_table_exists"):

        @om.mark.likelihood
        def loglike(x):
            return -(x**2)

        em.estimate_msm(
            simulate_moments=lambda x: x,
            empirical_moments=np.zeros(3),
            moments_cov=np.eye(3),
            params=np.arange(3),
            optimize_options={"algorithm": "scipy_lbfgsb"},
            logging=tmp_path / "log_1.db",
            log_options={"fast_logging": True, "if_table_exists": "replace"},
        )


def test_log_options_are_deprecated_in_minimize(tmp_path):
    with pytest.warns(FutureWarning, match="LogOptions"):
        om.minimize(
            lambda x: x @ x,
            np.arange(3),
            algorithm="scipy_lbfgsb",
            logging=tmp_path / "log.db",
            log_options={"fast_logging": True, "if_database_exists": "replace"},
        )

    with pytest.warns(FutureWarning, match="if_table_exists"):
        om.minimize(
            lambda x: x @ x,
            np.arange(3),
            algorithm="scipy_lbfgsb",
            logging=tmp_path / "log_1.db",
            log_options={"fast_logging": True, "if_table_exists": "replace"},
        )


def test_log_options_are_deprecated_in_maximize(tmp_path):
    with pytest.warns(FutureWarning, match="LogOptions"):
        om.maximize(
            lambda x: -x @ x,
            np.arange(3),
            algorithm="scipy_lbfgsb",
            logging=tmp_path / "log.db",
            log_options={"fast_logging": True, "if_database_exists": "replace"},
        )

    with pytest.warns(FutureWarning, match="if_table_exists"):
        om.maximize(
            lambda x: -x @ x,
            np.arange(3),
            algorithm="scipy_lbfgsb",
            logging=tmp_path / "log_1.db",
            log_options={"fast_logging": True, "if_table_exists": "replace"},
        )


def test_dict_constraints_are_deprecated_in_minimize():
    msg = "Specifying constraints as a dictionary is deprecated and"
    with pytest.warns(FutureWarning, match=msg):
        om.minimize(
            lambda x: x @ x,
            np.arange(3),
            algorithm="scipy_lbfgsb",
            constraints={"type": "fixed", "loc": [0, 1]},
        )


def test_dict_constraints_are_deprecated_in_maximize():
    msg = "Specifying constraints as a dictionary is deprecated and"
    with pytest.warns(FutureWarning, match=msg):
        om.maximize(
            lambda x: -x @ x,
            np.arange(3),
            algorithm="scipy_lbfgsb",
            constraints={"type": "fixed", "loc": [0, 1]},
        )


def test_dict_constraints_are_deprecated_in_estimate_ml():
    msg = "Specifying constraints as a dictionary is deprecated and"
    with pytest.warns(FutureWarning, match=msg):

        @om.mark.likelihood
        def loglike(x):
            return -(x**2)

        em.estimate_ml(
            loglike=loglike,
            params=np.arange(3),
            optimize_options={"algorithm": "scipy_lbfgsb"},
            constraints={"type": "fixed", "loc": [0, 1]},
        )


def test_dict_constraints_are_deprecated_in_estimate_msm():
    msg = "Specifying constraints as a dictionary is deprecated and"
    with pytest.warns(FutureWarning, match=msg):
        em.estimate_msm(
            simulate_moments=lambda x: x,
            empirical_moments=np.zeros(3),
            moments_cov=np.eye(3),
            params=np.arange(3),
            optimize_options={"algorithm": "scipy_lbfgsb"},
            constraints={"type": "fixed", "loc": [0, 1]},
        )


@pytest.fixture
def dummy_func():
    return lambda x: x


def test_pre_process_constraints_trivial_case(dummy_func):
    constraints = om.FixedConstraint(selector=dummy_func)
    expected = [{"type": "fixed", "selector": dummy_func}]
    assert pre_process_constraints(constraints) == expected


def test_pre_process_constraints_list_of_constraints(dummy_func):
    constraints = [
        om.FixedConstraint(selector=dummy_func),
        om.IncreasingConstraint(selector=dummy_func),
    ]
    expected = [
        {"type": "fixed", "selector": dummy_func},
        {"type": "increasing", "selector": dummy_func},
    ]
    assert pre_process_constraints(constraints) == expected


def test_pre_process_constraints_none_case():
    assert pre_process_constraints(None) == []


def test_pre_process_constraints_mixed_case(dummy_func):
    constraints = [
        om.FixedConstraint(selector=dummy_func),
        {"type": "increasing", "selector": dummy_func},
    ]
    expected = [
        {"type": "fixed", "selector": dummy_func},
        {"type": "increasing", "selector": dummy_func},
    ]
    assert pre_process_constraints(constraints) == expected


def test_pre_process_constraints_dict_case(dummy_func):
    constraints = {"type": "fixed", "selector": dummy_func}
    expected = [{"type": "fixed", "selector": dummy_func}]
    assert pre_process_constraints(constraints) == expected


def test_pre_process_constraints_invalid_case():
    constraints = "invalid"
    msg = "Invalid constraint type: <class 'str'>"
    with pytest.raises(InvalidConstraintError, match=msg):
        pre_process_constraints(constraints)


def test_pre_process_constraints_invalid_mixed_case():
    constraints = [
        {"type": "fixed", "loc": [0, 1]},
        om.FixedConstraint(),
        "invalid",
    ]
    msg = "Invalid constraint types: {<class 'str'>}"
    with pytest.raises(InvalidConstraintError, match=msg):
        pre_process_constraints(constraints)


def test_deprecated_log_reader(example_db):
    with pytest.warns(FutureWarning, match="SQLiteLogReader"):
        reader = OptimizeLogReader(example_db)
        res = reader.read_start_params()
        assert res == {"a": 1, "b": 2, "c": 3}


def test_estimagic_joblib_batch_evaluator_is_deprecated():
    msg = "estimagic.batch_evaluators.joblib_batch_evaluator has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        batch_evaluators.joblib_batch_evaluator(lambda x: x, [1, 2], n_cores=1)


def test_estimagic_process_batch_evaluator_is_deprecated():
    msg = "estimagic.batch_evaluators.process_batch_evaluator has been deprecated"
    with pytest.warns(FutureWarning, match=msg):
        batch_evaluators.process_batch_evaluator("joblib")
