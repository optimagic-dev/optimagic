import warnings
from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

import optimagic as om
from optimagic import get_benchmark_problems
from optimagic.benchmarking.benchmark_problem import (
    BenchmarkProblem,
    LeastSquaresBenchmarkFunction,
    ScalarBenchmarkFunction,
    aggregate_function_value,
)
from optimagic.benchmarking.benchmark_result import BenchmarkResult
from optimagic.benchmarking.run_benchmark import (
    _process_one_result,
    _process_optimize_options,
    _set_benchmark_defaults,
    run_benchmark,
)
from optimagic.optimization.optimize_options import OptimizeOptions
from optimagic.optimization.optimize_result import OptimizeResult
from optimagic.typing import AggregationLevel


def _get_two_problems(name="more_wild"):
    all_problems = get_benchmark_problems(name)
    return {name: all_problems[name] for name in list(all_problems)[:2]}


# ======================================================================================
# run_benchmark
# ======================================================================================


def test_run_benchmark_with_all_types_of_options():
    optimize_options = {
        "name": "scipy_lbfgsb",
        "class": om.algos.scipy_lbfgsb,
        "instance": om.algos.scipy_lbfgsb(convergence_ftol_rel=1e-10),
        "options": OptimizeOptions(
            algorithm="scipy_lbfgsb", algo_options={"stopping_maxfun": 30}
        ),
    }

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        result = run_benchmark(
            problems=_get_two_problems(),
            optimize_options=optimize_options,
            error_handling="raise",
        )

    expected_keys = {
        (problem, options)
        for problem in ["linear_full_rank_good_start", "linear_full_rank_bad_start"]
        for options in optimize_options
    }
    assert set(result) == expected_keys

    for res in result.values():
        assert isinstance(res, BenchmarkResult)
        assert isinstance(res.optimize_result, OptimizeResult)
        assert res.traceback is None
        n_evals = len(res.fun_history)
        assert res.params_history.shape == (n_evals, 9)
        assert res.time_history.shape == (n_evals,)
        assert res.batches_history.shape == (n_evals,)


def test_run_benchmark_history_starts_at_start_params():
    problems = _get_two_problems("example")
    result = run_benchmark(problems, {"nm": om.algos.scipy_neldermead})
    for (problem_name, _), res in result.items():
        problem = problems[problem_name]
        aaae(res.params_history[0], problem.start_params)
        assert np.allclose(res.fun_history[0], problem.start_fun)


def test_run_benchmark_failing():
    all_problems = get_benchmark_problems("more_wild")
    failing_name = "jennrich_sampson"
    failing = {failing_name: all_problems[failing_name]}

    with pytest.warns():
        result = run_benchmark(
            problems=failing, optimize_options={"lbfgsb": "scipy_lbfgsb"}
        )

    res = result[(failing_name, "lbfgsb")]
    assert res.optimize_result is None
    assert isinstance(res.traceback, str)
    aaae(res.params_history, failing[failing_name].start_params.reshape(1, -1))
    aaae(res.fun_history, [failing[failing_name].start_fun])
    assert res.time_history[0] == np.inf


def test_run_benchmark_noisy_scaled_problem():
    """Ported from #723, where the first function value was calculated wrongly."""
    all_problems = get_benchmark_problems("example", noise=True, scaling=True, seed=0)
    name = "rosenbrock_good_start"
    problems = {name: all_problems[name]}

    result = run_benchmark(
        problems=problems,
        optimize_options={"nm": "scipy_neldermead"},
        max_criterion_evaluations=20,
    )

    res = result[(name, "nm")]
    assert np.allclose(res.fun_history[0], 24.2)
    # the history contains function values without noise
    noise_free_fun = problems[name].noise_free_fun
    expected = [
        aggregate_function_value(noise_free_fun(p), AggregationLevel.LEAST_SQUARES)
        for p in res.params_history
    ]
    aaae(res.fun_history, expected)


def test_all_optimizers_see_the_same_noise():
    problem = get_benchmark_problems("example", noise=True, seed=5)["box_3d"]
    options = {
        "first": om.algos.scipy_neldermead,
        "second": om.algos.scipy_neldermead,
    }
    result = run_benchmark({"box_3d": problem}, options, max_criterion_evaluations=30)
    aaae(
        result[("box_3d", "first")].params_history,
        result[("box_3d", "second")].params_history,
    )


def test_run_benchmark_passes_bounds():
    class LogSum(ScalarBenchmarkFunction):
        def fun(self, x):
            return float(x @ x - np.log(x).sum())

        def bounds(self, n_params):
            return om.Bounds(lower=np.full(n_params, 1e-8))

    problem = BenchmarkProblem(function=LogSum(), start_params=np.array([3.0, 0.5]))
    result = run_benchmark(
        {"log_sum": problem}, {"lbfgsb": om.algos.scipy_lbfgsb}, error_handling="raise"
    )
    res = result[("log_sum", "lbfgsb")]
    assert (res.params_history > 0).all()
    assert res.optimize_result is not None
    aaae(res.optimize_result.params, np.sqrt([0.5, 0.5]), decimal=4)


def test_run_benchmark_uses_closed_form_derivatives():
    n_jac_calls = []

    class Residuals(LeastSquaresBenchmarkFunction):
        def fun(self, x):
            return x - 1

        def jac(self, x):
            n_jac_calls.append(1)
            return np.eye(len(x))

    problem = BenchmarkProblem(function=Residuals(), start_params=np.zeros(3))
    result = run_benchmark(
        {"residuals": problem},
        {"lbfgsb": om.algos.scipy_lbfgsb, "ls_lm": om.algos.scipy_ls_lm},
        error_handling="raise",
        batch_evaluator="threading",
    )

    assert len(n_jac_calls) > 0
    for res in result.values():
        # derivative evaluations are not part of the histories
        assert not np.isnan(res.fun_history).any()
        assert len(res.fun_history) == len(res.params_history)
        assert res.optimize_result is not None
        aaae(res.optimize_result.params, np.ones(3), decimal=5)


def test_process_one_result_without_history_raises():
    problem = get_benchmark_problems("example")["box_3d"]
    res = om.minimize(
        fun=problem.fun, params=problem.start_params, algorithm="scipy_lbfgsb"
    )
    with pytest.raises(ValueError, match="collect a history"):
        _process_one_result(replace(res, history=None), problem=problem)


# ======================================================================================
# deprecated optimize_options
# ======================================================================================


def test_list_of_algorithms_is_deprecated():
    msg = "Passing a list of algorithms as `optimize_options`"
    with pytest.warns(FutureWarning, match=msg):
        result = run_benchmark(
            problems=_get_two_problems("example"),
            optimize_options=["scipy_lbfgsb", om.algos.scipy_neldermead],
        )

    # as in previous versions, algorithm classes are named after the class
    expected_keys = {
        ("helical_valley_good_start", "scipy_lbfgsb"),
        ("rosenbrock_good_start", "scipy_lbfgsb"),
        ("helical_valley_good_start", "ScipyNelderMead"),
        ("rosenbrock_good_start", "ScipyNelderMead"),
    }
    assert set(result) == expected_keys


def test_list_of_configured_algorithms_uses_algorithm_names():
    with pytest.warns(FutureWarning):
        processed = _process_optimize_options(
            [om.algos.scipy_lbfgsb(stopping_maxfun=3)],
            max_evals=10,
            disable_convergence=True,
        )
    assert list(processed) == ["scipy_lbfgsb"]
    assert processed["scipy_lbfgsb"]["algorithm"].stopping_maxfun == 3


def test_dict_options_are_deprecated():
    optimize_options = {
        "default_lbfgsb": "scipy_lbfgsb",
        "tuned_lbfgsb": {
            "algorithm": "scipy_lbfgsb",
            "algo_options": {"convergence_ftol_rel": 1e-10},
        },
    }

    msg = "dictionaries with keyword arguments for `minimize` is deprecated"
    with pytest.warns(FutureWarning, match=msg):
        result = run_benchmark(
            problems=_get_two_problems(),
            optimize_options=optimize_options,
            error_handling="raise",
        )

    expected_keys = {
        ("linear_full_rank_good_start", "default_lbfgsb"),
        ("linear_full_rank_bad_start", "default_lbfgsb"),
        ("linear_full_rank_good_start", "tuned_lbfgsb"),
        ("linear_full_rank_bad_start", "tuned_lbfgsb"),
    }
    assert set(result) == expected_keys


def test_deprecation_messages_name_the_version_of_removal():
    with pytest.warns(FutureWarning, match="removed in optimagic version 0.7.0"):
        _process_optimize_options(
            ["scipy_lbfgsb"], max_evals=10, disable_convergence=True
        )
    with pytest.warns(FutureWarning, match="removed in optimagic version 0.7.0"):
        _process_optimize_options(
            {"a": {"algorithm": "scipy_lbfgsb"}},
            max_evals=10,
            disable_convergence=True,
        )


# ======================================================================================
# processing of optimize_options
# ======================================================================================


def test_process_optimize_options_returns_configured_algorithms():
    processed = _process_optimize_options(
        {"a": "scipy_lbfgsb", "b": om.algos.scipy_neldermead},
        max_evals=50,
        disable_convergence=True,
    )
    assert isinstance(processed["a"]["algorithm"], om.algos.scipy_lbfgsb)
    assert isinstance(processed["b"]["algorithm"], om.algos.scipy_neldermead)
    assert processed["a"]["algorithm"].stopping_maxfun == 50
    assert processed["a"]["algorithm"].convergence_ftol_rel == 1e-14


def test_process_optimize_options_passes_on_optimize_options():
    options = OptimizeOptions(
        algorithm=om.algos.scipy_lbfgsb,
        algo_options={"stopping_maxfun": 7},
        numdiff_options=om.NumdiffOptions(method="central"),
        scaling=True,
    )
    processed = _process_optimize_options(
        {"a": options}, max_evals=50, disable_convergence=False
    )["a"]
    assert processed["algo_options"] == {"stopping_maxfun": 7}
    assert processed["numdiff_options"] == om.NumdiffOptions(method="central")
    assert processed["scaling"] is True
    assert processed["multistart"] is False


def test_process_optimize_options_with_deprecated_dict_keeps_other_entries():
    with pytest.warns(FutureWarning):
        processed = _process_optimize_options(
            {"a": {"algorithm": "scipy_lbfgsb", "scaling": True, "skip_checks": True}},
            max_evals=50,
            disable_convergence=False,
        )["a"]
    assert processed["scaling"] is True
    assert processed["skip_checks"] is True


def test_dict_options_without_algorithm_raise():
    with pytest.warns(FutureWarning):
        with pytest.raises(ValueError, match="do not contain an algorithm"):
            _process_optimize_options(
                {"a": {"algo_options": {}}}, max_evals=50, disable_convergence=False
            )


def test_invalid_algorithm_name_raises():
    with pytest.raises(ValueError, match="Invalid algorithm: scipy_bla"):
        _process_optimize_options(
            {"a": "scipy_bla"}, max_evals=50, disable_convergence=False
        )


def test_invalid_type_of_options_raises():
    with pytest.raises(TypeError, match="algorithm must be the name"):
        _process_optimize_options(
            {"a": 1},  # ty: ignore[invalid-argument-type]
            max_evals=50,
            disable_convergence=False,
        )


def test_benchmark_defaults_are_set():
    algorithm = _set_benchmark_defaults(
        om.algos.scipy_lbfgsb(), max_evals=20, disable_convergence=True
    )
    assert algorithm.stopping_maxfun == 20
    assert algorithm.stopping_maxiter == 20
    assert algorithm.convergence_ftol_rel == 1e-14

    algorithm = _set_benchmark_defaults(
        om.algos.scipy_trust_constr(), max_evals=20, disable_convergence=True
    )
    assert algorithm.convergence_xtol_rel == 1e-14


def test_benchmark_defaults_can_be_switched_off():
    default = om.algos.scipy_lbfgsb()
    algorithm = _set_benchmark_defaults(
        default, max_evals=None, disable_convergence=False
    )
    assert algorithm == default


def test_benchmark_defaults_do_not_overwrite_user_options():
    configured = om.algos.scipy_lbfgsb(stopping_maxfun=5, convergence_ftol_rel=1e-3)
    algorithm = _set_benchmark_defaults(
        configured, max_evals=20, disable_convergence=True
    )
    assert algorithm.stopping_maxfun == 5
    assert algorithm.convergence_ftol_rel == 1e-3
    assert algorithm.stopping_maxiter == 20


def test_benchmark_defaults_skip_options_an_algorithm_does_not_have():
    # scipy_neldermead has no gradient based convergence criterion
    algorithm = _set_benchmark_defaults(
        om.algos.scipy_neldermead(), max_evals=20, disable_convergence=True
    )
    assert algorithm.stopping_maxfun == 20
    assert not hasattr(algorithm, "convergence_gtol_rel")


def test_configured_algorithm_limits_number_of_evaluations():
    problems = _get_two_problems("example")
    result = run_benchmark(
        problems, {"nm": om.algos.scipy_neldermead(stopping_maxfun=15)}
    )
    for res in result.values():
        assert len(res.fun_history) <= 16


# ======================================================================================
# _process_one_result
# ======================================================================================


def test_process_one_result_with_traceback():
    problem = get_benchmark_problems("example")["box_3d"]
    res = _process_one_result("some traceback", problem=problem)
    assert res.traceback == "some traceback"
    assert res.optimize_result is None
    assert res.batches_history.tolist() == [0]
