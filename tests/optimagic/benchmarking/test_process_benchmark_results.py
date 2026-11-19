import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

import optimagic as om
from optimagic.benchmarking.benchmark_problem import (
    BenchmarkFunction,
    BenchmarkProblem,
    BenchmarkSolution,
)
from optimagic.benchmarking.benchmark_result import BenchmarkResult
from optimagic.benchmarking.process_benchmark_results import (
    get_reference_solutions,
    process_benchmark_results,
)


class ShiftedSphere(BenchmarkFunction):
    """Sphere with minimum 1 at [1, 1, ...]."""

    def fun(self, x):
        return float((x - 1) @ (x - 1)) + 1.0


def _problem(solution):
    return BenchmarkProblem(
        function=ShiftedSphere(), start_params=np.array([3.0, 1.0]), solution=solution
    )


def _result(params_history, traceback=None):
    function = ShiftedSphere()
    params_history = np.array(params_history, dtype=np.float64)
    n_evals = len(params_history)
    return BenchmarkResult(
        params_history=params_history,
        fun_history=np.array([function.fun(p) for p in params_history]),
        time_history=np.arange(n_evals, dtype=np.float64),
        batches_history=np.arange(n_evals),
        optimize_result=None,
        traceback=traceback,
    )


@pytest.fixture
def results():
    return {
        # gets close to the solution
        ("sphere", "good"): _result([[3.0, 1.0], [2.0, 1.0], [1.001, 1.0]]),
        # only gets half way
        ("sphere", "bad"): _result([[3.0, 1.0], [2.5, 1.0], [2.0, 1.0]]),
    }


# ======================================================================================
# get_reference_solutions
# ======================================================================================


def test_reference_is_stored_solution_if_exact(results):
    solution = BenchmarkSolution(fun=1.0, params=np.ones(2), exact=True)
    references = get_reference_solutions({"sphere": _problem(solution)}, results)
    assert references == {"sphere": solution}


def test_reference_is_exact_solution_even_if_a_lower_value_was_found(results):
    solution = BenchmarkSolution(fun=1.5, params=np.ones(2), exact=True)
    references = get_reference_solutions({"sphere": _problem(solution)}, results)
    assert references["sphere"] is solution


def test_reference_is_best_found_value_without_stored_solution(results):
    references = get_reference_solutions({"sphere": _problem(None)}, results)
    reference = references["sphere"]
    assert np.allclose(reference.fun, 1.0 + 1e-6)
    assert reference.params is None
    assert reference.exact is False


def test_reference_is_best_known_solution_if_nothing_better_was_found(results):
    solution = BenchmarkSolution(fun=1.0, params=np.ones(2))
    references = get_reference_solutions({"sphere": _problem(solution)}, results)
    assert references["sphere"] is solution


def test_best_known_solution_is_replaced_if_something_better_was_found(results):
    solution = BenchmarkSolution(fun=1.2, params=np.array([1.4, 1.0]))
    references = get_reference_solutions({"sphere": _problem(solution)}, results)
    reference = references["sphere"]
    assert np.allclose(reference.fun, 1.0 + 1e-6)
    aaae(np.asarray(reference.params), np.array([1.001, 1.0]))
    assert reference.exact is False


def test_best_known_solution_without_params_stays_without_params(results):
    solution = BenchmarkSolution(fun=1.2, params=None)
    references = get_reference_solutions({"sphere": _problem(solution)}, results)
    assert np.allclose(references["sphere"].fun, 1.0 + 1e-6)
    assert references["sphere"].params is None


def test_references_only_contain_problems_with_results(results):
    problems = {"sphere": _problem(None), "unused": _problem(None)}
    assert set(get_reference_solutions(problems, results)) == {"sphere"}


def test_nan_function_values_are_ignored_for_references():
    result = _result([[3.0, 1.0], [2.0, 1.0]])
    fun_history = result.fun_history.copy()
    fun_history[1] = np.nan
    result = BenchmarkResult(
        params_history=result.params_history,
        fun_history=fun_history,
        time_history=result.time_history,
        batches_history=result.batches_history,
        optimize_result=None,
    )
    references = get_reference_solutions(
        {"sphere": _problem(None)}, {("sphere", "a"): result}
    )
    assert references["sphere"].fun == 5.0


# ======================================================================================
# process_benchmark_results
# ======================================================================================


def test_process_benchmark_results_with_exact_solution(results):
    solution = BenchmarkSolution(fun=1.0, params=np.ones(2), exact=True)
    df, converged = process_benchmark_results(
        {"sphere": _problem(solution)},
        results,
        stopping_criterion="y",
        y_precision=1e-4,
    )

    assert converged.loc["sphere"].to_dict() == {"good": True, "bad": False}

    good = df.query("algorithm == 'good'")
    aaae(good["criterion"], [5.0, 2.0, 1.000001])
    aaae(good["criterion_normalized"], [1.0, 0.25, 2.5e-7])
    aaae(good["parameter_distance"], [2.0, 1.0, 0.001])
    aaae(good["parameter_distance_normalized"], [1.0, 0.5, 0.0005])
    assert good["n_evaluations"].tolist() == [0, 1, 2]


def test_process_benchmark_results_without_solution(results):
    df, converged = process_benchmark_results(
        {"sphere": _problem(None)}, results, stopping_criterion="y"
    )

    # the best optimizer defines the solution and therefore always converges
    assert converged.loc["sphere"].to_dict() == {"good": True, "bad": False}

    good = df.query("algorithm == 'good'")
    aaae(good["criterion_normalized"], [1.0, 0.25, 0.0], decimal=5)
    assert good["parameter_distance"].isna().all()


@pytest.mark.parametrize("stopping_criterion", ["x", "x_and_y"])
def test_convergence_in_params_is_impossible_without_solution_params(
    results, stopping_criterion
):
    _, converged = process_benchmark_results(
        {"sphere": _problem(None)}, results, stopping_criterion=stopping_criterion
    )
    assert converged.loc["sphere"].to_dict() == {"good": False, "bad": False}


def test_x_or_y_works_without_solution_params(results):
    _, converged = process_benchmark_results(
        {"sphere": _problem(None)}, results, stopping_criterion="x_or_y"
    )
    assert converged.loc["sphere"].to_dict() == {"good": True, "bad": False}


def test_values_below_an_exact_solution_are_clipped(results):
    solution = BenchmarkSolution(fun=1.5, params=np.ones(2), exact=True)
    df, _ = process_benchmark_results(
        {"sphere": _problem(solution)}, results, stopping_criterion=None
    )
    good = df.query("algorithm == 'good'")
    aaae(good["criterion"], [5.0, 2.0, 1.5])
    assert (good["criterion_normalized"] >= 0).all()


def test_histories_are_cut_after_convergence():
    results = {
        ("sphere", "a"): _result([[3.0, 1.0], [1.0, 1.0], [1.5, 1.0], [1.0, 1.0]])
    }
    solution = BenchmarkSolution(fun=1.0, params=np.ones(2), exact=True)
    df, converged = process_benchmark_results(
        {"sphere": _problem(solution)}, results, stopping_criterion="y"
    )
    assert converged.loc["sphere", "a"]
    assert len(df) == 2


def test_without_stopping_criterion_nothing_converges(results):
    solution = BenchmarkSolution(fun=1.0, params=np.ones(2), exact=True)
    df, converged = process_benchmark_results(
        {"sphere": _problem(solution)}, results, stopping_criterion=None
    )
    assert converged.loc["sphere"].to_dict() == {"good": False, "bad": False}
    assert len(df) == 6


def test_problem_without_solution_in_plots_and_reports():
    problem = BenchmarkProblem(function=ShiftedSphere(), start_params=np.full(3, 4.0))
    problems = {"sphere": problem}
    results = om.run_benchmark(
        problems,
        {"lbfgsb": om.algos.scipy_lbfgsb, "nm": om.algos.scipy_neldermead},
        max_criterion_evaluations=100,
    )

    report = om.convergence_report(problems, results)
    assert "success" in set(report.loc["sphere"])

    om.profile_plot(problems, results)
    om.convergence_plot(problems, results, normalize_distance=False)
