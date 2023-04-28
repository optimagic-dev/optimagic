import pytest
from itertools import product

from estimagic import get_benchmark_problems
from estimagic.benchmarking.run_benchmark import run_benchmark
from estimagic.benchmarking.benchmark_reports import (
    convergence_report,
    rank_report,
    traceback_report,
)


@pytest.fixture
def benchmark_example():
    problems = get_benchmark_problems("example")
    stop_after_10 = {
        "stopping_max_criterion_evaluations": 10,
        "stopping_max_iterations": 10,
    }
    optimizers = {
        "lbfgsb": {"algorithm": "scipy_lbfgsb", "algo_options": stop_after_10},
        "nm": {"algorithm": "scipy_neldermead", "algo_options": stop_after_10},
    }
    results = run_benchmark(
        problems,
        optimizers,
        n_cores=1,  # must be 1 for the test to work
    )

    return optimizers, problems, results


# ====================================================================================
# Convergence report
# ====================================================================================

keys = ["stopping_criterion"]
stopping_criterion = ["x_and_y", "x_or_y", "x", "y"]
x_precision = [1e-4, 1e-6]
y_precision = [1e-4, 1e-6]
CONVERGENCE_REPORT_OPTIONS = [
    dict(zip(keys, value))
    for value in product(stopping_criterion, x_precision, y_precision)
]


@pytest.mark.parametrize("options", CONVERGENCE_REPORT_OPTIONS)
def test_convergence_report(options, benchmark_example):
    optimizers, problems, results = benchmark_example

    df = convergence_report(problems=problems, results=results, **options)

    expected_columns = list(optimizers.keys()) + ["dimensionality"]
    assert df.shape == (len(problems), len(expected_columns))
    assert set(df.columns) == set(expected_columns)


def test_convergence_report_with_failed_and_error(benchmark_example):
    _, problems, results = benchmark_example
    failed_problem = ("bard_good_start", "nm")
    error_problem = ("box_3d", "nm")
    results[error_problem]["solution"] = "some traceback"

    df = convergence_report(problems=problems, results=results)

    assert df[failed_problem[1]].loc[failed_problem[0]] == "failed"
    assert df[error_problem[1]].loc[error_problem[0]] == "error"


# ====================================================================================
# Rank report
# ====================================================================================

keys = ["runtime_measure", "normalize_runtime", "stopping_criterion"]
runtime_measure = ["n_evaluations", "walltime", "n_batches"]
y_precision = [True, False]
RANK_REPORT_OPTIONS = [
    dict(zip(keys, value))
    for value in product(runtime_measure, y_precision, stopping_criterion)
]


@pytest.mark.parametrize("options", RANK_REPORT_OPTIONS)
def test_rank_report(options, benchmark_example):
    optimizers, problems, results = benchmark_example

    df = rank_report(problems=problems, results=results, **options)

    assert df.shape == (len(problems), len(optimizers))
    assert set(df.columns) == set(optimizers.keys())


def test_rank_report_with_failed_and_error(benchmark_example):
    _, problems, results = benchmark_example
    failed_problem = ("bard_good_start", "nm")
    error_problem = ("box_3d", "nm")
    results[error_problem]["solution"] = "some traceback"

    df = rank_report(problems=problems, results=results)

    assert df[failed_problem[1]].loc[failed_problem[0]] == "failed"
    assert df[error_problem[1]].loc[error_problem[0]] == "error"


# ====================================================================================
# Traceback report
# ====================================================================================


def test_traceback_report(benchmark_example):
    *_, results = benchmark_example
    traceback_report(results=results)


def test_traceback_report_with_error(benchmark_example):
    *_, results = benchmark_example
    results[("box_3d", "nm")]["solution"] = "some traceback"

    df = traceback_report(results=results)

    assert df.shape[0] > 0
