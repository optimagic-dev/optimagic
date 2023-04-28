import pytest

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


def test_convergence_report(benchmark_example):
    optimizers, problems, results = benchmark_example

    df = convergence_report(problems=problems, results=results)

    expected_columns = list(optimizers.keys()) + ["dimensionality"]
    assert df.shape == (len(problems), len(expected_columns))
    assert set(df.columns) == set(expected_columns)


def test_rank_report(benchmark_example):
    optimizers, problems, results = benchmark_example

    df = rank_report(problems=problems, results=results)

    assert df.shape == (len(problems), len(optimizers))
    assert set(df.columns) == set(optimizers.keys())


def test_traceback_report(benchmark_example):
    *_, results = benchmark_example

    traceback_report(results=results)
