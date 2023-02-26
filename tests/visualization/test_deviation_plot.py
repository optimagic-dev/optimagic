import pytest
from estimagic import get_benchmark_problems
from estimagic.benchmarking.run_benchmark import run_benchmark
from estimagic.visualization.deviation_plot import (
    deviation_plot,
)

# integration test to make sure non default argument do not throw Errors
profile_options = [
    {"distance_measure": "parameter_distance"},
    {"distance_measure": "criterion"},
    {"monotone": True},
    {"monotone": False},
    {"runtime_measure": "walltime"},
    {"stopping_criterion": None},
    {"stopping_criterion": "x"},
    {"stopping_criterion": "x_and_y"},
    {"stopping_criterion": "x_or_y"},
    {"x_precision": 1e-5},
    {"y_precision": 1e-5},
]


@pytest.mark.parametrize("options", profile_options)
def test_convergence_plot_options(options):
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

    deviation_plot(problems=problems, results=results, **options)
