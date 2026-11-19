import pytest

from optimagic import algos, get_benchmark_problems
from optimagic.benchmarking.run_benchmark import run_benchmark
from optimagic.visualization.deviation_plot import (
    deviation_plot,
)

# integration test to make sure non default argument do not throw Errors
profile_options = [
    {"distance_measure": "parameter_distance"},
    {"distance_measure": "criterion"},
    {"monotone": True},
    {"monotone": False},
    {"runtime_measure": "n_evaluations"},
    {"runtime_measure": "n_batches"},
]


@pytest.mark.parametrize("options", profile_options)
def test_convergence_plot_options(options):
    problems = get_benchmark_problems("example")
    optimizers = {"lbfgsb": algos.scipy_lbfgsb, "nm": algos.scipy_neldermead}
    results = run_benchmark(
        problems,
        optimizers,
        n_cores=1,  # must be 1 for the test to work
    )

    deviation_plot(problems=problems, results=results, **options)
