import matplotlib.pyplot as plt
import pytest
from estimagic.examples.benchmarking import get_problems
from estimagic.examples.benchmarking import run_benchmark
from estimagic.visualization.convergence_plot import convergence_plot

# integration test to make sure non default argument do not throw Errors
profile_options = [
    {"n_cols": 3},
    {"distance_measure": "parameter_distance"},
    {"monotone": False},
    {"normalize_distance": False},
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
    full_problems = get_problems("more_wild")
    stop_after_100 = {
        "stopping_max_criterion_evaluations": 100,
        "stopping_max_iterations": 100,
    }
    optimizers = {
        "lbfgsb": {"algorithm": "scipy_lbfgsb", "algo_options": stop_after_100},
        "neldermead": {
            "algorithm": "scipy_neldermead",
            "algo_options": stop_after_100,
        },
    }
    problems = {
        k: v for k, v in full_problems.items() if k in sorted(full_problems)[:5]
    }

    results = run_benchmark(
        problems,
        optimizers,
        n_cores=1,  # must be 1 for the test to work
        logging_directory="logging",
    )

    convergence_plot(problems=problems, results=results, **options)
    plt.close()
