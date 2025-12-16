import pytest

from optimagic import get_benchmark_problems
from optimagic.benchmarking.process_benchmark_results import process_benchmark_results
from optimagic.benchmarking.run_benchmark import run_benchmark
from optimagic.visualization.convergence_plot import (
    _check_only_allowed_subset_provided,
    _extract_convergence_plot_lines,
    convergence_plot,
)


@pytest.fixture()
def benchmark_results():
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
    return problems, results


def test_convergence_plot_default_options(benchmark_results):
    problems, results = benchmark_results

    convergence_plot(
        problems=problems,
        results=results,
        problem_subset=["bard_good_start"],
    )


# integration test to make sure non default argument do not throw Errors
profile_options = [
    {"n_cols": 3},
    {"distance_measure": "parameter_distance"},
    {"monotone": False},
    {"normalize_distance": False},
    {"runtime_measure": "walltime"},
    {"runtime_measure": "n_batches"},
    {"stopping_criterion": "x"},
    {"stopping_criterion": "x_and_y"},
    {"stopping_criterion": "x_or_y"},
    {"x_precision": 1e-5},
    {"y_precision": 1e-5},
    {"backend": "matplotlib"},
    {"backend": "bokeh"},
    {"backend": "altair"},
]


@pytest.mark.parametrize("options", profile_options)
@pytest.mark.parametrize("grid", [True, False])
def test_convergence_plot_options(options, grid, benchmark_results, close_mpl_figures):
    problems, results = benchmark_results

    convergence_plot(
        problems=problems,
        results=results,
        problem_subset=["bard_good_start"],
        combine_plots_in_grid=grid,
        **options,
    )


def test_convergence_plot_stopping_criterion_none(benchmark_results):
    problems, results = benchmark_results

    fig = convergence_plot(
        problems=problems,
        results=results,
        problem_subset=["bard_good_start"],
        stopping_criterion=None,
    )

    assert fig is not None


def test_check_only_allowed_subset_provided_none():
    allowed = ["a", "b", "c"]
    _check_only_allowed_subset_provided(None, allowed, "name")


def test_check_only_allowed_subset_provided_all_included():
    allowed = ["a", "b", "c"]
    _check_only_allowed_subset_provided(["a", "b"], allowed, "name")


def test_check_only_allowed_subset_provided_missing():
    allowed = ["a", "b", "c"]
    with pytest.raises(ValueError):
        _check_only_allowed_subset_provided(["d"], allowed, "name")


def test_extract_convergence_plot_lines(benchmark_results):
    problems, results = benchmark_results

    df, _ = process_benchmark_results(
        problems=problems, results=results, stopping_criterion="y"
    )

    lines_list, titles = _extract_convergence_plot_lines(
        df=df,
        problems=problems,
        runtime_measure="n_evaluations",
        outcome="criterion_normalized",
        palette=["red", "green", "blue"],
        combine_plots_in_grid=True,
        backend="bla",
    )

    assert isinstance(lines_list, list) and isinstance(titles, list)
    assert len(lines_list) == len(titles) == len(problems)

    for subplot_lines in lines_list:
        assert isinstance(subplot_lines, list) and len(subplot_lines) == 2
        assert subplot_lines[0].name == "lbfgsb"
        assert subplot_lines[1].name == "nm"
        assert subplot_lines[0].color == "red"
        assert subplot_lines[1].color == "green"
