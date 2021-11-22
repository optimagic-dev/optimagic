from estimagic import get_benchmark_problems
from estimagic.benchmarking.run_benchmark import run_benchmark


def test_run_benchmark_dict_options(tmpdir):
    all_problems = get_benchmark_problems("more_wild")
    first_two_names = list(all_problems)[:2]
    first_two = {name: all_problems[name] for name in first_two_names}

    optimize_options = {
        "default_lbfgsb": "scipy_lbfgsb",
        "tuned_lbfgsb": {
            "algorithm": "scipy_lbfgsb",
            "algo_options": {"convergence.relative_criterion_tolerance": 1e-10},
        },
    }

    logging_directory = tmpdir / "benchmark_logs"

    res = run_benchmark(
        problems=first_two,
        optimize_options=optimize_options,
        logging_directory=logging_directory,
    )

    expected_keys = {
        ("linear_full_rank_good_start", "default_lbfgsb"),
        ("linear_full_rank_bad_start", "default_lbfgsb"),
        ("linear_full_rank_good_start", "tuned_lbfgsb"),
        ("linear_full_rank_bad_start", "tuned_lbfgsb"),
    }

    assert set(res) == expected_keys


def test_run_benchmark_list_options(tmpdir):
    all_problems = get_benchmark_problems("example")
    first_two_names = list(all_problems)[:2]
    first_two = {name: all_problems[name] for name in first_two_names}

    optimize_options = ["scipy_lbfgsb", "scipy_neldermead"]

    logging_directory = tmpdir / "benchmark_logs"

    res = run_benchmark(
        problems=first_two,
        optimize_options=optimize_options,
        logging_directory=logging_directory,
    )

    expected_keys = {
        ("linear_full_rank_good_start", "scipy_lbfgsb"),
        ("rosenbrock_good_start", "scipy_lbfgsb"),
        ("linear_full_rank_good_start", "scipy_neldermead"),
        ("rosenbrock_good_start", "scipy_neldermead"),
    }

    assert set(res) == expected_keys
