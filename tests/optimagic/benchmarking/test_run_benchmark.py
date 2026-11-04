import numpy as np
import pytest

from optimagic import get_benchmark_problems
from optimagic.benchmarking.run_benchmark import run_benchmark


def test_run_benchmark_dict_options():
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

    result = run_benchmark(
        problems=first_two,
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


def test_run_benchmark_list_options():
    all_problems = get_benchmark_problems("example")
    first_two_names = list(all_problems)[:2]
    first_two = {name: all_problems[name] for name in first_two_names}
    optimize_options = ["scipy_lbfgsb", "scipy_neldermead"]

    result = run_benchmark(
        problems=first_two,
        optimize_options=optimize_options,
    )

    expected_keys = {
        ("helical_valley_good_start", "scipy_lbfgsb"),
        ("rosenbrock_good_start", "scipy_lbfgsb"),
        ("helical_valley_good_start", "scipy_neldermead"),
        ("rosenbrock_good_start", "scipy_neldermead"),
    }
    assert set(result) == expected_keys


def test_run_benchmark_failing():
    all_problems = get_benchmark_problems("more_wild")
    failing_name = "jennrich_sampson"
    failing = {failing_name: all_problems[failing_name]}

    optimize_options = ["scipy_lbfgsb"]

    with pytest.warns():
        result = run_benchmark(problems=failing, optimize_options=optimize_options)

    key = (failing_name, "scipy_lbfgsb")
    assert isinstance(result[key]["solution"], str)


def test_run_benchmark_likelihood_problems():
    all_problems = get_benchmark_problems("likelihood_small")
    names = ["logit_5_good_start", "garch_3_bad_start"]
    problems = {name: all_problems[name] for name in names}

    with pytest.warns():
        result = run_benchmark(
            problems=problems,
            optimize_options=["scipy_lbfgsb", "scipy_ls_trf"],
            max_criterion_evaluations=50,
        )

    for name, problem in problems.items():
        # least-squares optimizers fail on likelihood problems; the start value of the
        # failed run is the sum of the likelihood contributions
        failed = result[(name, "scipy_ls_trf")]
        assert isinstance(failed["solution"], str)
        np.testing.assert_allclose(
            failed["criterion_history"][0], problem["start_criterion"]
        )

        history = result[(name, "scipy_lbfgsb")]["criterion_history"]
        np.testing.assert_allclose(history[0], problem["start_criterion"])
        assert history[-1] < problem["start_criterion"]


def test_run_benchmark_noisy_likelihood_problem():
    all_problems = get_benchmark_problems("likelihood_small", additive_noise=True)
    name = "poisson_5_good_start"
    problems = {name: all_problems[name]}

    result = run_benchmark(
        problems=problems,
        optimize_options=["scipy_neldermead"],
        max_criterion_evaluations=20,
    )

    history = result[(name, "scipy_neldermead")]["criterion_history"]
    np.testing.assert_allclose(history[0], problems[name]["start_criterion"])
