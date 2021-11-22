from itertools import product

import numpy as np
import pytest
from estimagic.examples.benchmarking import get_problems
from estimagic.examples.benchmarking import run_benchmark

PARMETRIZATION = []
for name in ["more_wild", "cartis_roberts"]:
    for additive, multiplicative in product([False, True], repeat=2):
        PARMETRIZATION.append((name, additive, multiplicative))


@pytest.mark.parametrize("name, additive_noise, multiplicative_noise", PARMETRIZATION)
def test_get_problems(name, additive_noise, multiplicative_noise):
    is_noisy = any((additive_noise, multiplicative_noise))
    problems = get_problems(
        name=name,
        additive_noise=additive_noise,
        multiplicative_noise=multiplicative_noise,
    )
    first_name = list(problems)[0]
    first = problems[first_name]
    func = first["inputs"]["criterion"]
    params = first["inputs"]["params"]

    np.random.seed()
    first_eval = func(params)["value"]
    second_eval = func(params)["value"]

    if is_noisy:
        assert first_eval != second_eval
    else:
        assert first_eval == second_eval


def test_run_benchmark_dict_options(tmpdir):
    all_problems = get_problems("more_wild")
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

    expetced_keys = {
        ("linear_full_rank_good_start", "default_lbfgsb"),
        ("linear_full_rank_bad_start", "default_lbfgsb"),
        ("linear_full_rank_good_start", "tuned_lbfgsb"),
        ("linear_full_rank_bad_start", "tuned_lbfgsb"),
    }

    assert set(res) == expetced_keys


def test_run_benchmark_list_options(tmpdir):
    all_problems = get_problems("more_wild")
    first_two_names = list(all_problems)[:2]
    first_two = {name: all_problems[name] for name in first_two_names}

    optimize_options = ["scipy_lbfgsb", "scipy_neldermead"]

    logging_directory = tmpdir / "benchmark_logs"

    res = run_benchmark(
        problems=first_two,
        optimize_options=optimize_options,
        logging_directory=logging_directory,
    )

    expetced_keys = {
        ("linear_full_rank_good_start", "scipy_lbfgsb"),
        ("linear_full_rank_bad_start", "scipy_lbfgsb"),
        ("linear_full_rank_good_start", "scipy_neldermead"),
        ("linear_full_rank_bad_start", "scipy_neldermead"),
    }

    assert set(res) == expetced_keys
