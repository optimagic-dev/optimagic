import numpy as np
import pytest

from optimagic.benchmarking.benchmark_problem import BenchmarkProblem
from optimagic.benchmarking.more_wild import (
    MORE_WILD_PROBLEMS,
    Chebyquad,
    HelicalValley,
    LinearFullRank,
    Watson,
    get_start_points_mancino,
)
from optimagic.typing import AggregationLevel

# Sum of squared residuals at the start parameters, taken from the literature. They
# are used to check the implementation of the benchmark functions.
START_FUN = {
    "linear_full_rank_good_start": 72,
    "linear_full_rank_bad_start": 1125,
    "linear_rank_one_good_start": 1.165420e7,
    "linear_rank_one_bad_start": 1.168591e9,
    "linear_rank_one_zero_columns_rows_good_start": 4.989195e6,
    "linear_rank_one_zero_columns_rows_bad_start": 5.009356e8,
    "rosenbrock_good_start": 24.2,
    "rosenbrock_bad_start": 1.795769e6,
    "helical_valley_good_start": 2500,
    "helical_valley_bad_start": 10600,
    "powell_singular_good_start": 215,
    "powell_singular_bad_start": 1.615400e6,
    "freudenstein_roth_good_start": 400.5,
    "freudenstein_roth_bad_start": 1.545754e8,
    "bard_good_start": 41.68170,
    "bard_bad_start": 1306.234,
    "kowalik_osborne": 5.313172e-3,
    "meyer": 1.693608e9,
    "watson_6_good_start": 16.43083,
    "watson_6_bad_start": 2.323367e6,
    "watson_9_good_start": 26.90417,
    "watson_9_bad_start": 8.158877e6,
    "watson_12_good_start": 73.67821,
    "watson_12_bad_start": 2.059384e7,
    "box_3d": 1031.154,
    "jennrich_sampson": 4171.306,
    "brown_dennis_good_start": 7.926693e6,
    "brown_dennis_bad_start": 3.081064e11,
    "chebyquad_6": 4.642817e-2,
    "chebyquad_7": 3.377064e-2,
    "chebyquad_8": 3.861770e-2,
    "chebyquad_9": 2.888298e-2,
    "chebyquad_10": 3.376327e-2,
    "chebyquad_11": 2.674060e-2,
    "brown_almost_linear": 273.2480,
    "osborne_one": 16.17411,
    "osborne_two_good_start": 2.093420,
    "osborne_two_bad_start": 199.6847,
    "bdqrtic_8": 904,
    "bdqrtic_10": 1356,
    "bdqrtic_11": 1582,
    "bdqrtic_12": 1808,
    "cube_5": 56.5,
    "cube_6": 70.5625,
    "cube_8": 98.6875,
    "mancino_5_good_start": 2.539084e9,
    "mancino_5_bad_start": 6.873795e12,
    "mancino_8": 3.367961e9,
    "mancino_10": 3.735127e9,
    "mancino_12_good_start": 3.991072e9,
    "mancino_12_bad_start": 1.130015e13,
    "heart_eight_good_start": 9.385672,
    "heart_eight_bad_start": 3.365815e10,
    "brown_almost_linear_medium": 2.524757e5,
}


def test_start_values_are_known_for_all_problems():
    assert set(START_FUN) == set(MORE_WILD_PROBLEMS)


@pytest.mark.parametrize("name, problem", list(MORE_WILD_PROBLEMS.items()))
def test_more_wild_function_at_start_x(name, problem):
    assert isinstance(problem, BenchmarkProblem)
    assert np.allclose(problem.start_fun, START_FUN[name])

    solution = problem.solution
    if solution.params is not None:
        residuals = problem.function.fun(solution.params)
        assert np.allclose(residuals @ residuals, solution.fun, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("name, problem", list(MORE_WILD_PROBLEMS.items()))
def test_more_wild_function_is_least_squares_function(name, problem):  # noqa: ARG001
    assert problem.function.problem_type == AggregationLevel.LEAST_SQUARES
    assert problem.function.fun(problem.start_params).ndim == 1


@pytest.mark.parametrize("name, problem", list(MORE_WILD_PROBLEMS.items()))
def test_more_wild_solutions_are_exact_if_zero(name, problem):  # noqa: ARG001
    assert problem.solution.exact == (problem.solution.fun == 0)


def test_same_function_is_used_for_good_and_bad_start():
    good = MORE_WILD_PROBLEMS["linear_full_rank_good_start"]
    bad = MORE_WILD_PROBLEMS["linear_full_rank_bad_start"]
    assert good.function == bad.function == LinearFullRank(dim_out=45)
    assert not np.allclose(good.start_params, bad.start_params)


def test_supported_numbers_of_params():
    assert HelicalValley().supports_n_params(3)
    assert not HelicalValley().supports_n_params(4)

    assert Watson().supports_n_params(2)
    assert Watson().supports_n_params(31)
    assert not Watson().supports_n_params(32)

    assert LinearFullRank(dim_out=5).supports_n_params(5)
    assert not LinearFullRank(dim_out=5).supports_n_params(6)

    assert Chebyquad(dim_out=8).supports_n_params(8)
    assert not Chebyquad(dim_out=8).supports_n_params(9)


def test_get_start_points_mancino():
    expected = (np.array([102.4824, 96.3335, 90.4363, 84.7852, 79.3747]),)
    result = get_start_points_mancino(5)
    assert np.allclose(expected, result)
