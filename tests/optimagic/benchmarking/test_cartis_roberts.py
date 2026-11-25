import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from optimagic.benchmarking.benchmark_problem import BenchmarkProblem
from optimagic.benchmarking.cartis_roberts import (
    CARTIS_ROBERTS_PROBLEMS,
    get_start_points_bdvalues,
    get_start_points_msqrta,
)
from optimagic.parameters.bounds import Bounds
from optimagic.typing import AggregationLevel

# Sum of squared residuals at the start parameters, taken from the literature. They
# are used to check the implementation of the benchmark functions.
START_FUN = {
    "arglale": 700,
    "arglble": 5.460944e14,
    "argtrig": 32.99641,
    "artif": 36.59115,
    "arwhdne": 495,
    "bdvalues": 1.943417e7,
    "bratu_2d": 0.1560738,
    "bratu_2d_t": 0.4521311,
    "bratu_3d": 4.888529,
    "brownale": 2.524757e5,
    "broydn_3d": 111,
    "cbratu_2d": 0.4822531,
    "broydn_bd": 2404,
    "chandheq": 6.923365,
    "chemrcta": 3.0935,
    "chemrctb": 1.446513,
    "chnrsbne": 7635.84,
    "drcavty1": 0.4513889,
    "drcavty2": 0.4513889,
    "drcavty3": 0.4513889,
    "eigena": 285,
    "eigenb": 19,
    "flosp2hh": 519,
    "flosp2hl": 519,
    "flosp2hm": 519,
    "flosp2th": 516,
    "flosp2tl": 516,
    "flosp2tm": 516,
    "freurone": 9.95565e4,
    "hatfldg": 27,
    "hydcar20": 1341.663,
    "hydcar6": 704.1073,
    "integreq": 0.5730503,
    "luksan11": 626.0640,
    "luksan12": 3.2160e4,
    "luksan13": 6.4352e4,
    "luksan14": 2.6880e4,
    "luksan15": 2.701585e4,
    "luksan16": 1.306848e4,
    "luksan17": 1.687370e6,
    "luksan21": 99.98751,
    "luksan22": 2.487686e4,
    "methanb8": 1.043105,
    "methanl8": 4345.100,
    "morebvne": 3.633100e-4,
    "msqrta": 212.7162,
    "msqrtb": 205.0753,
    "oscigrne": 6.120720e8,
    "penalty_1": 1.144806e11,
    "penalty_2": 1.591383e6,
    "powellse": 41875,
    "qr3d": 1.2,
    "qr3dbd": 1.2,
    "spmsqrt": 74.33542,
    "semicn2u": 2.025037e4,
    "semicon2": 2.025037e4,
    "vardimne": 1.310584e14,
    "watsonne": 30,
    "yatpsq_1": 2.073643e6,
    "yatpsq_2": 1.831687e5,
}


def test_start_values_are_known_for_all_problems():
    assert set(START_FUN) == set(CARTIS_ROBERTS_PROBLEMS)


@pytest.mark.parametrize("name, problem", list(CARTIS_ROBERTS_PROBLEMS.items()))
def test_cartis_roberts_function_at_start_x(name, problem):
    assert isinstance(problem, BenchmarkProblem)
    assert np.allclose(problem.start_fun, START_FUN[name])


@pytest.mark.parametrize("name, problem", list(CARTIS_ROBERTS_PROBLEMS.items()))
def test_cartis_roberts_function_at_solution_x(name, problem):  # noqa: ARG001
    solution = problem.solution
    if solution.params is not None:
        residuals = problem.function.fun(solution.params)
        assert np.allclose(residuals @ residuals, solution.fun, atol=1e-7)


@pytest.mark.parametrize("name, problem", list(CARTIS_ROBERTS_PROBLEMS.items()))
def test_cartis_roberts_function_is_least_squares_function(name, problem):  # noqa: ARG001
    """Ported from #723, where functions defined via partial were scalar functions."""
    assert problem.function.problem_type == AggregationLevel.LEAST_SQUARES
    assert problem.function.fun(problem.start_params).ndim == 1


def test_problems_with_the_same_function_and_parameters_have_equal_functions():
    problems = CARTIS_ROBERTS_PROBLEMS
    assert problems["methanb8"].function == problems["methanl8"].function
    assert problems["flosp2hh"].function != problems["flosp2hl"].function
    assert hash(problems["flosp2hh"].function) == hash(problems["flosp2hh"].function)


def test_bounds_of_original_problems_are_only_stored_in_info():
    with_bounds = {
        name for name, problem in CARTIS_ROBERTS_PROBLEMS.items() if problem.info
    }
    assert with_bounds == {
        "chemrcta",
        "chemrctb",
        "eigena",
        "qr3d",
        "qr3dbd",
        "semicon2",
    }
    for name in with_bounds:
        problem = CARTIS_ROBERTS_PROBLEMS[name]
        assert isinstance(problem.info["bounds"], Bounds)
        assert problem.bounds is None


def test_get_start_points_bdvalues():
    expected = np.array([-0.1389, -0.2222, -0.2500, -0.2222, -0.1389])
    result = get_start_points_bdvalues(5)
    assert_array_almost_equal(expected, result, decimal=4)


def test_get_start_points_msqrta():
    matlab_mat = np.array(
        [
            [0.8415, -0.7568, 0.4121, -0.2879, -0.1324],
            [-0.9918, -0.9538, 0.9200, -0.6299, -0.5064],
            [0.9988, -0.4910, -0.6020, 0.9395, -0.9301],
            [-0.9992, -0.0265, -0.4041, 0.2794, -0.8509],
            [0.9235, 0.1935, 0.9365, -0.8860, 0.1760],
        ]
    )
    expected = 0.2 * matlab_mat.flatten()
    result = get_start_points_msqrta(5)
    assert_array_almost_equal(result, expected, decimal=4)
