"""Define the More-Wild Benchmark Set.

This benchmark set contains 78+ test cases for single objective optimization.
The test cases are built out of 78 functions, which originate from the python
implementation of benchmark test functions for single objective optimization by
Axel Thevenot.

We use the following sources of information to construct the
benchmark set:

- https://towardsdatascience.com/optimization-eye-pleasure-78-benchmark-test-functions-for-single-objective-optimization-92e7ed1d1f12
- https://github.com/AxelThevenot/Python_Benchmark_Test_Optimization_Function_Single_Objective
"""
import numpy as np


def ackley(x, a=20, b=0.2, c=2 * np.pi):
    d = x.shape[0]
    assert (d is None) or (
        isinstance(d, int) and (not d < 0)
    ), "The dimension d must be None or a positive integer"
    res = -a * np.exp(-b * np.sqrt(np.mean(x ** 2)))
    out = res - np.exp(np.mean(np.cos(c * x))) + a + np.exp(1)
    return out


def rosenbrock2(x, a=1, b=100):
    d = x.shape[0]
    assert (d is None) or (
        isinstance(d, int) and (not d < 0)
    ), "The dimension d must be None or a positive integer"
    res = np.sum(np.abs(b * (x[1:] - x[:-1] ** 2) ** 2 + (a - x[:-1]) ** 2))
    return res


SCALAR_FUNCTION_PROBLEMS = {
    "ackley_good_start": {
        "criterion": ackley,
        "start_x": np.full(10, 3),
        "solution_x": np.zeros(10),
        "start_criterion": 9.023767278119472,
        "solution_criterion": 0,
    },
    "ackley_bad_start": {
        "criterion": ackley,
        "start_x": np.full(10, 3) * 10,
        "solution_x": np.zeros(10),
        "start_criterion": 19.950424956466673,
        "solution_criterion": 0,
    },
    "rosenbrock_good_start": {
        "criterion": rosenbrock2,
        "start_x": np.array([-1.2, 1]),
        "solution_x": np.ones(2),
        "start_criterion": 24.2,
        "solution_criterion": 0,
    },
}
