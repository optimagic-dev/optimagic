"""Define the Scalar Benchmark Set.

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
from estimagic.benchmarking import more_wild as mw


def ackley(x):
    res = -20 * np.exp(-0.2 * np.sqrt(np.mean(x ** 2)))
    out = res - np.exp(np.mean(np.cos(2 * np.pi * x))) + 20 + np.exp(1)
    return out


def ackley2(x):
    x_1, x_2 = x
    out = -200 * np.exp(-0.2 * np.sqrt(x_1 ** 2 + x_2 ** 2))
    return out


def ackley3(x):
    x_1, x_2 = x
    res = -200 * np.exp(-0.2 * np.sqrt(x_1 ** 2 + x_2 ** 2))
    out = res + 5 * np.exp(np.cos(3 * x_1) + np.sin(3 * x_2))
    return out


def ackley4(x):
    x_1, x_2 = x
    out = np.sum(
        np.exp(-0.2) * np.sqrt(x_1 ** 2 + x_2 ** 2)
        + 3 * (np.cos(2 * x_1) + np.sin(2 * x_2))
    )
    return out


def adjiman(x):
    x_1, x_2 = x
    out = np.cos(x_1) * np.sin(x_2) - x_1 / (x_2 ** 2 + 1)
    return out


def alpine1(x):
    out = np.sum(np.abs(x * np.sin(x) + 0.1 * x))
    return out


def alpine2(x):
    out = -np.prod(np.sqrt(x) * np.sin(x))
    return out


def rosenbrock(x):
    out = mw.rosenbrock(x) @ mw.rosenbrock(x)
    return out


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
        "start_x": np.full(10, 30),
        "solution_x": np.zeros(10),
        "start_criterion": 19.950424956466673,
        "solution_criterion": 0,
    },
    "ackley2_good_start": {
        "criterion": ackley2,
        "start_x": np.full(2, 3),
        "solution_x": np.zeros(2),
        "start_criterion": -85.60889823804698,
        "solution_criterion": -200,
    },
    "ackley2_bad_start": {
        "criterion": ackley2,
        "start_x": np.full(2, 25),
        "solution_x": np.zeros(2),
        "start_criterion": -0.1698651409438339,
        "solution_criterion": -200,
    },
    "ackley3_good_start": {
        "criterion": ackley3,
        "start_x": np.full(2, 3),
        # no unique solution
        "solution_x": None,
        "start_criterion": -82.57324651934985,
        "solution_criterion": -170.07756299785044,
    },
    "ackley3_bad_start": {
        "criterion": ackley3,
        "start_x": np.full(2, 25),
        # no unique solution
        "solution_x": None,
        "start_criterion": 8.358584120180984,
        "solution_criterion": -170.07756299785044,
    },
    "ackley4_good_start": {
        "criterion": ackley4,
        "start_x": np.full(2, 3),
        "solution_x": np.array([-1.51, -0.755]),
        "start_criterion": 4.5901006651507235,
        "solution_criterion": -4.5901006651507235,
    },
    "ackley4_bad_start": {
        "criterion": ackley4,
        "start_x": np.full(2, 25),
        "solution_x": np.array([-1.51, -0.755]),
        "start_criterion": 31.054276897735043,
        "solution_criterion": -4.5901006651507235,
    },
    "adjiman": {
        "criterion": adjiman,
        "start_x": np.array([-1, 1]),
        "solution_x": np.array([2, 0.10578]),
        "start_criterion": 0.954648713412841,
        "solution_criterion": -2.0218067833370204,
    },
    "alpine1_good_start": {
        "criterion": alpine1,
        "start_x": np.full(10, 2),
        "solution_x": np.zeros(10),
        "start_criterion": 20.18594853651364,
        "solution_criterion": 0,
    },
    "alpine1_bad_start": {
        "criterion": alpine1,
        "start_x": np.full(10, 10),
        "solution_x": np.zeros(10),
        "start_criterion": 44.40211108893698,
        "solution_criterion": 0,
    },
    "alpine2_good_start": {
        "criterion": alpine2,
        "start_x": np.full(10, 9),
        "solution_x": np.full(10, 7.917),
        "start_criterion": -8.345137486473694,
        "solution_criterion": -30491.15748225926,
    },
    "alpine2_bad_start": {
        "criterion": alpine2,
        "start_x": np.ones(10),
        "solution_x": np.full(10, 7.917),
        "start_criterion": -0.177988299732403,
        "solution_criterion": -30491.15748225926,
    },
    "rosenbrock_good_start": {
        "criterion": rosenbrock,
        "start_x": np.array([-1.2, 1]),
        "solution_x": np.ones(2),
        "start_criterion": 24.2,
        "solution_criterion": 0,
    },
}

SCALAR_FUNCTION_TAGS = {
    "ackley": {
        "continuous": True,
        "convex": False,
        "separable": True,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": True,
    },
    "ackley2": {
        "continuous": False,
        "convex": True,
        "separable": False,
        "differentiable": True,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "ackley3": {
        "continuous": False,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "ackley4": {
        "continuous": False,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "adjiman": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "alpine1": {
        "continuous": False,
        "convex": False,
        "separable": True,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "alpine2": {
        "continuous": True,
        "convex": False,
        "separable": True,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "bartels": {
        "continuous": False,
        "convex": False,
        "separable": False,
        "differentiable": False,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "beale": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": True,
    },
    "bird": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": True,
    },
    "bohachevsky1": {
        "continuous": True,
        "convex": True,
        "separable": True,
        "differentiable": True,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "bohachevsky2": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "bohachevsky3": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "booth": {
        "continuous": True,
        "convex": True,
        "separable": False,
        "differentiable": True,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "branin": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "brent": {
        "continuous": True,
        "convex": True,
        "separable": False,
        "differentiable": True,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "brown": {
        "continuous": True,
        "convex": True,
        "separable": False,
        "differentiable": True,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "bukin6": {
        "continuous": True,
        "convex": True,
        "separable": False,
        "differentiable": False,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "colville": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "crossintray": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": False,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "dejong5": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": True,
    },
    "deckkersaarts": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "dixonprice": {
        "continuous": True,
        "convex": True,
        "separable": False,
        "differentiable": True,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "dropwave": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "easom": {
        "continuous": True,
        "convex": False,
        "separable": True,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "eggcrate": {
        "continuous": True,
        "convex": False,
        "separable": True,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "eggholder": {
        "continuous": False,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "exponential": {
        "continuous": True,
        "convex": True,
        "separable": True,
        "differentiable": True,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "forrester": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "goldsteinprice": {
        "continuous": False,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "gramacylee": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "griewank": {
        "continuous": True,
        "convex": False,
        "separable": True,
        "differentiable": True,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "happycat": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": True,
    },
    "himmelblau": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "holdertable": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": False,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "keane": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "langermann": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": True,
    },
    "leon": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "levy13": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "salamon": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "schaffel1": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "schaffel2": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "schaffel3": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "schaffel4": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "schwefel": {
        "continuous": True,
        "convex": False,
        "separable": True,
        "differentiable": False,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "schwefel2_20": {
        "continuous": True,
        "convex": True,
        "separable": True,
        "differentiable": False,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "schwefel2_21": {
        "continuous": True,
        "convex": True,
        "separable": True,
        "differentiable": False,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "schwefel2_22": {
        "continuous": True,
        "convex": True,
        "separable": True,
        "differentiable": False,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "schwefel2_23": {
        "continuous": True,
        "convex": True,
        "separable": True,
        "differentiable": False,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "shekel": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": True,
    },
    "shubert": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "shubert3": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "shubert4": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "sphere": {
        "continuous": True,
        "convex": True,
        "separable": True,
        "differentiable": False,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "styblinskitank": {
        "continuous": True,
        "convex": False,
        "separable": True,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "sumsquares": {
        "continuous": True,
        "convex": True,
        "separable": True,
        "differentiable": True,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "threehumps": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "trid": {
        "continuous": True,
        "convex": True,
        "separable": False,
        "differentiable": True,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "wolfe": {
        "continuous": True,
        "convex": False,
        "separable": False,
        "differentiable": True,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "xinsheyang": {
        "continuous": False,
        "convex": False,
        "separable": True,
        "differentiable": False,
        "mutimodal": True,
        "randomized_term": True,
        "parametric": False,
    },
    "xinsheyang2": {
        "continuous": False,
        "convex": False,
        "separable": False,
        "differentiable": False,
        "mutimodal": True,
        "randomized_term": False,
        "parametric": False,
    },
    "xinsheyang3": {
        "continuous": True,
        "convex": True,
        "separable": False,
        "differentiable": True,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": True,
    },
    "xinsheyang4": {
        "continuous": True,
        "convex": True,
        "separable": False,
        "differentiable": False,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
    "zakharov": {
        "continuous": False,
        "convex": False,
        "separable": False,
        "differentiable": False,
        "mutimodal": False,
        "randomized_term": False,
        "parametric": False,
    },
}
