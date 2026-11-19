from dataclasses import dataclass

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

import optimagic as om
from optimagic.benchmarking.benchmark_problem import (
    BenchmarkProblem,
    BenchmarkSolution,
    CallableBenchmarkFunction,
    LeastSquaresBenchmarkFunction,
    LikelihoodBenchmarkFunction,
    ScalarBenchmarkFunction,
    aggregate_function_value,
)
from optimagic.benchmarking.noise import AdditiveBenchmarkNoise
from optimagic.benchmarking.scaling import LinspaceBenchmarkScaler
from optimagic.shared.process_user_function import infer_aggregation_level
from optimagic.typing import AggregationLevel

# ======================================================================================
# Benchmark functions used in the tests
# ======================================================================================


class Sphere(ScalarBenchmarkFunction):
    differentiable = True
    convex = True

    def fun(self, x):
        return float(x @ x)

    def jac(self, x):
        return 2 * x


@dataclass(frozen=True)
class ShiftedResiduals(LeastSquaresBenchmarkFunction):
    shift: float = 1.0

    min_n_params = 2
    max_n_params = 4

    def fun(self, x):
        return np.append(x - self.shift, x[0] * x[1])

    def jac(self, x):
        last_row = np.zeros(len(x))
        last_row[0] = x[1]
        last_row[1] = x[0]
        return np.vstack([np.eye(len(x)), last_row])


class Contributions(LikelihoodBenchmarkFunction):
    def fun(self, x):
        return np.array([x[0] ** 2, x[1] ** 2 + x[0], 3.0])

    def jac(self, x):
        return np.array([[2 * x[0], 0.0], [1.0, 2 * x[1]], [0.0, 0.0]])


class EvenSphere(ScalarBenchmarkFunction):
    def fun(self, x):
        return float(x @ x)

    def supports_n_params(self, n_params):
        return n_params % 2 == 0


class LogSum(ScalarBenchmarkFunction):
    def fun(self, x):
        return float(np.log(x).sum() + x @ x)

    def bounds(self, n_params):
        return om.Bounds(lower=np.zeros(n_params))


FUNCTIONS_WITH_JAC = [Sphere(), ShiftedResiduals(), Contributions()]


# ======================================================================================
# BenchmarkFunction
# ======================================================================================


def test_problem_type_is_set_by_base_classes():
    assert Sphere().problem_type == AggregationLevel.SCALAR
    assert ShiftedResiduals().problem_type == AggregationLevel.LEAST_SQUARES
    assert Contributions().problem_type == AggregationLevel.LIKELIHOOD


def test_default_properties_are_unknown():
    function = Contributions()
    assert function.differentiable is None
    assert function.convex is None
    assert function.min_n_params == 1
    assert function.max_n_params is None
    assert function.bounds(3) is None


def test_properties_can_be_set_in_subclasses():
    assert Sphere().differentiable is True
    assert Sphere().convex is True


def test_has_jac():
    assert Sphere().has_jac
    assert not EvenSphere().has_jac


def test_jac_raises_if_not_available():
    with pytest.raises(NotImplementedError, match="EvenSphere does not have"):
        EvenSphere().jac(np.ones(2))


def test_supports_n_params_without_limits():
    assert Sphere().supports_n_params(1)
    assert Sphere().supports_n_params(10_000)
    assert not Sphere().supports_n_params(0)


def test_supports_n_params_with_limits():
    function = ShiftedResiduals()
    assert not function.supports_n_params(1)
    assert function.supports_n_params(2)
    assert function.supports_n_params(4)
    assert not function.supports_n_params(5)


def test_supports_n_params_can_be_overridden():
    assert EvenSphere().supports_n_params(4)
    assert not EvenSphere().supports_n_params(3)


@pytest.mark.parametrize(
    "value, problem_type, expected",
    [
        (3.0, AggregationLevel.SCALAR, 3.0),
        (np.array([1.0, 2.0]), AggregationLevel.LEAST_SQUARES, 5.0),
        (np.array([1.0, 2.0]), AggregationLevel.LIKELIHOOD, 3.0),
    ],
)
def test_aggregate_function_value(value, problem_type, expected):
    got = aggregate_function_value(value, problem_type=problem_type)
    assert isinstance(got, float)
    assert got == expected


# ======================================================================================
# CallableBenchmarkFunction
# ======================================================================================


def test_callable_benchmark_function():
    function = CallableBenchmarkFunction(
        func=lambda x: x - 1,
        problem_type=AggregationLevel.LEAST_SQUARES,
        jac_func=lambda x: np.eye(len(x)),
        differentiable=True,
        min_n_params=2,
        max_n_params=3,
    )
    aaae(function.fun(np.array([1.0, 3.0])), np.array([0.0, 2.0]))
    aaae(function.jac(np.array([1.0, 3.0])), np.eye(2))
    assert function.has_jac
    assert function.differentiable is True
    assert function.convex is None
    assert function.supports_n_params(3)
    assert not function.supports_n_params(4)


def test_callable_benchmark_function_without_jac():
    function = CallableBenchmarkFunction(
        func=lambda x: float(x @ x), problem_type=AggregationLevel.SCALAR
    )
    assert not function.has_jac
    with pytest.raises(NotImplementedError):
        function.jac(np.ones(2))


def test_callable_benchmark_function_with_invalid_problem_type_raises():
    with pytest.raises(TypeError, match="problem_type must be an AggregationLevel"):
        CallableBenchmarkFunction(
            func=lambda x: x,
            problem_type="least_squares",  # ty: ignore[invalid-argument-type]
        )


# ======================================================================================
# BenchmarkProblem: construction
# ======================================================================================


def test_benchmark_problem_defaults():
    problem = BenchmarkProblem(function=Sphere(), start_params=np.ones(3))
    assert problem.n_params == 3
    assert problem.solution is None
    assert problem.bounds is None
    assert problem.noise is None
    assert problem.scaler is None
    assert problem.seed is None
    assert problem.info == {}


def test_number_of_params_is_fixed_by_start_params():
    function = Sphere()
    assert BenchmarkProblem(function, np.ones(2)).n_params == 2
    assert BenchmarkProblem(function, np.ones(100)).n_params == 100


def test_start_params_are_converted_and_copied():
    start = [1, 2, 3]
    problem = BenchmarkProblem(
        function=Sphere(),
        start_params=start,  # ty: ignore[invalid-argument-type]
    )
    assert isinstance(problem.start_params, np.ndarray)
    assert problem.start_params.dtype == np.float64

    arr = np.ones(3)
    problem = BenchmarkProblem(function=Sphere(), start_params=arr)
    arr[0] = 100
    aaae(problem.start_params, np.ones(3))


def test_arrays_of_problems_and_solutions_are_read_only():
    solution = BenchmarkSolution(fun=0, params=[0, 0, 0])  # ty: ignore[invalid-argument-type]
    assert isinstance(solution.fun, float)
    assert isinstance(solution.params, np.ndarray)
    with pytest.raises(ValueError, match="read-only"):
        solution.params[0] = 1

    problem = BenchmarkProblem(
        function=Sphere(), start_params=np.ones(3), solution=solution
    )
    with pytest.raises(ValueError, match="read-only"):
        problem.start_params[0] = 100

    scaled = problem.with_scaling(LinspaceBenchmarkScaler())
    with pytest.raises(ValueError, match="read-only"):
        scaled.start_params[0] = 100


def test_invalid_function_raises():
    with pytest.raises(TypeError, match="function must be an instance"):
        BenchmarkProblem(
            function=lambda x: x @ x,  # ty: ignore[invalid-argument-type]
            start_params=np.ones(3),
        )


def test_start_params_with_two_dimensions_raise():
    with pytest.raises(ValueError, match="one-dimensional array"):
        BenchmarkProblem(function=Sphere(), start_params=np.ones((2, 2)))


@pytest.mark.parametrize("function", [ShiftedResiduals(), EvenSphere()])
def test_unsupported_number_of_params_raises(function):
    with pytest.raises(ValueError, match="does not support 5 parameters"):
        BenchmarkProblem(function=function, start_params=np.ones(5))


def test_solution_with_wrong_number_of_params_raises():
    with pytest.raises(ValueError, match="The solution has params of shape"):
        BenchmarkProblem(
            function=Sphere(),
            start_params=np.ones(3),
            solution=BenchmarkSolution(fun=0.0, params=np.zeros(2)),
        )


def test_bounds_default_to_bounds_of_function():
    problem = BenchmarkProblem(function=LogSum(), start_params=np.ones(3))
    assert isinstance(problem.bounds, om.Bounds)
    aaae(np.asarray(problem.bounds.lower), np.zeros(3))
    assert problem.bounds.upper is None


def test_bounds_can_be_tighter_than_bounds_of_function():
    bounds = om.Bounds(lower=np.full(3, 0.5), upper=np.full(3, 2.0))
    problem = BenchmarkProblem(
        function=LogSum(), start_params=np.ones(3), bounds=bounds
    )
    assert problem.bounds is bounds


def test_bounds_wider_than_bounds_of_function_raise():
    with pytest.raises(ValueError, match="must not be wider"):
        BenchmarkProblem(
            function=LogSum(),
            start_params=np.ones(3),
            bounds=om.Bounds(lower=np.full(3, -1.0)),
        )


def test_bounds_with_wrong_shape_raise():
    with pytest.raises(ValueError, match="Bounds must have shape"):
        BenchmarkProblem(
            function=Sphere(),
            start_params=np.ones(3),
            bounds=om.Bounds(lower=np.zeros(2)),
        )


def test_bounds_with_wrong_type_raise():
    with pytest.raises(TypeError, match="bounds must be an instance"):
        BenchmarkProblem(
            function=Sphere(),
            start_params=np.ones(3),
            bounds=[(0, 1)] * 3,  # ty: ignore[invalid-argument-type]
        )


# ======================================================================================
# BenchmarkProblem: functions
# ======================================================================================


@pytest.mark.parametrize("function", FUNCTIONS_WITH_JAC)
def test_fun_is_marked_with_problem_type(function):
    problem = BenchmarkProblem(function=function, start_params=np.ones(2))
    assert infer_aggregation_level(problem.fun) == function.problem_type
    assert infer_aggregation_level(problem.noise_free_fun) == function.problem_type


@pytest.mark.parametrize("function", FUNCTIONS_WITH_JAC)
def test_fun_without_noise_and_scaling_is_the_benchmark_function(function):
    x = np.array([0.5, -2.0])
    problem = BenchmarkProblem(function=function, start_params=np.ones(2))
    aaae(problem.fun(x), function.fun(x))
    aaae(problem.noise_free_fun(x), function.fun(x))


def test_start_fun():
    scalar = BenchmarkProblem(function=Sphere(), start_params=np.array([1.0, 2.0]))
    assert scalar.start_fun == 5.0

    least_squares = BenchmarkProblem(ShiftedResiduals(), np.array([2.0, 3.0]))
    assert least_squares.start_fun == 1.0 + 4.0 + 36.0

    likelihood = BenchmarkProblem(Contributions(), np.array([2.0, 3.0]))
    assert likelihood.start_fun == 4.0 + 11.0 + 3.0


def test_noisy_fun_differs_between_evaluations():
    problem = BenchmarkProblem(
        function=ShiftedResiduals(),
        start_params=np.ones(2),
        noise=AdditiveBenchmarkNoise(std=0.1),
        seed=0,
    )
    fun = problem.fun
    first = fun(problem.start_params)
    second = fun(problem.start_params)
    assert not np.allclose(first, second)
    assert not np.allclose(first, problem.noise_free_fun(problem.start_params))


def test_noise_restarts_from_seed_for_each_new_fun():
    problem = BenchmarkProblem(
        function=Sphere(),
        start_params=np.ones(2),
        noise=AdditiveBenchmarkNoise(std=0.1),
        seed=3,
    )
    assert problem.fun(problem.start_params) == problem.fun(problem.start_params)


def test_noise_without_seed_differs_for_each_new_fun():
    problem = BenchmarkProblem(
        function=Sphere(),
        start_params=np.ones(2),
        noise=AdditiveBenchmarkNoise(std=0.1),
    )
    assert problem.fun(problem.start_params) != problem.fun(problem.start_params)


def test_start_fun_ignores_noise():
    problem = BenchmarkProblem(
        function=Sphere(),
        start_params=np.array([1.0, 2.0]),
        noise=AdditiveBenchmarkNoise(std=10),
    )
    assert problem.start_fun == 5.0


# ======================================================================================
# BenchmarkProblem: with_noise and with_scaling
# ======================================================================================


def test_with_noise():
    problem = BenchmarkProblem(function=Sphere(), start_params=np.ones(2))
    noise = AdditiveBenchmarkNoise(std=0.5)
    noisy = problem.with_noise(noise, seed=4)

    assert noisy.noise is noise
    assert noisy.seed == 4
    assert problem.noise is None

    without_noise = noisy.with_noise(None)
    assert without_noise.noise is None
    assert without_noise.seed is None


def test_with_noise_with_invalid_type_raises():
    problem = BenchmarkProblem(function=Sphere(), start_params=np.ones(2))
    with pytest.raises(TypeError, match="noise must be an instance"):
        problem.with_noise(True)  # ty: ignore[invalid-argument-type]


def test_with_scaling():
    scaler = LinspaceBenchmarkScaler(min_scale=2, max_scale=4)
    problem = BenchmarkProblem(
        function=LogSum(),
        start_params=np.array([1.0, 2.0, 3.0]),
        solution=BenchmarkSolution(fun=-1.0, params=np.array([0.5, 0.5, 0.5])),
        bounds=om.Bounds(lower=np.full(3, 0.1), upper=np.full(3, 10.0)),
    )
    scaled = problem.with_scaling(scaler)

    assert scaled.scaler is scaler
    aaae(scaled.start_params, np.array([2.0, 6.0, 12.0]))
    assert scaled.solution is not None
    assert scaled.bounds is not None
    aaae(np.asarray(scaled.solution.params), np.array([1.0, 1.5, 2.0]))
    assert scaled.solution.fun == -1.0
    aaae(np.asarray(scaled.bounds.lower), np.array([0.2, 0.3, 0.4]))
    aaae(np.asarray(scaled.bounds.upper), np.array([20.0, 30.0, 40.0]))

    # the original problem is not changed
    assert problem.scaler is None
    aaae(problem.start_params, np.array([1.0, 2.0, 3.0]))


def test_scaling_does_not_change_function_values():
    """Ported from #723, where the noise free function ignored the scaling."""
    problem = BenchmarkProblem(
        function=ShiftedResiduals(),
        start_params=np.array([1.0, 2.0, 3.0]),
        noise=AdditiveBenchmarkNoise(),
    )
    scaled = problem.with_scaling(LinspaceBenchmarkScaler())

    assert np.allclose(scaled.start_fun, problem.start_fun)
    aaae(
        scaled.noise_free_fun(scaled.start_params),
        problem.noise_free_fun(problem.start_params),
    )


def test_scaling_of_default_bounds():
    scaler = LinspaceBenchmarkScaler(min_scale=2, max_scale=4)
    problem = BenchmarkProblem(function=LogSum(), start_params=np.ones(3))
    scaled = problem.with_scaling(scaler)
    assert scaled.bounds is not None
    aaae(np.asarray(scaled.bounds.lower), np.zeros(3))

    # bounds that are valid in the scaled space but not in the original space
    with pytest.raises(ValueError, match="must not be wider"):
        BenchmarkProblem(
            function=LogSum(),
            start_params=np.ones(3),
            bounds=om.Bounds(lower=np.full(3, -1.0)),
            scaler=scaler,
        )


def test_scaling_twice_raises():
    problem = BenchmarkProblem(function=Sphere(), start_params=np.ones(2))
    scaled = problem.with_scaling(LinspaceBenchmarkScaler())
    with pytest.raises(ValueError, match="The problem is already scaled"):
        scaled.with_scaling(LinspaceBenchmarkScaler())


def test_with_scaling_with_invalid_type_raises():
    problem = BenchmarkProblem(function=Sphere(), start_params=np.ones(2))
    with pytest.raises(TypeError, match="scaler must be an instance"):
        problem.with_scaling(True)  # ty: ignore[invalid-argument-type]


# ======================================================================================
# BenchmarkProblem: derivatives
# ======================================================================================

EXPECTED_DERIVATIVE_LEVELS = {
    AggregationLevel.SCALAR: [AggregationLevel.SCALAR],
    AggregationLevel.LEAST_SQUARES: [
        AggregationLevel.SCALAR,
        AggregationLevel.LIKELIHOOD,
        AggregationLevel.LEAST_SQUARES,
    ],
    AggregationLevel.LIKELIHOOD: [AggregationLevel.SCALAR, AggregationLevel.LIKELIHOOD],
}


def _evaluate_at_level(function, x, level):
    """Evaluate a benchmark function as an optimizer of the given level sees it."""
    raw = function.fun(x)
    if level == function.problem_type:
        return raw
    if level == AggregationLevel.SCALAR:
        return aggregate_function_value(raw, function.problem_type)
    # least-squares function seen by a likelihood optimizer
    return raw**2


@pytest.mark.parametrize("function", FUNCTIONS_WITH_JAC)
def test_jac_matches_numerical_derivatives(function):
    x = np.array([0.7, -1.3])
    problem = BenchmarkProblem(function=function, start_params=np.ones(2))

    derivatives = problem.jac
    assert derivatives is not None
    levels = [infer_aggregation_level(derivative) for derivative in derivatives]
    assert levels == EXPECTED_DERIVATIVE_LEVELS[function.problem_type]

    for derivative, level in zip(derivatives, levels, strict=True):
        expected = om.first_derivative(
            lambda x, level=level: _evaluate_at_level(function, x, level), x
        ).derivative
        aaae(derivative(x), expected, decimal=5)


def test_jac_is_none_without_closed_form_derivative():
    problem = BenchmarkProblem(function=EvenSphere(), start_params=np.ones(2))
    assert problem.jac is None


def test_jac_is_none_with_noise_or_scaling():
    problem = BenchmarkProblem(function=Sphere(), start_params=np.ones(2))
    assert problem.with_noise(AdditiveBenchmarkNoise()).jac is None
    assert problem.with_scaling(LinspaceBenchmarkScaler()).jac is None


@pytest.mark.parametrize("algorithm", ["scipy_lbfgsb", "scipy_ls_lm"])
def test_problem_can_be_passed_to_minimize(algorithm):
    problem = BenchmarkProblem(
        function=ShiftedResiduals(shift=2.0), start_params=np.array([3.0, 0.0])
    )
    derivatives = problem.jac
    assert derivatives is not None
    res = om.minimize(
        fun=problem.fun,
        params=problem.start_params,
        algorithm=algorithm,
        jac=derivatives,
        bounds=problem.bounds,
    )
    assert res.fun < problem.start_fun
    expected_fun = aggregate_function_value(
        problem.function.fun(res.params), AggregationLevel.LEAST_SQUARES
    )
    assert np.allclose(res.fun, expected_fun)
    # the first entry of jac is the gradient of the sum of squared residuals
    aaae(derivatives[0](res.params), np.zeros(2), decimal=4)
