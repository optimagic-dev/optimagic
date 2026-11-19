import pytest

import optimagic as om
from optimagic.optimization.optimize_options import OptimizeOptions


@pytest.mark.parametrize(
    "algorithm",
    ["scipy_lbfgsb", om.algos.scipy_lbfgsb, om.algos.scipy_lbfgsb(stopping_maxfun=5)],
)
def test_optimize_options_with_valid_algorithms(algorithm):
    options = OptimizeOptions(algorithm=algorithm)
    assert options.algorithm is algorithm
    assert options.algo_options is None
    assert options.numdiff_options is None
    assert options.scaling is False
    assert options.multistart is False


def test_optimize_options_with_all_fields():
    options = OptimizeOptions(
        algorithm="scipy_lbfgsb",
        algo_options={"stopping_maxfun": 10},
        numdiff_options=om.NumdiffOptions(method="central"),
        scaling=om.ScalingOptions(method="bounds"),
        multistart=om.MultistartOptions(n_samples=10),
    )
    assert options.algo_options == {"stopping_maxfun": 10}
    assert options.numdiff_options == om.NumdiffOptions(method="central")
    assert options.scaling == om.ScalingOptions(method="bounds")
    assert options.multistart == om.MultistartOptions(n_samples=10)


def test_optimize_options_are_exported():
    assert om.OptimizeOptions is OptimizeOptions


@pytest.mark.parametrize("algorithm", [1, None, {"algorithm": "scipy_lbfgsb"}, dict])
def test_optimize_options_with_invalid_algorithm_raise(algorithm):
    with pytest.raises(TypeError, match="algorithm must be the name of an algorithm"):
        OptimizeOptions(algorithm=algorithm)


def test_optimize_options_with_invalid_algo_options_raise():
    with pytest.raises(TypeError, match="algo_options must be a dict or None"):
        OptimizeOptions(
            algorithm="scipy_lbfgsb",
            algo_options=[("stopping_maxfun", 10)],  # ty: ignore[invalid-argument-type]
        )
