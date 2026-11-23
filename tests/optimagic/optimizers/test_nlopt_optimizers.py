import numpy as np
import pytest
from numpy.testing import assert_allclose

import optimagic as om
from optimagic import mark
from optimagic.config import IS_NLOPT_INSTALLED
from optimagic.exceptions import InvalidAlgoOptionError
from optimagic.optimization.optimize import minimize
from optimagic.optimizers.nlopt_optimizers import NloptESCH
from optimagic.parameters.bounds import Bounds

if IS_NLOPT_INSTALLED:
    import nlopt


pytestmark = [
    pytest.mark.nlopt,
    pytest.mark.skipif(not IS_NLOPT_INSTALLED, reason="nlopt not installed"),
]


def _run(algorithm, options, *, flat=False):
    evaluations = []

    @mark.scalar
    def objective(x):
        evaluations.append(x.copy())
        return 1.0 if flat else float(x @ x)

    bounds = (
        None
        if algorithm == "nlopt_praxis"
        else Bounds(lower=np.full(3, -1.0), upper=np.full(3, 1.0))
    )
    minimize(
        fun=objective,
        params=np.array([0.5, -0.4, 0.3]),
        algorithm=algorithm,
        algo_options={"stopping_maxfun": 80, **options},
        collect_history=False,
        bounds=bounds,
    )
    return np.array(evaluations)


@pytest.mark.parametrize(
    ("algorithm", "options"),
    [
        ("nlopt_praxis", {}),
        ("nlopt_esch", {}),
        ("nlopt_isres", {}),
        ("nlopt_crs2_lm", {}),
        ("nlopt_direct", {"locally_biased": True, "random_search": True}),
    ],
)
def test_seed_controls_evaluation_sequence(algorithm, options):
    # A flat objective makes PRAXIS take random steps.
    flat = algorithm == "nlopt_praxis"
    first = _run(algorithm, {**options, "seed": 0}, flat=flat)
    repeated = _run(algorithm, {**options, "seed": 0}, flat=flat)
    different = _run(algorithm, {**options, "seed": 1}, flat=flat)

    assert_allclose(first, repeated, rtol=0, atol=0)
    assert first.shape != different.shape or not np.array_equal(first, different)


def test_none_seed_preserves_external_seed():
    nlopt.srand(123)
    first = _run("nlopt_esch", {"seed": None})
    nlopt.srand(123)
    second = _run("nlopt_esch", {"seed": None})
    nlopt.srand(456)
    third = _run("nlopt_esch", {"seed": None})

    assert_allclose(first, second, rtol=0, atol=0)
    assert first.shape != third.shape or not np.array_equal(first, third)


def test_deterministic_direct_does_not_reset_random_state():
    nlopt.srand(456)
    expected = _run("nlopt_esch", {"seed": None})
    nlopt.srand(456)
    _run("nlopt_direct", {"seed": 123})
    actual = _run("nlopt_esch", {"seed": None})

    assert_allclose(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("seed", [0, 2**32 - 1])
def test_seed_limits_are_valid(seed):
    assert NloptESCH(seed=seed).seed == seed
    _run("nlopt_esch", {"seed": seed})


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_seed_outside_portable_range_is_rejected(seed):
    with pytest.raises(InvalidAlgoOptionError):
        NloptESCH(seed=seed)


ONE_SIDED_BOUNDS_CASES = [
    # start params, bounds, solution of min (x - 3) @ (x - 3) subject to bounds
    (np.zeros(2), Bounds(upper=np.full(2, 1.0)), np.array([1.0, 1.0])),
    (np.full(2, 5.0), Bounds(lower=np.full(2, 4.0)), np.array([4.0, 4.0])),
    (
        np.zeros(2),
        Bounds(lower=np.array([-1.0, -np.inf]), upper=np.array([1.0, np.inf])),
        np.array([1.0, 3.0]),
    ),
]


@pytest.mark.parametrize("algorithm", ["nlopt_newuoa", "nlopt_praxis"])
@pytest.mark.parametrize(("params", "bounds", "expected"), ONE_SIDED_BOUNDS_CASES)
def test_one_sided_bounds_are_respected(algorithm, params, bounds, expected):
    res = minimize(
        fun=lambda x: (x - 3) @ (x - 3),
        params=params,
        algorithm=algorithm,
        bounds=bounds,
    )
    assert_allclose(res.params, expected, atol=1e-4)


def test_praxis_is_marked_as_supporting_bounds():
    assert "nlopt_praxis" in om.algos.Bounded.AllNames


def test_direct_random_search_without_locally_biased_raises():
    with pytest.raises(ValueError, match="requires locally_biased=True"):
        minimize(
            fun=lambda x: x @ x,
            params=np.array([0.5, 0.5]),
            algorithm=om.algos.nlopt_direct(random_search=True),
            bounds=Bounds(lower=np.full(2, -1.0), upper=np.full(2, 1.0)),
        )
