import numpy as np
import pytest
from numpy.testing import assert_allclose

from optimagic import mark
from optimagic.config import IS_NLOPT_INSTALLED
from optimagic.exceptions import InvalidAlgoOptionError
from optimagic.optimization.optimize import minimize
from optimagic.optimizers.nlopt_optimizers import NloptESCH
from optimagic.parameters.bounds import Bounds

if IS_NLOPT_INSTALLED:
    import nlopt


pytestmark = pytest.mark.skipif(not IS_NLOPT_INSTALLED, reason="nlopt not installed")


def _run(algorithm, options):
    evaluations = []

    @mark.scalar
    def objective(x):
        evaluations.append(x.copy())
        return float(x @ x)

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
        (
            "nlopt_direct",
            {"locally_biased": True, "random_search": True, "unscaled_bounds": True},
        ),
    ],
)
def test_seed_reproduces_evaluations_and_is_forwarded(algorithm, options, monkeypatch):
    seeds = []
    original_srand = nlopt.srand

    def record_seed(seed):
        seeds.append(seed)
        original_srand(seed)

    monkeypatch.setattr(nlopt, "srand", record_seed)
    first = _run(algorithm, {**options, "seed": 0})
    second = _run(algorithm, {**options, "seed": 0})

    assert seeds == [0, 0]
    assert_allclose(first, second, rtol=0, atol=0)


@pytest.mark.parametrize("algorithm", ["nlopt_esch", "nlopt_isres", "nlopt_crs2_lm"])
def test_different_seeds_change_evaluations(algorithm):
    first = _run(algorithm, {"seed": 123})
    second = _run(algorithm, {"seed": 456})

    assert first.shape != second.shape or not np.array_equal(first, second)


def test_none_seed_preserves_external_seed():
    nlopt.srand(123)
    first = _run("nlopt_esch", {"seed": None})
    nlopt.srand(123)
    second = _run("nlopt_esch", {"seed": None})
    nlopt.srand(456)
    third = _run("nlopt_esch", {"seed": None})

    assert_allclose(first, second, rtol=0, atol=0)
    assert first.shape != third.shape or not np.array_equal(first, third)


def test_deterministic_direct_does_not_reset_random_state(monkeypatch):
    def fail_on_seed(seed):
        pytest.fail(f"Unexpected seed: {seed}")

    monkeypatch.setattr(nlopt, "srand", fail_on_seed)
    _run("nlopt_direct", {"seed": 123})


@pytest.mark.parametrize("seed", [0, 2**32 - 1])
def test_seed_limits_are_valid(seed):
    assert NloptESCH(seed=seed).seed == seed
    _run("nlopt_esch", {"seed": seed})


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_seed_outside_portable_range_is_rejected(seed):
    with pytest.raises(InvalidAlgoOptionError):
        NloptESCH(seed=seed)
