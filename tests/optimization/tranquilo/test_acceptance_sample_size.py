import pytest
from estimagic.optimization.tranquilo.acceptance_sample_size import (
    _compute_factor,
    get_optimal_sample_sizes,
)
from scipy.stats import norm

TEST_CASES = [
    (0.5, 0.5, 0.5, 0),
    (1.0, norm.cdf(0.5), norm.sf(0.5), 1),
    (2.0, norm.cdf(0.5), norm.sf(0.5), 1 / 4),
]


@pytest.mark.parametrize(
    "minimal_effect_size, power_level, significance_level, expected_factor", TEST_CASES
)
def test_factor(minimal_effect_size, power_level, significance_level, expected_factor):
    assert (
        abs(
            expected_factor
            - _compute_factor(minimal_effect_size, power_level, significance_level)
        )
        < 1e-6
    )


@pytest.mark.parametrize("minimal_effect_size", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("power_level", [0.25, 0.5, 0.75])
@pytest.mark.parametrize("significance_level", [0.01, 0.05, 0.1, 0.2])
def test_bounds(minimal_effect_size, power_level, significance_level):
    res = [
        get_optimal_sample_sizes(
            sd_1=1,
            sd_2=1,
            existing_n1=_n1,
            minimal_effect_size=minimal_effect_size,
            power_level=power_level,
            significance_level=significance_level,
        )
        for _n1 in (0, 10)
    ]
    # test that if both sample sizes are chosen optimally the overall number is smaller
    assert sum(res[0]) <= sum(res[1]) + 10
    # test that if there are existing samples in the first group, the second group
    # can be smaller than if there are no existing samples in the first group
    assert res[0][1] >= res[1][1]


def test_standard_deviation_influence():
    n1, n2 = get_optimal_sample_sizes(
        sd_1=1,
        sd_2=3,
        existing_n1=0,
        minimal_effect_size=0.5,
        power_level=0.5,
        significance_level=0.2,
    )
    assert n1 < n2


def test_inequality():
    # Test that the inequality condition is satisfied
    n1, n2 = get_optimal_sample_sizes(
        sd_1=1,
        sd_2=2,
        existing_n1=0,
        minimal_effect_size=0.5,
        power_level=0.5,
        significance_level=0.2,
    )
    factor = _compute_factor(0.5, 0.5, 0.2)
    lhs = (1 / n1 + 2 / n2) ** (-1)
    assert lhs >= factor


def test_first_group_is_not_sampled():
    n1, _ = get_optimal_sample_sizes(
        sd_1=1,
        sd_2=1,
        existing_n1=10,
        minimal_effect_size=0.5,
        power_level=0.5,
        significance_level=0.2,
    )
    assert n1 == 0
