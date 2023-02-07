import numpy as np
from scipy.stats import norm

from estimagic.optimization.tranquilo.get_component import get_component


def get_acceptance_sample_size_calculator(size_calculator, acceptance_options):
    func_dict = {
        "naive": get_naive_n_acceptance_points,
        "asymptotic": get_asymptotic_n_acceptance_points,
    }

    default_options = {
        "acceptance_options": acceptance_options,
    }

    out = get_component(
        name_or_func=size_calculator,
        func_dict=func_dict,
        default_options=default_options,
    )

    return out


def get_naive_n_acceptance_points():
    return 0, 5


def get_asymptotic_n_acceptance_points(
    sigma,
    existing_n1,
    expected_improvement,
    acceptance_options,
):
    n1, n2 = get_optimal_sample_sizes(
        sd_1=sigma,
        sd_2=sigma,
        existing_n1=existing_n1,
        minimal_effect_size=expected_improvement,
        power_level=acceptance_options.power_level,
        significance_level=1 - acceptance_options.confidence_level,
    )

    n1 = int(np.clip(n1, 0, acceptance_options.n_max - existing_n1))
    n2 = int(np.clip(n2, acceptance_options.n_min, acceptance_options.n_max))
    return n1, n2


def get_optimal_sample_sizes(
    sd_1, sd_2, existing_n1, minimal_effect_size, power_level, significance_level
):
    """Return missing sample sizes.

    Args:
        sd_1 (float): Standard deviation of the first group.
        sd_2 (float): Standard deviation of the second group.
        existing_n1 (int): Number of samples in the first group.
        minimal_effect_size (float): Minimal effect size.
        power_level (float): Power level.
        significance_level (float): Significance level.

    Returns:
        tuple: Missing sample sizes.

    """
    factor = _compute_factor(minimal_effect_size, power_level, significance_level)

    n1_optimal = (sd_1 * (sd_2 + sd_1)) * factor
    n2_optimal = (sd_2 * (sd_2 + sd_1)) * factor

    if existing_n1 <= n1_optimal:
        n1 = n1_optimal - existing_n1
        n2 = n2_optimal
    else:
        n1 = 0
        n2 = sd_2**2 * (factor ** (-1) - sd_1**2 / existing_n1) ** (-1)

    n1 = int(np.ceil(n1))
    n2 = int(np.ceil(n2))

    return n1, n2


def _compute_factor(minimal_effect_size, power_level, significance_level):
    factor = (
        (norm.ppf(1 - significance_level) + norm.ppf(power_level)) / minimal_effect_size
    ) ** 2
    return factor
