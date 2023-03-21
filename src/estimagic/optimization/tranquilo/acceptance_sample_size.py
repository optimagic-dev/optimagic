import numpy as np
from scipy.stats import norm


def get_acceptance_sample_sizes(
    sigma,
    existing_n1,
    expected_improvement,
    power_level,
    confidence_level,
    n_min,
    n_max,
):
    n1_raw, n2_raw = _get_optimal_sample_sizes(
        sd_1=sigma,
        sd_2=sigma,
        existing_n1=existing_n1,
        minimal_effect_size=np.clip(expected_improvement, 1e-8, np.inf),
        power_level=power_level,
        significance_level=1 - confidence_level,
    )

    n1 = int(np.ceil(np.clip(n1_raw, 0, max(0, n_max - existing_n1))))
    n2 = int(np.ceil(np.clip(n2_raw, n_min, n_max)))
    return n1, n2


def _get_optimal_sample_sizes(
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

    return n1, n2


def _compute_factor(minimal_effect_size, power_level, significance_level):
    # avoid division by zero warning; will be clipped later
    if minimal_effect_size == 0:
        factor = np.inf
    else:
        factor = (
            (norm.ppf(1 - significance_level) + norm.ppf(power_level))
            / minimal_effect_size
        ) ** 2
    return factor
