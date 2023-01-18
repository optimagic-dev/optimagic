import numpy as np


def adjust_radius(radius, rho, step, options):
    """Adjust the trustregion radius based on relative improvement and stepsize.

    This is just a slight generalization of the pounders radius adjustment. With default
    options it yields the same result.

    Noise handling is not built-in here. It will be achieved by calling the
    function with a noise-adjusted rho.

    Args:
        radius (float): The current trust-region radius.
        rho (float): Actual over expected improvement between the last two accepted
            parameter vectors.
        step (np.ndarray): The step between the last two accepted parameter vectors.
        options (NamedTuple): Options for radius management.

    Returns:
        float: The updated radius.

    """
    step_length = np.linalg.norm(step)
    is_large_step = step_length / radius >= options.large_step

    if rho >= options.rho_increase and is_large_step:
        new_radius = radius * options.expansion_factor
    elif rho >= options.rho_decrease:
        new_radius = radius
    else:
        new_radius = radius * options.shrinking_factor

    if np.isfinite(options.max_radius_to_step_ratio):
        max_radius = np.min(
            [options.max_radius, step_length * options.max_radius_to_step_ratio]
        )
    else:
        max_radius = options.max_radius

    new_radius = np.clip(new_radius, options.min_radius, max_radius)

    return new_radius
