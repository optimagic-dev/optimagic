import numpy as np


def adjust_radius(
    radius, rho, step, options, is_good_geometry, disable_safety_iterations
):
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
    needs_safety = False

    if rho >= options.rho_increase and is_large_step:
        new_radius = radius * options.expansion_factor
    elif rho >= options.rho_decrease:
        new_radius = radius
    elif is_good_geometry or disable_safety_iterations:
        new_radius = radius * options.shrinking_factor
    else:
        new_radius = radius
        needs_safety = True

    new_radius = np.clip(new_radius, options.min_radius, options.max_radius)

    return new_radius, needs_safety
