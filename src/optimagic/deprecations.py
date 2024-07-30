import warnings


def throw_criterion_future_warning():
    msg = (
        "To align optimagic with scipy.optimize, the `criterion` argument has been "
        "renamed to `fun`. Please use `fun` instead of `criterion`. Using `criterion` "
        " will become an error in optimagic version 0.6.0 and later."
    )
    warnings.warn(msg, FutureWarning)


def throw_criterion_kwargs_future_warning():
    msg = (
        "To align optimagic with scipy.optimize, the `criterion_kwargs` argument has "
        "been renamed to `fun_kwargs`. Please use `fun_kwargs` instead of "
        "`criterion_kwargs`. Using `criterion_kwargs` will become an error in "
        "optimagic version 0.6.0 and later."
    )
    warnings.warn(msg, FutureWarning)


def throw_derivative_future_warning():
    msg = (
        "To align optimagic with scipy.optimize, the `derivative` argument has been "
        "renamed to `jac`. Please use `jac` instead of `derivative`. Using `derivative`"
        " will become an error in optimagic version 0.6.0 and later."
    )
    warnings.warn(msg, FutureWarning)


def throw_derivative_kwargs_future_warning():
    msg = (
        "To align optimagic with scipy.optimize, the `derivative_kwargs` argument has "
        "been renamed to `jac_kwargs`. Please use `jac_kwargs` instead of "
        "`derivative_kwargs`. Using `derivative_kwargs` will become an error in "
        "optimagic version 0.6.0 and later."
    )
    warnings.warn(msg, FutureWarning)


def throw_criterion_and_derivative_future_warning():
    msg = (
        "To align optimagic with scipy.optimize, the `criterion_and_derivative` "
        "argument has been renamed to `fun_and_jac`. Please use `fun_and_jac` "
        "instead of `criterion_and_derivative`. Using `criterion_and_derivative` "
        "will become an error in optimagic version 0.6.0 and later."
    )
    warnings.warn(msg, FutureWarning)


def throw_criterion_and_derivative_kwargs_future_warning():
    msg = (
        "To align optimagic with scipy.optimize, the `criterion_and_derivative_kwargs` "
        "argument has been renamed to `fun_and_jac_kwargs`. Please use "
        "`fun_and_jac_kwargs` instead of `criterion_and_derivative_kwargs`. Using "
        "`criterion_and_derivative_kwargs` will become an error in optimagic version "
        "0.6.0 and later."
    )
    warnings.warn(msg, FutureWarning)


def throw_scaling_options_future_warning():
    msg = (
        "The `scaling_options` argument will be deprecated in favor of `scaling` in "
        "optimagic version 0.6.0 and later. You can simply pass the scaling options to "
        "`scaling` instead of `scaling_options`. Using `scaling_options` will become "
        "an error in optimagic version 0.6.0 and later."
    )
    warnings.warn(msg, FutureWarning)


def replace_and_warn_about_deprecated_algo_options(algo_options):
    if not isinstance(algo_options, dict):
        return algo_options

    algo_options = {k.replace(".", "_"): v for k, v in algo_options.items()}

    replacements = {
        "stopping_max_criterion_evaluations": "stopping_maxfun",
        "stopping_max_iterations": "stopping_maxiter",
        "convergence_absolute_criterion_tolerance": "convergence_ftol_abs",
        "convergence_relative_criterion_tolerance": "convergence_ftol_rel",
        "convergence_scaled_criterion_tolerance": "convergence_ftol_scaled",
        "convergence_absolute_params_tolerance": "convergence_xtol_abs",
        "convergence_relative_params_tolerance": "convergence_xtol_rel",
        "convergence_absolute_gradient_tolerance": "convergence_gtol_abs",
        "convergence_relative_gradient_tolerance": "convergence_gtol_rel",
        "convergence_scaled_gradient_tolerance": "convergence_gtol_scaled",
    }

    present = sorted(set(algo_options) & set(replacements))
    if present:
        msg = (
            "The following keys in `algo_options` are deprecated and will be removed "
            "in optimagic version 0.6.0 and later. Please replace them as follows:\n"
        )
        for k in present:
            msg += f"  {k} -> {replacements[k]}\n"

        warnings.warn(msg, FutureWarning)

    out = {k: v for k, v in algo_options.items() if k not in present}
    for k in present:
        out[replacements[k]] = algo_options[k]

    return out
