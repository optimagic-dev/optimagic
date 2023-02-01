import numpy as np

from estimagic.optimization.tranquilo.get_component import get_component
from estimagic.optimization.tranquilo.noise_models import NoiseModel


def get_noise_fitter(fitter, user_options):
    func_dict = {"naive": _fit_noise_from_last_acceptance_samples}

    out = get_component(
        name_or_func=fitter,
        func_dict=func_dict,
        user_options=user_options,
    )

    return out


def _fit_noise_from_last_acceptance_samples(history, states, noise_options):
    fit_flags = [
        noise_options.fit_intercept,
        noise_options.fit_slope,
        noise_options.fit_square,
    ]

    n_states = min(sum(fit_flags), len(states))

    relevant_states = states[-n_states:]

    y_list, x_list = [], []
    for state in relevant_states:
        y, x = _get_regression_data(state, history)
        y_list.append(y)
        x_list.append(x)

    y = np.array(y_list)
    x = np.array(x_list)

    keep = np.array(
        noise_options.fit_intercept,
        noise_options.fit_slope and n_states >= 2 or not noise_options.fit_intercept,
        noise_options.fit_square and n_states >= 3,
    )

    names = [name for name, k in zip(["intercept", "slope", "square"], keep) if k]

    x = x[:, keep]

    beta = np.linalg.lstsq(x, y, rcond=None)[0]

    noise_model = NoiseModel(**dict(zip(names, beta)))

    return noise_model


def _get_regression_data(state, history):
    sample = history.get_fvals(state.acceptance_indices)
    sigma = np.std(sample)
    mu = np.mean(sample)
    regressors = [1, mu, mu**2]

    return sigma, regressors
