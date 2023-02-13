import numpy as np

from estimagic.optimization.tranquilo.acceptance_decision import calculate_rho


def simulate_rho_noise(
    model_xs,
    vector_model,
    trustregion,
    noise_cov,
    model_fitter,
    model_aggregator,
    subsolver,
    rng,
    n_draws=100,
    ignore_corelation=True,
):
    """Simulate a rho that would obtain on average if there is no approximation error.

    This can be used to adjust the sample size in the presence of noise.

    Throughout this function the prefix true refers to what is considered as ground
    truth for the purpose of the simulation. The prefix sim refers to the simulated
    quantities.

    Args:
        model_xs (np.ndarray): Array of points that was used to fit the model.
        model (ScalarModel): A scalar surrogate model that is taken as true model
            for the simulation.
        noise_model (NoiseModel): A model of the standard deviation of noise over
            the parameter space.
        model_fitter (callable): A function that fits a model.
        subsolver (Subsolver): Subsolver object.
        n_draws (int): Number of draws used to estimate the rho noise.

    """
    n_samples, n_params = model_xs.shape
    n_residuals = len(noise_cov)

    centered_xs = (model_xs - trustregion.center) / trustregion.radius

    true_fvecs = vector_model.predict(centered_xs)

    true_scalar_model = model_aggregator(vector_model=vector_model)

    true_current_fval = true_scalar_model.predict(np.zeros(n_params))

    if ignore_corelation:
        noise_cov = np.diag(np.diag(noise_cov))

    noise = rng.multivariate_normal(
        mean=np.zeros(n_residuals), cov=noise_cov, size=n_draws * n_samples
    ).reshape(n_draws, n_samples, n_residuals)

    rhos = []
    for draw in noise:
        sim_fvecs = true_fvecs + draw
        sim_vector_model = model_fitter(centered_xs, sim_fvecs)
        sim_scalar_model = model_aggregator(vector_model=sim_vector_model)
        sim_sub_sol = subsolver(sim_scalar_model, trustregion)

        sim_candidate_fval = true_scalar_model.predict(sim_sub_sol.centered_x)
        sim_actual_improvement = -(sim_candidate_fval - true_current_fval)

        sim_rho = calculate_rho(
            actual_improvement=sim_actual_improvement,
            expected_improvement=sim_sub_sol.expected_improvement,
        )

        rhos.append(sim_rho)

    return np.array(rhos)
