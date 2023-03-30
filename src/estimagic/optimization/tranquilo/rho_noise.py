import numpy as np

from estimagic.optimization.tranquilo.acceptance_decision import calculate_rho


def simulate_rho_noise(
    xs,
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
        xs (np.ndarray): Sample of points on which surrogate models will be
            fitted during the simulation. This sample is not scaled to the trustregion.
        vector_model (VectorModel): A vector surrogate model that is taken as true model
            for the simulation. In many cases this model was fitted on xs but this is
            not a requirement.
        trustregion (Region): The trustregion in which the optimization is performed.
        noise_cov(np.ndarray): Covariance matrix of the noise. The noise is assumed to
            be drawn from a multivariate normal distribution with mean zero and this
            covariance matrix.
        model_fitter (callable): A function that fits a model.
        model_aggregator (callable): A function that aggregates a vector model to a
            scalar model.
        subsolver (callable): A function that solves the subproblem.
        rng (np.random.Generator): Random number generator.
        n_draws (int): Number of draws used to estimate the rho noise.
        ignore_corelation (bool): If True, the noise is assumed to be uncorrelated and
            only the diagonal entries of the covariance matrix are used.

    """
    n_samples, n_params = xs.shape
    n_residuals = len(noise_cov)

    x_unit = trustregion.map_to_unit(xs)

    true_fvecs = vector_model.predict(x_unit)

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
        sim_vector_model = model_fitter(
            xs,
            sim_fvecs,
            weights=None,
            region=trustregion,
            old_model=None,
        )
        sim_scalar_model = model_aggregator(vector_model=sim_vector_model)
        sim_sub_sol = subsolver(sim_scalar_model, trustregion)

        sim_candidate_fval = true_scalar_model.predict(sim_sub_sol.x_unit)
        sim_actual_improvement = -(sim_candidate_fval - true_current_fval)

        sim_rho = calculate_rho(
            actual_improvement=sim_actual_improvement,
            expected_improvement=sim_sub_sol.expected_improvement,
        )

        rhos.append(sim_rho)

    return np.array(rhos)
