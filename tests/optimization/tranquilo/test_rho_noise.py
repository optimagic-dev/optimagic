import numpy as np
import pytest
from tranquilo.optimization.tranquilo.aggregate_models import get_aggregator
from tranquilo.optimization.tranquilo.fit_models import get_fitter
from tranquilo.optimization.tranquilo.region import Region
from tranquilo.optimization.tranquilo.bounds import Bounds
from tranquilo.optimization.tranquilo.rho_noise import simulate_rho_noise
from tranquilo.optimization.tranquilo.solve_subproblem import get_subsolver
from numpy.testing import assert_array_almost_equal as aaae


@pytest.mark.parametrize("functype", ["scalar", "least_squares"])
def test_convergence_to_one_if_noise_is_tiny(functype):
    """Test simulate_rho_noise.

    For the test, the "true" model is a standard sphere function.

    """
    xs = (
        np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.0, -1.0],
            ]
        )
        + 0.5
    )

    if functype == "least_squares":
        fvecs = xs.copy()
        model_type = "linear"
        model_aggregator = get_aggregator(
            aggregator="least_squares_linear",
        )
        n_residuals = 2
    else:
        fvecs = (xs**2).sum(axis=1).reshape(-1, 1)
        model_type = "quadratic"
        model_aggregator = get_aggregator(
            aggregator="identity",
        )
        n_residuals = 1

    noise_cov = np.eye(n_residuals) * 1e-12

    trustregion = Region(center=np.ones(2) * 0.5, radius=1.0, bounds=Bounds(None, None))
    model_fitter = get_fitter(
        fitter="ols",
        model_type=model_type,
        residualize=False,
        infinity_handling="relative",
    )

    vector_model = model_fitter(
        xs, fvecs, weights=None, region=trustregion, old_model=None
    )

    subsolver = get_subsolver(sphere_solver="gqtpar", cube_solver="bntr")

    rng = np.random.default_rng(123)

    got = simulate_rho_noise(
        xs=xs,
        vector_model=vector_model,
        trustregion=trustregion,
        noise_cov=noise_cov,
        model_fitter=model_fitter,
        model_aggregator=model_aggregator,
        subsolver=subsolver,
        rng=rng,
        n_draws=100,
        ignore_corelation=True,
    )

    aaae(got, np.ones_like(got), decimal=4)
