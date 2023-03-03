import numpy as np
import pytest
from estimagic.optimization.tranquilo.aggregate_models import get_aggregator
from estimagic.optimization.tranquilo.fit_models import get_fitter
from estimagic.optimization.tranquilo.models import ModelInfo
from estimagic.optimization.tranquilo.options import Region
from estimagic.optimization.tranquilo.rho_noise import simulate_rho_noise
from estimagic.optimization.tranquilo.solve_subproblem import get_subsolver
from numpy.testing import assert_array_almost_equal as aaae


@pytest.mark.parametrize("functype", ["scalar", "least_squares"])
def test_convergence_to_one_if_nois_is_tiny(functype):
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
        model_info = ModelInfo(False, False)
        model_aggregator = get_aggregator(
            aggregator="least_squares_linear",
            functype="least_squares",
            model_info=model_info,
        )
        n_residuals = 2
    else:
        fvecs = (xs**2).sum(axis=1).reshape(-1, 1)
        model_info = ModelInfo(True, True)
        model_aggregator = get_aggregator(
            aggregator="identity",
            functype="scalar",
            model_info=model_info,
        )
        n_residuals = 1

    noise_cov = np.eye(n_residuals) * 1e-12

    xs_centered = xs - 0.5
    trustregion = Region(center=np.ones(2) * 0.5, radius=1, shape="sphere")
    model_fitter = get_fitter(fitter="ols", model_info=model_info)

    vector_model = model_fitter(
        xs_centered, fvecs, weights=None, region=trustregion, old_model=None
    )

    subsolver = get_subsolver(solver="gqtpar")

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
