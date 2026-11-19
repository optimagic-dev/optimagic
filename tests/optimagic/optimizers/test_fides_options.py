"""Smoke tests for fides that run by default.

Detailed tests of the options are in test_fides_wrappers.py.

"""

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from optimagic.config import IS_FIDES_INSTALLED
from optimagic.optimization.optimize import minimize
from optimagic.parameters.bounds import Bounds

if IS_FIDES_INSTALLED:
    from fides.hessian_approximation import FX, SR1, Broyden

    from optimagic.optimizers.fides import Fides
else:
    FX = lambda: None
    SR1 = lambda: None
    Broyden = lambda phi: None  # noqa: ARG005

test_cases_no_contribs_needed = [
    {},
    {"hessian_update": "bfgs"},
    {"hessian_update": "BFGS"},
    {"hessian_update": SR1()},
    {"hessian_update": Broyden(phi=0.5)},
    {"hessian_update": "sr1"},
    {"hessian_update": "DFP"},
    {"hessian_update": "bb"},
    {"convergence_ftol_rel": 1e-6},
    {"convergence_xtol_abs": 1e-6},
    {"convergence_gtol_abs": 1e-6},
    {"convergence_gtol_rel": 1e-6},
    {"stopping_maxiter": 100},
    {"stopping_maxtime": 200},
    {"trustregion_initial_radius": 20, "stepback_strategy": "truncate"},
    {"subspace_solver": "full"},
    {"subspace_solver": "2D", "stepback_strategy": "refine"},
    {"subspace_solver": "scg"},
    {"max_stepback_fraction": 0.8},
    {"trustregion_shrinking_threshold": 0.4, "trustregion_shrinking_factor": 0.2},
    {"trustregion_expansion_threshold": 0.9, "trustregion_expansion_factor": 4},
]


def criterion_and_derivative(x):
    return (x**2).sum(), 2 * x


def criterion(x):
    return (x**2).sum()


@pytest.mark.skipif(not IS_FIDES_INSTALLED, reason="fides not installed.")
@pytest.mark.parametrize("algo_options", test_cases_no_contribs_needed)
def test_fides_correct_algo_options(algo_options):
    res = minimize(
        fun_and_jac=criterion_and_derivative,
        fun=criterion,
        x0=np.array([1, -5, 3]),
        bounds=Bounds(
            lower=np.array([-10, -10, -10]),
            upper=np.array([10, 10, 10]),
        ),
        algorithm=Fides(**algo_options),
    )
    aaae(res.params, np.zeros(3), decimal=4)


test_cases_needing_contribs = [
    {"hessian_update": FX()},
    {"hessian_update": "ssm"},
    {"hessian_update": "TSSM"},
    {"hessian_update": "gnsbfgs"},
]


@pytest.mark.skipif(not IS_FIDES_INSTALLED, reason="fides not installed.")
@pytest.mark.parametrize("algo_options", test_cases_needing_contribs)
def test_fides_unimplemented_algo_options(algo_options):
    with pytest.raises(NotImplementedError):
        minimize(
            fun_and_jac=criterion_and_derivative,
            fun=criterion,
            x0=np.array([1, -5, 3]),
            bounds=Bounds(
                lower=np.array([-10, -10, -10]),
                upper=np.array([10, 10, 10]),
            ),
            algorithm=Fides(**algo_options),
        )


@pytest.mark.skipif(not IS_FIDES_INSTALLED, reason="fides not installed.")
def test_fides_stop_after_one_iteration():
    res = minimize(
        fun_and_jac=criterion_and_derivative,
        fun=criterion,
        x0=np.array([1, -5, 3]),
        bounds=Bounds(
            lower=np.array([-10, -10, -10]),
            upper=np.array([10, 10, 10]),
        ),
        algorithm=Fides(stopping_maxiter=1),
    )
    assert not res.success
    assert res.n_iterations == 1


@pytest.mark.skipif(not IS_FIDES_INSTALLED, reason="fides not installed.")
def test_fides_deprecated_option_name_still_works():
    with pytest.warns(FutureWarning, match="Use `subspace_solver='scg'` instead"):
        deprecated: dict[str, Any] = {"trustregion_subspace_dimension": "scg"}
        algo = Fides(**deprecated)
    assert algo == Fides(subspace_solver="scg")
