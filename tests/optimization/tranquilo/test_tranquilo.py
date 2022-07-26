import numpy as np
from estimagic.optimization.optimize import minimize
from estimagic.optimization.tranquilo.tranquilo import tranquilo
from estimagic.optimization.tranquilo.tranquilo import tranquilo_ls
from numpy.testing import assert_array_almost_equal as aaae


def test_internal_tranquilo_scalar_sphere_defaults():
    res = tranquilo(
        criterion=lambda x: x @ x,
        x=np.arange(5),
    )
    aaae(res["solution_x"], np.zeros(5), decimal=5)


def test_internal_tranquilo_ls_sphere_defaults():
    res = tranquilo_ls(
        criterion=lambda x: x,
        x=np.arange(5),
    )
    aaae(res["solution_x"], np.zeros(5), decimal=5)


def test_external_tranquilo_scalar_sphere_defaults():
    res = minimize(
        criterion=lambda x: x @ x,
        params=np.arange(5),
        algorithm="tranquilo",
    )

    aaae(res.params, np.zeros(5), decimal=5)


def test_external_tranquilo_ls_sphere_defaults():
    res = minimize(
        criterion=lambda x: x,
        params=np.arange(5),
        algorithm="tranquilo_ls",
    )

    aaae(res.params, np.zeros(5), decimal=5)
