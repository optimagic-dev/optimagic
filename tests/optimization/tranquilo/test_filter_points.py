import numpy as np
import pytest
from estimagic.optimization.tranquilo.filter_points import (
    _scaled_square_features,
)
from estimagic.optimization.tranquilo.region import Region
from estimagic.optimization.tranquilo.tranquilo import State
from estimagic.optimization.tranquilo.history import History
from numpy.testing import assert_equal


@pytest.fixture()
def basic_case():
    x_accepted = np.array([0.16004745, 0.00572722, 0.01158929])
    radius = 0.0125

    xs = np.array(
        [
            [0.15, 0.008, 0.01],
            [0.25, 0.008, 0.01],
            [0.15, 0.108, 0.01],
            [0.15, 0.008, 0.11],
            [0.15961778, -0.07539625, 0.08766385],
            [0.2, 0.00851182, -0.00302887],
            [0.15049526, -0.04199751, 0.00993654],
            [0.13739276, 0.00793654, -0.03838443],
            [0.15046527, 0.00796766, 0.01269269],
            [0.14986784, 0.00809919, 0.00927703],
            [0.12530518, 0.00613383, 0.01349762],
            [0.14987566, 0.0081864, 0.00937541],
            [0.15076496, 0.00570962, 0.01295807],
            [0.15074537, 0.00526659, 0.01240602],
            [0.15069081, 0.00552219, 0.0121367],
            [0.15067245, 0.00559504, 0.01191949],
            [0.15141789, 0.0056498, 0.01210095],
            [0.16317245, 0.00558118, 0.01208116],
            [0.15692245, 0.00559149, 0.01197266],
            [0.15379745, 0.00562833, 0.01182871],
            [0.16004745, 0.00572722, 0.01158929],  # 20
        ]
    )
    indices = np.arange(len(xs))

    trustregion = Region(center=x_accepted, radius=radius)

    state = State(
        trustregion=trustregion,
        model_indices=None,
        model=None,
        index=20,
        x=x_accepted,
        fval=0,
        rho=None,
        accepted=True,
        new_indices=None,
        old_indices_discarded=None,
        old_indices_used=None,
        candidate_index=None,
        candidate_x=None,
        vector_model=None,
    )

    expected_indices = np.array([20, 19, 18, 17, 16, 15, 13, 12, 8, 5, 4, 3, 2, 1, 0])
    expected_xs = np.array(
        [
            [0.16004745, 0.00572722, 0.01158929],
            [0.15379745, 0.00562833, 0.01182871],
            [0.15692245, 0.00559149, 0.01197266],
            [0.16317245, 0.00558118, 0.01208116],
            [0.15141789, 0.0056498, 0.01210095],
            [0.15067245, 0.00559504, 0.01191949],
            [0.15074537, 0.00526659, 0.01240602],
            [0.15076496, 0.00570962, 0.01295807],
            [0.15046527, 0.00796766, 0.01269269],
            [0.2, 0.00851182, -0.00302887],
            [0.15961778, -0.07539625, 0.08766385],
            [0.15, 0.008, 0.11],
            [0.15, 0.108, 0.01],
            [0.25, 0.008, 0.01],
            [0.15, 0.008, 0.01],
        ]
    )

    return xs, indices, state, expected_xs, expected_indices


@pytest.fixture()
def reordered_case(basic_case):
    _, indices, state, expected_xs, expected_indices = basic_case
    state = state._replace(index=13)

    xs = np.array(
        [
            [0.15, 0.008, 0.01],
            [0.25, 0.008, 0.01],
            [0.15, 0.108, 0.01],
            [0.15, 0.008, 0.11],
            [0.15961778, -0.07539625, 0.08766385],
            [0.2, 0.00851182, -0.00302887],
            [0.15049526, -0.04199751, 0.00993654],
            [0.13739276, 0.00793654, -0.03838443],
            [0.15046527, 0.00796766, 0.01269269],
            [0.14986784, 0.00809919, 0.00927703],
            [0.12530518, 0.00613383, 0.01349762],
            [0.14987566, 0.0081864, 0.00937541],
            [0.15076496, 0.00570962, 0.01295807],
            [0.16004745, 0.00572722, 0.01158929],
            [0.15074537, 0.00526659, 0.01240602],
            [0.15069081, 0.00552219, 0.0121367],
            [0.15067245, 0.00559504, 0.01191949],
            [0.15141789, 0.0056498, 0.01210095],
            [0.16317245, 0.00558118, 0.01208116],
            [0.15692245, 0.00559149, 0.01197266],
            [0.15379745, 0.00562833, 0.01182871],
        ]
    )

    expected_indices = np.array([13, 20, 19, 18, 17, 16, 14, 12, 8, 5, 4, 3, 2, 1, 0])

    return xs, indices, state, expected_xs, expected_indices


@pytest.fixture()
def truncated_case(reordered_case):
    _, _, state, expected_xs, _ = reordered_case

    xs = np.array(
        [
            [0.15049526, -0.04199751, 0.00993654],
            [0.13739276, 0.00793654, -0.03838443],
            [0.15046527, 0.00796766, 0.01269269],
            [0.14986784, 0.00809919, 0.00927703],
            [0.12530518, 0.00613383, 0.01349762],
            [0.14987566, 0.0081864, 0.00937541],
            [0.15076496, 0.00570962, 0.01295807],
            [0.16004745, 0.00572722, 0.01158929],
            [0.15074537, 0.00526659, 0.01240602],
            [0.15067245, 0.00559504, 0.01191949],
            [0.15141789, 0.0056498, 0.01210095],
            [0.16317245, 0.00558118, 0.01208116],
            [0.15692245, 0.00559149, 0.01197266],
            [0.15379745, 0.00562833, 0.01182871],
        ]
    )
    indices = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20])

    expected_indices = np.array([13, 20, 19, 18, 17, 16, 14, 12, 8])
    expected_xs = expected_xs[:-6]

    return xs, indices, state, expected_xs, expected_indices


@pytest.fixture()
def sparse_case(reordered_case):
    _, _, state, *_ = reordered_case

    xs = np.array(
        [
            [0.15046527, 0.00796766, 0.01269269],
            [0.14986784, 0.00809919, 0.00927703],
            [0.12530518, 0.00613383, 0.01349762],
            [0.14987566, 0.0081864, 0.00937541],
            [0.15076496, 0.00570962, 0.01295807],
            [0.16004745, 0.00572722, 0.01158929],
            [0.15074537, 0.00526659, 0.01240602],
            [0.15067245, 0.00559504, 0.01191949],
            [0.15141789, 0.0056498, 0.01210095],
            [0.15379745, 0.00562833, 0.01182871],
        ]
    )
    indices = np.array([8, 9, 10, 11, 12, 13, 14, 16, 17, 20])

    expected_indices = np.array([13, 20, 17, 16, 14, 12, 11, 10, 9, 8])
    expected_xs = np.array(
        [
            [0.16004745, 0.00572722, 0.01158929],
            [0.15379745, 0.00562833, 0.01182871],
            [0.15141789, 0.0056498, 0.01210095],
            [0.15067245, 0.00559504, 0.01191949],
            [0.15074537, 0.00526659, 0.01240602],
            [0.15076496, 0.00570962, 0.01295807],
            [0.14987566, 0.0081864, 0.00937541],
            [0.12530518, 0.00613383, 0.01349762],
            [0.14986784, 0.00809919, 0.00927703],
            [0.15046527, 0.00796766, 0.01269269],
        ]
    )

    return xs, indices, state, expected_xs, expected_indices


def test_indices_in_trust_region(basic_case):
    functype = "scalar"
    history = History(functype=functype)

    xs, *_ = basic_case
    x_accepted = np.array([0.16004745, 0.00572722, 0.01158929])
    radius = 0.0125

    trustregion = Region(center=x_accepted, radius=radius)
    x_indices = history.add_xs(xs)
    history.add_evals(x_indices, np.zeros(xs.shape[0]))

    indices_in_tr = history.get_x_indices_in_region(trustregion)

    expected_indices = np.array([0, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    assert_equal(indices_in_tr, expected_indices)


def test_scaled_square_features():
    x = np.arange(4).reshape(2, 2)
    got = _scaled_square_features(x)
    expected = np.array([[0, 0, 1 / 2], [2, 6 / np.sqrt(2), 9 / 2]])
    assert_equal(got, expected)
