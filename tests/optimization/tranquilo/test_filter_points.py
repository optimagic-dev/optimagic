from tranquilo.optimization.tranquilo.filter_points import get_sample_filter
from tranquilo.optimization.tranquilo.tranquilo import State
from tranquilo.optimization.tranquilo.region import Region
from numpy.testing import assert_array_equal as aae
import pytest
import numpy as np


@pytest.fixture()
def state():
    out = State(
        trustregion=Region(center=np.ones(2), radius=0.3),
        model_indices=None,
        model=None,
        vector_model=None,
        candidate_index=5,
        candidate_x=np.array([1.1, 1.2]),
        index=2,
        x=np.ones(2),
        fval=15,
        rho=None,
        accepted=True,
        old_indices_used=None,
        old_indices_discarded=None,
        new_indices=None,
        step_length=0.1,
        relative_step_length=0.1 / 0.3,
    )
    return out


def test_discard_all(state):
    filter = get_sample_filter("discard_all")
    xs = np.arange(10).reshape(5, 2)
    indices = np.arange(5)
    got_xs, got_idxs = filter(xs=xs, indices=indices, state=state)
    expected_xs = np.ones((1, 2))
    aae(got_xs, expected_xs)
    aae(got_idxs, np.array([2]))


def test_keep_all():
    filter = get_sample_filter("keep_all")
    xs = np.arange(10).reshape(5, 2)
    indices = np.arange(5)
    got_xs, got_idxs = filter(xs=xs, indices=indices, state=None)
    aae(got_xs, xs)
    aae(got_idxs, indices)
