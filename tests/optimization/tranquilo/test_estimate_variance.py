import numpy as np
import pytest
from estimagic.optimization.tranquilo.estimate_variance import (
    _estimate_variance_classic,
)
from estimagic.optimization.tranquilo.new_history import History
from estimagic.optimization.tranquilo.tranquilo import Region
from numpy.testing import assert_array_almost_equal as aaae


@pytest.mark.parametrize("model_type", ["scalar", "vector"])
def test_estimate_variance_classic(model_type):
    xs = np.array(
        [
            [0.0, 0.0],  # center with multiple evaluations
            [10, -10],  # far away with multiple evaluations
            [0.1, 0.1],  # close to center with too few evaluations
        ]
    )

    history = History(functype="scalar")
    idxs = history.add_xs(xs)

    repetitions = np.array([5, 5, 2])

    # squaring makes sure variance is not the same across all subsamples
    evals = np.arange(12) ** 2

    history.add_evals(idxs.repeat(repetitions), evals)

    got = _estimate_variance_classic(
        trustregion=Region(center=np.array([0.0, 0.0]), sphere_radius=1.0),
        history=history,
        model_type=model_type,
        max_distance_factor=1.0,
        min_n_evals=4,
    )

    if model_type == "scalar":
        expected = np.var(evals[:5], ddof=1)
    else:
        expected = np.var(evals[:5], ddof=1).reshape(1, 1)

    aaae(got, expected)
