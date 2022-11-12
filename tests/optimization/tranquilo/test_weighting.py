import numpy as np
from estimagic.optimization.tranquilo.weighting import get_weighter


def test_no_weighting():
    weight_points = get_weighter(weighter="no_weights", bounds=None)
    assert weight_points(np.ones((4, 3)), trustregion=None) is None
