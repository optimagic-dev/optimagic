import numpy as np
from estimagic.optimization.tranquilo.count_points import get_counter


def test_count_all():
    count_points = get_counter(counter="count_all", bounds=None)
    assert count_points(np.ones((4, 3)), trustregion=None) == 4
