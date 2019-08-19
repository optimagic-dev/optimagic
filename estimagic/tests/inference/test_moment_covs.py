import numpy as np
from numpy.testing import assert_array_almost_equal

from estimagic.inference.moment_covs import average_contribution


def test_average_contribution():
    x, y, z = np.random.randint(1, 50, size=3)
    moment_cond = np.array(range(x * y * z)).reshape(shape=(x, y, z))
    control = np.empty(shape=(y, z))
    control_0_0 = np.sum(np.arange(0, x * y * z, y * z))
    start = -1
    for i in range(y):
        for j in range(z):
            start += 1
            control[i, j] = control_0_0 + start * x
    assert_array_almost_equal(average_contribution(moment_cond), control / x)
