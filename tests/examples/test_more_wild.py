import numpy as np
import pytest

from estimagic.examples.more_wild import get_start_points_mancino
from estimagic.examples.more_wild import MORE_WILD_PROBLEMS


@pytest.mark.parametrize("name, specification", list(MORE_WILD_PROBLEMS.items()))
def test_more_wild_function_at_start_x(name, specification):
    _criterion = specification["criterion"]
    _x = specification["start_x"]
    _contributions = _criterion(_x)
    calculated = _contributions @ _contributions
    expected = specification["start_criterion"]
    assert np.allclose(calculated, expected)


def test_get_start_points_mancino():
    expected = (np.array([102.4824, 96.3335, 90.4363, 84.7852, 79.3747]),)
    result = get_start_points_mancino(5)
    assert np.allclose(expected, result)
