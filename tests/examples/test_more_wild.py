import pytest
from estimagic.examples.more_wild import MORE_WILD_PROBLEMS


@pytest.mark.parametrize("name, specification", list(MORE_WILD_PROBLEMS.items()))
def test_function_at_start_x(name, specification):
    _criterion = specification["criterion"]
    _x = specification["start_x"]
    _contributions = _criterion(_x)
    calculated = _contributions @ _contributions
    expected = specification["start_criterion"]
    assert calculated == expected
