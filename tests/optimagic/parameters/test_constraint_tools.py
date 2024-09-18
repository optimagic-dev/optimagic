import pytest

import optimagic as om
from optimagic.exceptions import InvalidParamsError
from optimagic.parameters.constraint_tools import check_constraints, count_free_params


def test_count_free_params_no_constraints():
    params = {"a": 1, "b": 2, "c": [3, 3]}
    assert count_free_params(params) == 4


def test_count_free_params_with_constraints():
    params = {"a": 1, "b": 2, "c": [3, 3]}
    constraints = om.EqualityConstraint(lambda x: x["c"])
    assert count_free_params(params, constraints=constraints) == 3


def test_check_constraints():
    params = {"a": 1, "b": 2, "c": [3, 4]}
    constraints = om.EqualityConstraint(lambda x: x["c"])

    with pytest.raises(InvalidParamsError):
        check_constraints(params, constraints=constraints)
