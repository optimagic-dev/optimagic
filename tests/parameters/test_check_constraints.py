import pytest
import numpy as np
from estimagic.exceptions import InvalidParamsError
from estimagic.parameters.check_constraints import check_constraints_are_satisfied


def test_check_constraints_are_satisfied_type_equality():
    param_value = np.array([110, 22, 3])
    param_names = np.array(["a", "b", "c"])

    constraints = [{"type": "equality", "index": [0, 1]}]

    with pytest.raises(InvalidParamsError):
        check_constraints_are_satisfied(constraints, param_value, param_names)


def test_check_constraints_are_satisfied_type_covariance():
    # TODO not sure if its correct
    param_value = np.array([1, 2, 1])
    param_names = np.array(["a", "b", "c"])

    constraints = [{"type": "covariance", "index": [0, 1, 2]}]

    with pytest.raises(InvalidParamsError):
        check_constraints_are_satisfied(constraints, param_value, param_names)


def test_check_constraints_are_satisfied_type_sdcorr():
    param_value = np.array([110, 22, 3])
    param_names = np.array(["a", "b", "c"])

    constraints = [
        # index values based of docs
        # test passes regardless of internal checks of e < or e > or e =, how?
        # Is it because e is a matrix and not a float number
        {"type": "sdcorr", "index": [1, 1, 2]}
    ]

    with pytest.raises(InvalidParamsError):
        check_constraints_are_satisfied(constraints, param_value, param_names)


def test_check_constraints_are_satisfied_type_probability():
    param_value = np.array([0.6, 9, 0.25, 0.25, 1, 1])
    param_names = np.array(["a", "b", "c", "d", "e"])

    constraints = [{"type": "probability", "index": [0, 1, 2, 3, 4]}]

    with pytest.raises(InvalidParamsError):
        check_constraints_are_satisfied(constraints, param_value, param_names)
