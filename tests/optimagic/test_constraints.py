import pytest
from optimagic.constraints import (
    Constraint,
    DecreasingConstraint,
    EqualityConstraint,
    FixedConstraint,
    FlatCovConstraint,
    FlatSDCorrConstraint,
    IncreasingConstraint,
    LinearConstraint,
    NonlinearConstraint,
    PairwiseEqualityConstraint,
    ProbabilityConstraint,
    _all_none,
    _select_non_none,
)
from optimagic.exceptions import InvalidConstraintError


@pytest.fixture
def dummy_func():
    return lambda x: x


def test_fixed_constraint(dummy_func):
    constr = FixedConstraint(selector=dummy_func)
    dict_repr = {"type": "fixed", "selector": dummy_func}
    assert constr._to_dict() == dict_repr
    assert isinstance(constr, Constraint)


def test_increasing_constraint(dummy_func):
    constr = IncreasingConstraint(selector=dummy_func)
    dict_repr = {"type": "increasing", "selector": dummy_func}
    assert constr._to_dict() == dict_repr
    assert isinstance(constr, Constraint)


def test_decreasing_constraint(dummy_func):
    constr = DecreasingConstraint(selector=dummy_func)
    dict_repr = {"type": "decreasing", "selector": dummy_func}
    assert constr._to_dict() == dict_repr
    assert isinstance(constr, Constraint)


def test_equality_constraint(dummy_func):
    constr = EqualityConstraint(selector=dummy_func)
    dict_repr = {"type": "equality", "selector": dummy_func}
    assert constr._to_dict() == dict_repr
    assert isinstance(constr, Constraint)


def test_pairwise_equality_constraint(dummy_func):
    constr = PairwiseEqualityConstraint(selectors=[dummy_func, dummy_func])
    dict_repr = {"type": "pairwise_equality", "selectors": [dummy_func, dummy_func]}
    assert constr._to_dict() == dict_repr
    assert isinstance(constr, Constraint)


def test_probability_constraint(dummy_func):
    constr = ProbabilityConstraint(selector=dummy_func)
    dict_repr = {"type": "probability", "selector": dummy_func}
    assert constr._to_dict() == dict_repr
    assert isinstance(constr, Constraint)


def test_covariance_constraint(dummy_func):
    constr = FlatCovConstraint(selector=dummy_func)
    dict_repr = {"type": "covariance", "selector": dummy_func, "regularization": 0.0}
    assert constr._to_dict() == dict_repr
    assert isinstance(constr, Constraint)


def test_sdcorr_constraint(dummy_func):
    constr = FlatSDCorrConstraint(selector=dummy_func)
    dict_repr = {"type": "sdcorr", "selector": dummy_func, "regularization": 0.0}
    assert constr._to_dict() == dict_repr
    assert isinstance(constr, Constraint)


def test_linear_constraint_with_value(dummy_func):
    constr = LinearConstraint(selector=dummy_func, value=2.1, weights=[1, 2])
    dict_repr = {
        "type": "linear",
        "selector": dummy_func,
        "value": 2.1,
        "weights": [1, 2],
    }
    assert constr._to_dict() == dict_repr
    assert isinstance(constr, Constraint)


def test_linear_constraint_with_bounds(dummy_func):
    constr = LinearConstraint(
        selector=dummy_func, lower_bound=1.0, upper_bound=2.0, weights=[1, 2]
    )
    dict_repr = {
        "type": "linear",
        "selector": dummy_func,
        "lower_bound": 1.0,
        "upper_bound": 2.0,
        "weights": [1, 2],
    }
    assert constr._to_dict() == dict_repr


def test_linear_constraint_with_bounds_and_value(dummy_func):
    msg = "'value' cannot be used with 'lower_bound' or 'upper_bound'."
    with pytest.raises(InvalidConstraintError, match=msg):
        LinearConstraint(
            selector=dummy_func,
            lower_bound=1.0,
            upper_bound=2.0,
            value=2.1,
            weights=[1, 2],
        )


def test_linear_constraint_with_nothing(dummy_func):
    msg = "At least one of 'lower_bound', 'upper_bound', or 'value' must be non-None."
    with pytest.raises(InvalidConstraintError, match=msg):
        LinearConstraint(selector=dummy_func, weights=[1, 2])


def test_nonlinear_constraint_with_value(dummy_func):
    constr = NonlinearConstraint(selector=dummy_func, value=2.1, func=dummy_func)
    dict_repr = {
        "type": "nonlinear",
        "selector": dummy_func,
        "value": 2.1,
        "func": dummy_func,
        "tol": 1e-5,
    }
    assert constr._to_dict() == dict_repr
    assert isinstance(constr, Constraint)


def test_nonlinear_constraint_with_bounds(dummy_func):
    constr = NonlinearConstraint(
        selector=dummy_func, lower_bound=1.0, upper_bound=2.0, func=dummy_func
    )
    dict_repr = {
        "type": "nonlinear",
        "selector": dummy_func,
        "func": dummy_func,
        "lower_bounds": 1.0,
        "upper_bounds": 2.0,
        "tol": 1e-5,
    }
    assert constr._to_dict() == dict_repr


def test_nonlinear_constraint_with_bounds_and_value(dummy_func):
    msg = "'value' cannot be used with 'lower_bound' or 'upper_bound'."
    with pytest.raises(InvalidConstraintError, match=msg):
        NonlinearConstraint(
            selector=dummy_func,
            lower_bound=1.0,
            upper_bound=2.0,
            value=2.1,
            func=dummy_func,
        )


def test_nonlinear_constraint_with_nothing(dummy_func):
    msg = "At least one of 'lower_bound', 'upper_bound', or 'value' must be non-None."
    with pytest.raises(InvalidConstraintError, match=msg):
        NonlinearConstraint(selector=dummy_func, func=dummy_func)


def test_all_none():
    assert _all_none(None, None, None)
    assert not _all_none(None, 1, None)


def test_select_non_none():
    assert _select_non_none(a=None, b=None, c=None) == {}
    assert _select_non_none(a=None, b=1, c=None) == {"b": 1}
    assert _select_non_none(a=None, b=None, c=2) == {"c": 2}
    assert _select_non_none(a=1, b=2, c=3) == {"a": 1, "b": 2, "c": 3}
