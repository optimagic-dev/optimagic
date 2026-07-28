import numpy as np
import pytest
from numpy.testing import assert_array_equal as aae

from optimagic.constraints import (
    Constraint,
    ConstraintSource,
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
    ResolvedDecreasingConstraint,
    ResolvedEqualityConstraint,
    ResolvedFixedConstraint,
    ResolvedFlatCovConstraint,
    ResolvedFlatSDCorrConstraint,
    ResolvedIncreasingConstraint,
    ResolvedLinearConstraint,
    ResolvedPairwiseEqualityConstraint,
    ResolvedProbabilityConstraint,
    _all_none,
    _select_non_none,
)
from optimagic.exceptions import InvalidConstraintError
from optimagic.parameters.constraints.resolution import ResolutionContext
from optimagic.parameters.tree_registry import get_registry


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


# ======================================================================================
# _resolve
# ======================================================================================


def make_context(constraint, position=0, n_params=6):
    """Create a resolution context for a flat numpy params vector."""
    return ResolutionContext(
        helper=np.arange(n_params),
        registry=dict(get_registry(extended=True)),
        param_names=list("abcdef"),
        source=ConstraintSource(constraint=constraint, position=position),
    )


def _empty_selector(x):
    return x[np.array([], dtype=int)]


SINGLE_SELECTOR_CASES = [
    (FixedConstraint, ResolvedFixedConstraint),
    (IncreasingConstraint, ResolvedIncreasingConstraint),
    (DecreasingConstraint, ResolvedDecreasingConstraint),
    (EqualityConstraint, ResolvedEqualityConstraint),
    (ProbabilityConstraint, ResolvedProbabilityConstraint),
]


@pytest.mark.parametrize(("constraint_type", "resolved_type"), SINGLE_SELECTOR_CASES)
def test_resolve_returns_typed_constraint_with_positions(
    constraint_type, resolved_type
):
    constr = constraint_type(selector=lambda x: x[[1, 4]])
    resolved = constr._resolve(make_context(constr, position=3))
    assert isinstance(resolved, resolved_type)
    aae(resolved.index, np.array([1, 4]))
    assert resolved.sources[0].constraint is constr
    assert resolved.sources[0].position == 3


@pytest.mark.parametrize(
    "constraint",
    [
        FixedConstraint(selector=_empty_selector),
        IncreasingConstraint(selector=_empty_selector),
        DecreasingConstraint(selector=_empty_selector),
        EqualityConstraint(selector=_empty_selector),
        ProbabilityConstraint(selector=_empty_selector),
        FlatCovConstraint(selector=_empty_selector),
        FlatSDCorrConstraint(selector=_empty_selector),
        LinearConstraint(selector=_empty_selector, weights=1, value=1),
        PairwiseEqualityConstraint(selectors=[_empty_selector, _empty_selector]),
    ],
    ids=lambda constraint: type(constraint).__name__,
)
def test_resolve_empty_selection_returns_none(constraint):
    assert constraint._resolve(make_context(constraint)) is None


def test_resolve_fixed_constraint_has_no_explicit_value():
    constr = FixedConstraint(selector=lambda x: x[[0, 2]])
    resolved = constr._resolve(make_context(constr))
    assert resolved.value is None


def test_resolve_pairwise_equality_constraint():
    constr = PairwiseEqualityConstraint(
        selectors=[lambda x: x[[0, 2]], lambda x: x[[1, 3]]]
    )
    resolved = constr._resolve(make_context(constr))
    assert isinstance(resolved, ResolvedPairwiseEqualityConstraint)
    aae(resolved.indices[0], np.array([0, 2]))
    aae(resolved.indices[1], np.array([1, 3]))
    assert resolved.sources[0].constraint is constr


def test_resolve_pairwise_equality_constraint_with_different_lengths_raises():
    constr = PairwiseEqualityConstraint(
        selectors=[lambda x: x[[0, 2]], lambda x: x[[1, 3, 5]]]
    )
    with pytest.raises(InvalidConstraintError, match="same length"):
        constr._resolve(make_context(constr))


@pytest.mark.parametrize(
    ("constraint_type", "resolved_type"),
    [
        (FlatCovConstraint, ResolvedFlatCovConstraint),
        (FlatSDCorrConstraint, ResolvedFlatSDCorrConstraint),
    ],
)
def test_resolve_carries_regularization(constraint_type, resolved_type):
    constr = constraint_type(selector=lambda x: x[[0, 1, 2]], regularization=0.1)
    resolved = constr._resolve(make_context(constr))
    assert isinstance(resolved, resolved_type)
    aae(resolved.index, np.array([0, 1, 2]))
    assert resolved.regularization == 0.1


def test_resolve_linear_constraint_broadcasts_scalar_weights():
    constr = LinearConstraint(selector=lambda x: x[[0, 2]], weights=2, value=3)
    resolved = constr._resolve(make_context(constr))
    assert isinstance(resolved, ResolvedLinearConstraint)
    aae(resolved.weights, np.array([2.0, 2.0]))


def test_resolve_linear_constraint_aligns_weight_sequence():
    constr = LinearConstraint(
        selector=lambda x: x[[0, 2, 4]], weights=[1, 2, 3], upper_bound=5
    )
    resolved = constr._resolve(make_context(constr))
    aae(resolved.index, np.array([0, 2, 4]))
    aae(resolved.weights, np.array([1.0, 2.0, 3.0]))


def test_resolve_linear_constraint_fills_absent_bounds_with_sentinels():
    constr = LinearConstraint(selector=lambda x: x[[0, 2]], weights=1, lower_bound=1)
    resolved = constr._resolve(make_context(constr))
    assert resolved.lower_bound == 1
    assert resolved.upper_bound == np.inf
    assert np.isnan(resolved.value)


def test_resolve_linear_constraint_with_misaligned_weights_raises():
    constr = LinearConstraint(selector=lambda x: x[[0, 2]], weights=[1, 2, 3], value=3)
    with pytest.raises(InvalidConstraintError, match="aligned"):
        constr._resolve(make_context(constr))


def test_resolve_nonlinear_constraint_raises():
    constr = NonlinearConstraint(
        selector=lambda x: x[[0, 1]], func=lambda x: x @ x, value=1
    )
    with pytest.raises(NotImplementedError, match="must not be resolved"):
        constr._resolve(make_context(constr))
