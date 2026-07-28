"""Characterization tests for constraint processing at the get_converter seam.

These tests pin the behavior of the whole constraint pipeline (selector processing,
checking, consolidation and space conversion) at its stable outer boundary. They are
deliberately independent of the internal representation of constraints, so they must
stay green, unchanged, throughout the constraints refactoring.

The golden values in ``_EXPECTED`` were generated from the pre-refactoring
implementation. If a change requires updating one of these numbers, that change alters
the internal reparametrization and needs explicit justification; do not simply
regenerate the values.

The tests cover four invariants for a corpus of constraint sets:

1. Golden internal parameters: values, bounds and free_mask after conversion.
2. Round trips: ``params_to_internal`` and ``params_from_internal`` are inverse to
   each other, at the start values and at randomly sampled internal points.
3. Feasibility: ``params_from_internal(x)`` satisfies the original constraints for
   any internal ``x`` within the internal bounds. This is the core correctness
   property of the reparametrization approach.
4. Derivatives: ``derivative_to_internal`` coincides with a numerical Jacobian of
   ``params_from_internal``.

In addition, the first-error behavior for misspecified constraints and violated start
values is pinned to exception types (not messages).

Nonlinear constraints are out of scope: they are passed on to the optimizer rather than
reparametrized, so they never reach the reparametrization pipeline exercised here.

"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from pybaum import tree_just_flatten

import optimagic as om
from optimagic import first_derivative
from optimagic.deprecations import pre_process_constraints
from optimagic.exceptions import InvalidConstraintError, InvalidParamsError
from optimagic.parameters.conversion import get_converter
from optimagic.parameters.tree_registry import get_registry
from optimagic.typing import AggregationLevel
from optimagic.utilities import cov_params_to_matrix, get_rng, sdcorr_params_to_matrix

inf = float("inf")


@dataclass(frozen=True)
class Case:
    """One corpus entry: a params/constraints pair plus feasibility information."""

    params: Any
    """User provided params (numpy array, DataFrame with value column, or pytree)."""

    constraints: list[Any]
    """List of user provided constraint objects."""

    bounds: om.Bounds | None
    """User provided bounds or None."""

    checks: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    """Feasibility checks as (kind, payload) tuples. Positions refer to the flattened
    params. They are used to verify that any output of params_from_internal satisfies
    the original constraints."""

    @property
    def start_flat(self) -> np.ndarray:
        """The start params flattened to optimagic's internal flat order.

        Derived from ``params`` rather than spelled out per case: flattening is not what
        these tests exercise, so the flat start values should not be duplicated by hand.
        """
        registry = get_registry(extended=True)
        return np.array(tree_just_flatten(self.params, registry=registry), dtype=float)


def _all_but_hours(p):
    return {
        "work": {"hourly_wage": p["work"]["hourly_wage"]},
        "time_budget": p["time_budget"],
    }


CASES = {
    "fixed_at_start": Case(
        params=np.array([1.0, 1.5, 4.5]),
        constraints=[om.FixedConstraint(selector=lambda x: x[0])],
        bounds=None,
        checks=[("fixed", {"positions": [0]})],
    ),
    "equality": Case(
        params=np.array([1.0, 1.0, 1.0]),
        constraints=[om.EqualityConstraint(selector=lambda x: x[[0, 1, 2]])],
        bounds=None,
        checks=[("equality", {"positions": [0, 1, 2]})],
    ),
    "pairwise_equality": Case(
        params=np.array([2.0, 2.0, 3.0]),
        constraints=[
            om.PairwiseEqualityConstraint(selectors=[lambda x: x[0], lambda x: x[1]])
        ],
        bounds=None,
        checks=[("pairwise_equality", {"positions_list": [[0], [1]]})],
    ),
    "increasing": Case(
        params=np.array([1.0, 2.0, 3.0]),
        constraints=[om.IncreasingConstraint(selector=lambda x: x[[1, 2]])],
        bounds=None,
        checks=[("increasing", {"positions": [1, 2]})],
    ),
    "decreasing": Case(
        params=np.array([3.0, 2.0, 1.0]),
        constraints=[om.DecreasingConstraint(selector=lambda x: x[[0, 1]])],
        bounds=None,
        checks=[("decreasing", {"positions": [0, 1]})],
    ),
    "linear_value": Case(
        params=np.array([2.0, 1.0, 3.0]),
        constraints=[
            om.LinearConstraint(selector=lambda x: x[[0, 1]], value=4, weights=[1, 2])
        ],
        bounds=None,
        checks=[("linear", {"positions": [0, 1], "weights": [1, 2], "value": 4})],
    ),
    "linear_bounds": Case(
        params=np.array([1.0, 2.0, 3.0]),
        constraints=[
            om.LinearConstraint(
                selector=lambda x: x[[0, 1, 2]],
                lower_bound=0,
                upper_bound=8,
                weights=[1, 1, 1],
            )
        ],
        bounds=None,
        checks=[
            (
                "linear",
                {
                    "positions": [0, 1, 2],
                    "weights": [1, 1, 1],
                    "lower_bound": 0,
                    "upper_bound": 8,
                },
            )
        ],
    ),
    "overlapping_linear": Case(
        params=np.array([2.0, 1.0, 3.0]),
        constraints=[
            om.LinearConstraint(
                selector=lambda x: x[[0, 1]], weights=[1, 1], lower_bound=1
            ),
            om.LinearConstraint(
                selector=lambda x: x[[1, 2]], weights=[1, 1], upper_bound=10
            ),
        ],
        bounds=None,
        checks=[
            ("linear", {"positions": [0, 1], "weights": [1, 1], "lower_bound": 1}),
            ("linear", {"positions": [1, 2], "weights": [1, 1], "upper_bound": 10}),
        ],
    ),
    "linear_with_fixed": Case(
        params=np.array([2.0, 1.0, 3.0]),
        constraints=[
            om.FixedConstraint(selector=lambda x: x[0]),
            om.LinearConstraint(selector=lambda x: x[[0, 1]], value=4, weights=[1, 2]),
        ],
        bounds=None,
        checks=[
            ("fixed", {"positions": [0]}),
            ("linear", {"positions": [0, 1], "weights": [1, 2], "value": 4}),
        ],
    ),
    "linear_with_equality": Case(
        params=np.array([1.0, 1.0, 4.0]),
        constraints=[
            om.EqualityConstraint(selector=lambda x: x[[0, 1]]),
            om.LinearConstraint(
                selector=lambda x: x[[0, 1, 2]], weights=[1, 1, 1], value=6
            ),
        ],
        bounds=None,
        checks=[
            ("equality", {"positions": [0, 1]}),
            ("linear", {"positions": [0, 1, 2], "weights": [1, 1, 1], "value": 6}),
        ],
    ),
    "linear_negative_weights": Case(
        params=np.array([1.0, 2.0, 1.0]),
        constraints=[
            om.LinearConstraint(
                selector=lambda x: x[[0, 1]], weights=[-1, 1], lower_bound=0
            )
        ],
        bounds=None,
        checks=[
            ("linear", {"positions": [0, 1], "weights": [-1, 1], "lower_bound": 0})
        ],
    ),
    "linear_with_param_bounds": Case(
        params=np.array([1.0, 2.0, 3.0]),
        constraints=[
            om.LinearConstraint(
                selector=lambda x: x[[0, 1]], weights=[1, 1], upper_bound=5
            )
        ],
        bounds=om.Bounds(lower=np.array([0.0, -inf, -inf]), upper=np.full(3, inf)),
        checks=[
            ("linear", {"positions": [0, 1], "weights": [1, 1], "upper_bound": 5}),
            ("bounds", {"lower": [0.0, -inf, -inf], "upper": [inf, inf, inf]}),
        ],
    ),
    "duplicate_linear": Case(
        params=np.array([2.0, 1.0, 3.0]),
        constraints=[
            om.LinearConstraint(selector=lambda x: x[[0, 1]], weights=[1, 2], value=4),
            om.LinearConstraint(selector=lambda x: x[[0, 1]], weights=[1, 2], value=4),
        ],
        bounds=None,
        checks=[("linear", {"positions": [0, 1], "weights": [1, 2], "value": 4})],
    ),
    "probability": Case(
        params=np.array([0.8, 0.2, 3.0]),
        constraints=[om.ProbabilityConstraint(selector=lambda x: x[[0, 1]])],
        bounds=None,
        checks=[("probability", {"positions": [0, 1]})],
    ),
    "covariance": Case(
        params=np.array([1, -0.2, 1.2, -0.2, 0.1, 1.3, 0.1, -0.05, 0.2, 1, 10.0]),
        constraints=[om.FlatCovConstraint(selector=lambda x: x[:10])],
        bounds=None,
        checks=[("covariance", {"positions": list(range(10))})],
    ),
    "covariance_regularized": Case(
        params=np.array([1, 0.1, 2, 0.2, 0.3, 3.0]),
        constraints=[
            om.FlatCovConstraint(selector=lambda x: x[:6], regularization=0.1)
        ],
        bounds=None,
        checks=[("covariance", {"positions": list(range(6))})],
    ),
    "uncorrelated_covariance": Case(
        params=np.array([1, 0, 4, 0, 0, 9, 10.0]),
        constraints=[
            om.FlatCovConstraint(selector=lambda x: x[:6]),
            om.FixedConstraint(selector=lambda x: x[[1, 3, 4]]),
        ],
        bounds=None,
        checks=[
            ("covariance", {"positions": list(range(6))}),
            ("fixed", {"positions": [1, 3, 4]}),
        ],
    ),
    # A covariance block that does not start at flat position 0. Every other cov/sdcorr
    # case sits at position 0, which hides bugs in the block-local vs global index
    # handling of the covariance/sdcorr code paths.
    "shifted_covariance": Case(
        params=np.array([7.0, 1.0, 0.1, 2.0, 0.2, 0.3, 3.0]),
        constraints=[om.FlatCovConstraint(selector=lambda x: x[1:7])],
        bounds=None,
        checks=[("covariance", {"positions": [1, 2, 3, 4, 5, 6]})],
    ),
    # uncorrelated_covariance for a block that does not start at flat position 0. The
    # off-diagonals (global indices 2, 4, 5) are fixed to 0, so the constraint is
    # simplified to bounds on the variances. This used to be wrongly rejected because
    # the simplification compared block-local positions against global indices.
    "shifted_uncorrelated_covariance": Case(
        params=np.array([99.0, 1.0, 0.0, 4.0, 0.0, 0.0, 9.0]),
        constraints=[
            om.FlatCovConstraint(selector=lambda x: x[1:7]),
            om.FixedConstraint(selector=lambda x: x[[2, 4, 5]]),
        ],
        bounds=None,
        checks=[
            ("covariance", {"positions": [1, 2, 3, 4, 5, 6]}),
            ("fixed", {"positions": [2, 4, 5]}),
        ],
    ),
    "sdcorr": Case(
        params=np.array([2, 1.5, 3, 0.2, 0.15, 0.33, 10.0]),
        constraints=[om.FlatSDCorrConstraint(selector=lambda x: x[:6])],
        bounds=None,
        checks=[("sdcorr", {"positions": list(range(6))})],
    ),
    # A 2-dimensional sdcorr constraint is enforced by bounds instead of a kernel
    # transformation: the standard deviations get a lower bound of 0 and the single
    # correlation is bounded to [-1, 1].
    "sdcorr_simplified": Case(
        params=np.array([2.0, 3.0, 0.1, 5.0]),
        constraints=[om.FlatSDCorrConstraint(selector=lambda x: x[:3])],
        bounds=None,
        checks=[("sdcorr", {"positions": [0, 1, 2]})],
    ),
    # sdcorr counterpart of "uncorrelated_covariance": with all correlations fixed to 0
    # the constraint is simplified to bounds on the standard deviations.
    "uncorrelated_sdcorr": Case(
        params=np.array([2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 5.0]),
        constraints=[
            om.FlatSDCorrConstraint(selector=lambda x: x[:6]),
            om.FixedConstraint(selector=lambda x: x[[3, 4, 5]]),
        ],
        bounds=None,
        checks=[
            ("sdcorr", {"positions": [0, 1, 2, 3, 4, 5]}),
            ("fixed", {"positions": [3, 4, 5]}),
        ],
    ),
    "fixed_and_increasing": Case(
        params=np.array([1.0, 2.0, 3.0, 4.0, 1.0]),
        constraints=[
            om.IncreasingConstraint(selector=lambda x: x[[0, 1, 2, 3]]),
            om.FixedConstraint(selector=lambda x: x[2]),
        ],
        bounds=None,
        checks=[
            ("increasing", {"positions": [0, 1, 2, 3]}),
            ("fixed", {"positions": [2]}),
        ],
    ),
    "probability_and_pairwise": Case(
        params=np.array([0.1, 0.9, 0.9, 0.1]),
        constraints=[
            om.PairwiseEqualityConstraint(
                selectors=[lambda x: x[[0, 1]], lambda x: x[[3, 2]]]
            ),
            om.ProbabilityConstraint(selector=lambda x: x[[0, 1]]),
        ],
        bounds=None,
        checks=[
            ("pairwise_equality", {"positions_list": [[0, 1], [3, 2]]}),
            ("probability", {"positions": [0, 1]}),
        ],
    ),
    "dataframe_params": Case(
        params=pd.DataFrame({"value": [0.8, 0.2, 3.0]}),
        constraints=[om.ProbabilityConstraint(selector=lambda p: p.loc[[0, 1]])],
        bounds=None,
        checks=[("probability", {"positions": [0, 1]})],
    ),
    # flat order of the pytree: work_hourly_wage, work_hours, time_budget
    "pytree_params": Case(
        params={
            "work": {"hourly_wage": 25.5, "hours": 2000.0},
            "time_budget": 61320.0,
        },
        constraints=[
            om.FixedConstraint(selector=_all_but_hours),
            om.IncreasingConstraint(
                selector=lambda p: [p["work"]["hours"], p["time_budget"]]
            ),
        ],
        bounds=om.Bounds(lower={"work": {"hourly_wage": -inf, "hours": 0.0}}),
        checks=[
            ("fixed", {"positions": [0, 2]}),
            ("increasing", {"positions": [1, 2]}),
            ("bounds", {"lower": [-inf, 0.0, -inf], "upper": [inf, inf, inf]}),
        ],
    ),
}


@dataclass(frozen=True)
class ExpectedInternal:
    """Golden internal optimization parameters for one case.

    These describe the reparametrized problem that optimagic actually optimizes after
    all constraints have been applied to the start params: the vector of free internal
    parameters, their box bounds, and how they map back to the external parameters.
    They are internal quantities, not user facing params.
    """

    values: list[float]
    """Values of the free internal parameters at the start params."""

    lower_bounds: list[float]
    """Lower bounds of the internal parameters, aligned element-wise with ``values``."""

    upper_bounds: list[float]
    """Upper bounds of the internal parameters, aligned element-wise with ``values``."""

    free_mask: list[bool]
    """One flag per flattened external parameter: True where it maps to a free internal
    parameter and False where a constraint pins it. Its length is the number of external
    parameters, which can exceed ``len(values)``.
    """


# Golden internal parameters, generated from the pre-refactoring implementation.
# Do not regenerate; see module docstring.
_EXPECTED = {
    "fixed_at_start": ExpectedInternal(
        values=[1.5, 4.5],
        lower_bounds=[-inf, -inf],
        upper_bounds=[inf, inf],
        free_mask=[False, True, True],
    ),
    "equality": ExpectedInternal(
        values=[1.0],
        lower_bounds=[-inf],
        upper_bounds=[inf],
        free_mask=[True, False, False],
    ),
    "pairwise_equality": ExpectedInternal(
        values=[2.0, 3.0],
        lower_bounds=[-inf, -inf],
        upper_bounds=[inf, inf],
        free_mask=[True, False, True],
    ),
    "increasing": ExpectedInternal(
        values=[1.0, 2.0, -1.0],
        lower_bounds=[-inf, -inf, -inf],
        upper_bounds=[inf, inf, 0.0],
        free_mask=[True, True, True],
    ),
    "decreasing": ExpectedInternal(
        values=[3.0, 1.0, 1.0],
        lower_bounds=[-inf, 0.0, -inf],
        upper_bounds=[inf, inf, inf],
        free_mask=[True, True, True],
    ),
    "linear_value": ExpectedInternal(
        values=[2.0, 3.0],
        lower_bounds=[-inf, -inf],
        upper_bounds=[inf, inf],
        free_mask=[True, False, True],
    ),
    "linear_bounds": ExpectedInternal(
        values=[1.0, 2.0, 6.0],
        lower_bounds=[-inf, -inf, 0.0],
        upper_bounds=[inf, inf, 8.0],
        free_mask=[True, True, True],
    ),
    "overlapping_linear": ExpectedInternal(
        values=[2.0, 3.0, 4.0],
        lower_bounds=[-inf, 1.0, -inf],
        upper_bounds=[inf, inf, 10.0],
        free_mask=[True, True, True],
    ),
    "linear_with_fixed": ExpectedInternal(
        values=[3.0],
        lower_bounds=[-inf],
        upper_bounds=[inf],
        free_mask=[False, False, True],
    ),
    "linear_with_equality": ExpectedInternal(
        values=[1.0],
        lower_bounds=[-inf],
        upper_bounds=[inf],
        free_mask=[True, False, False],
    ),
    "linear_negative_weights": ExpectedInternal(
        values=[1.0, -1.0, 1.0],
        lower_bounds=[-inf, -inf, -inf],
        upper_bounds=[inf, 0.0, inf],
        free_mask=[True, True, True],
    ),
    "linear_with_param_bounds": ExpectedInternal(
        values=[3.0, 1.0, 3.0],
        lower_bounds=[-inf, 0.0, -inf],
        upper_bounds=[5.0, inf, inf],
        free_mask=[True, True, True],
    ),
    "duplicate_linear": ExpectedInternal(
        values=[2.0, 3.0],
        lower_bounds=[-inf, -inf],
        upper_bounds=[inf, inf],
        free_mask=[True, False, True],
    ),
    "probability": ExpectedInternal(
        values=[4.0, 3.0],
        lower_bounds=[0.0, -inf],
        upper_bounds=[inf, inf],
        free_mask=[True, False, True],
    ),
    "covariance": ExpectedInternal(
        values=[
            1.0,
            -0.2,
            1.0770329614269007,
            -0.2,
            0.05570860145311556,
            1.1211139780254895,
            0.1,
            -0.02785430072655778,
            0.19761748446677013,
            0.9747673916191802,
            10.0,
        ],
        lower_bounds=[
            0.0,
            -inf,
            0.0,
            -inf,
            -inf,
            0.0,
            -inf,
            -inf,
            -inf,
            0.0,
            -inf,
        ],
        upper_bounds=[inf] * 11,
        free_mask=[True] * 11,
    ),
    "covariance_regularized": ExpectedInternal(
        values=[
            1.0,
            0.1,
            1.4106735979665885,
            0.2,
            0.19848673740233402,
            1.708977183895495,
        ],
        lower_bounds=[
            0.31622776601683794,
            -inf,
            0.31622776601683794,
            -inf,
            -inf,
            0.31622776601683794,
        ],
        upper_bounds=[inf] * 6,
        free_mask=[True] * 6,
    ),
    "uncorrelated_covariance": ExpectedInternal(
        values=[1.0, 4.0, 9.0, 10.0],
        lower_bounds=[0.0, 0.0, 0.0, -inf],
        upper_bounds=[inf, inf, inf, inf],
        free_mask=[True, False, True, False, False, True, True],
    ),
    "shifted_covariance": ExpectedInternal(
        values=[
            7.0,
            1.0,
            0.1,
            1.4106735979665885,
            0.2,
            0.19848673740233402,
            1.708977183895495,
        ],
        lower_bounds=[-inf, 0.0, -inf, 0.0, -inf, -inf, 0.0],
        upper_bounds=[inf] * 7,
        free_mask=[True] * 7,
    ),
    "shifted_uncorrelated_covariance": ExpectedInternal(
        values=[99.0, 1.0, 4.0, 9.0],
        lower_bounds=[-inf, 0.0, 0.0, 0.0],
        upper_bounds=[inf, inf, inf, inf],
        free_mask=[True, True, False, True, False, False, True],
    ),
    "sdcorr": ExpectedInternal(
        values=[
            2.0,
            0.30000000000000004,
            1.469693845669907,
            0.44999999999999996,
            0.9185586535436917,
            2.820239351544475,
            10.0,
        ],
        lower_bounds=[0.0, -inf, 0.0, -inf, -inf, 0.0, -inf],
        upper_bounds=[inf] * 7,
        free_mask=[True] * 7,
    ),
    "sdcorr_simplified": ExpectedInternal(
        values=[2.0, 3.0, 0.1, 5.0],
        lower_bounds=[0.0, 0.0, -1.0, -inf],
        upper_bounds=[inf, inf, 1.0, inf],
        free_mask=[True, True, True, True],
    ),
    "uncorrelated_sdcorr": ExpectedInternal(
        values=[2.0, 3.0, 4.0, 5.0],
        lower_bounds=[0.0, 0.0, 0.0, -inf],
        upper_bounds=[inf, inf, inf, inf],
        free_mask=[True, True, True, False, False, False, True],
    ),
    "fixed_and_increasing": ExpectedInternal(
        values=[-1.0, 2.0, 4.0, 1.0],
        lower_bounds=[-inf, -inf, 3.0, -inf],
        upper_bounds=[0.0, 3.0, inf, inf],
        free_mask=[True, True, False, True, True],
    ),
    "probability_and_pairwise": ExpectedInternal(
        values=[0.11111111111111112],
        lower_bounds=[0.0],
        upper_bounds=[inf],
        free_mask=[True, False, False, False],
    ),
    "dataframe_params": ExpectedInternal(
        values=[4.0, 3.0],
        lower_bounds=[0.0, -inf],
        upper_bounds=[inf, inf],
        free_mask=[True, False, True],
    ),
    "pytree_params": ExpectedInternal(
        values=[2000.0],
        lower_bounds=[0.0],
        upper_bounds=[61320.0],
        free_mask=[False, True, False],
    ),
}


def _get_converter_and_internal(case):
    return get_converter(
        params=case.params,
        constraints=pre_process_constraints(case.constraints),
        bounds=case.bounds,
        func_eval=1.0,
        solver_type=AggregationLevel.SCALAR,
    )


def _assert_feasible(external, start_flat, checks):
    """Check that flat external params satisfy the original constraints."""
    tol = 1e-8
    for kind, info in checks:
        if kind == "bounds":
            assert np.all(external >= np.array(info["lower"]) - tol)
            assert np.all(external <= np.array(info["upper"]) + tol)
            continue

        if kind == "pairwise_equality":
            selections = [external[pos] for pos in info["positions_list"]]
            for sel in selections[1:]:
                aaae(sel, selections[0])
            continue

        sel = external[info["positions"]]
        if kind == "fixed":
            aaae(sel, start_flat[info["positions"]])
        elif kind == "equality":
            aaae(sel, np.full(len(sel), sel[0]))
        elif kind == "increasing":
            assert np.all(np.diff(sel) >= -tol)
        elif kind == "decreasing":
            assert np.all(np.diff(sel) <= tol)
        elif kind == "linear":
            weighted_sum = np.array(info["weights"]) @ sel
            if "value" in info:
                assert abs(weighted_sum - info["value"]) <= tol
            if "lower_bound" in info:
                assert weighted_sum >= info["lower_bound"] - tol
            if "upper_bound" in info:
                assert weighted_sum <= info["upper_bound"] + tol
        elif kind == "probability":
            assert abs(sel.sum() - 1) <= tol
            assert np.all(sel >= -tol)
        elif kind in ("covariance", "sdcorr"):
            if kind == "covariance":
                matrix = cov_params_to_matrix(sel)
            else:
                matrix = sdcorr_params_to_matrix(sel)
            assert np.all(np.linalg.eigvalsh(matrix) >= -tol)
        else:
            raise ValueError(f"Invalid check kind: {kind}")


def _sample_feasible_internal_values(internal, rng, n_points):
    """Sample internal parameter vectors that respect the internal bounds."""
    values = np.asarray(internal.values, dtype=float)
    noise = rng.uniform(-0.3, 0.3, size=(n_points, len(values)))
    sampled = values + noise
    lower = np.asarray(internal.lower_bounds, dtype=float)
    upper = np.asarray(internal.upper_bounds, dtype=float)
    return np.clip(sampled, lower + 1e-10, upper - 1e-10)


PARAMETRIZATION = list(CASES)


@pytest.mark.parametrize("name", PARAMETRIZATION)
def test_golden_internal_params(name):
    case = CASES[name]
    expected = _EXPECTED[name]
    _, internal = _get_converter_and_internal(case)

    aaae(internal.values, expected.values)
    aaae(internal.lower_bounds, expected.lower_bounds)
    aaae(internal.upper_bounds, expected.upper_bounds)
    assert internal.free_mask.tolist() == expected.free_mask
    assert len(internal.values) == sum(expected.free_mask)


@pytest.mark.parametrize("name", PARAMETRIZATION)
def test_round_trip_at_start_params(name):
    case = CASES[name]
    converter, internal = _get_converter_and_internal(case)

    aaae(converter.params_to_internal(case.params), internal.values)
    aaae(
        converter.params_from_internal(internal.values, return_type="flat"),
        case.start_flat,
    )


@pytest.mark.parametrize("name", PARAMETRIZATION)
def test_feasibility_and_round_trip_at_random_internal_params(name):
    case = CASES[name]
    converter, internal = _get_converter_and_internal(case)
    rng = get_rng(seed=abs(hash(name)) % 2**32)
    start_flat = np.array(case.start_flat)

    for x in _sample_feasible_internal_values(internal, rng, n_points=20):
        tree, flat = converter.params_from_internal(x, return_type="tree_and_flat")
        _assert_feasible(flat, start_flat, case.checks)
        aaae(converter.params_to_internal(tree), x)


@pytest.mark.parametrize("name", PARAMETRIZATION)
def test_derivative_to_internal_matches_numerical_jacobian(name):
    case = CASES[name]
    converter, internal = _get_converter_and_internal(case)

    numerical = first_derivative(
        lambda x: converter.params_from_internal(x, return_type="flat"),
        internal.values,
    ).derivative

    calculated = converter.derivative_to_internal(
        np.eye(len(case.start_flat)), internal.values, jac_is_flat=True
    )

    aaae(calculated, numerical)


def test_constraints_with_empty_selections_are_dropped():
    case = Case(
        params=np.array([1.0, 2.0, 3.0]),
        constraints=[
            om.FixedConstraint(selector=lambda x: x[np.array([], dtype=int)]),
            om.IncreasingConstraint(selector=lambda x: x[np.array([], dtype=int)]),
        ],
        bounds=None,
    )
    converter, internal = _get_converter_and_internal(case)

    assert not converter.has_transforming_constraints
    assert internal.free_mask.tolist() == [True, True, True]
    aaae(internal.values, [1.0, 2.0, 3.0])


# ======================================================================================
# First-error behavior for violated start params and misspecified constraints.
# Pinned to exception types, not messages.
# ======================================================================================


def _raises(params, constraints, bounds=None):
    return get_converter(
        params=params,
        constraints=pre_process_constraints(constraints),
        bounds=bounds,
        func_eval=1.0,
        solver_type=AggregationLevel.SCALAR,
    )


def test_violated_probability_constraint_raises_invalid_params_error():
    with pytest.raises(InvalidParamsError):
        _raises(
            np.arange(3),
            [om.ProbabilityConstraint(selector=lambda x: x[[1, 2]])],
        )


def test_violated_increasing_constraint_raises_invalid_params_error():
    with pytest.raises(InvalidParamsError):
        _raises(
            np.array([3.0, 2.0, 1.0]),
            [om.IncreasingConstraint(selector=lambda x: x[[0, 1, 2]])],
        )


def test_violated_linear_constraint_raises_invalid_params_error():
    with pytest.raises(InvalidParamsError):
        _raises(
            np.array([1.0, 1.0]),
            [
                om.LinearConstraint(
                    selector=lambda x: x[[0, 1]], weights=[1, 1], value=5
                )
            ],
        )


def test_violated_decreasing_constraint_raises_invalid_params_error():
    with pytest.raises(InvalidParamsError):
        _raises(
            np.array([1.0, 2.0, 3.0]),
            [om.DecreasingConstraint(selector=lambda x: x[[0, 1, 2]])],
        )


def test_violated_equality_constraint_raises_invalid_params_error():
    with pytest.raises(InvalidParamsError):
        _raises(
            np.array([1.0, 2.0]),
            [om.EqualityConstraint(selector=lambda x: x[[0, 1]])],
        )


def test_violated_linear_bound_raises_invalid_params_error():
    """The other violated-linear test pins the equality (value) branch; this one pins
    the bound branch.
    """
    with pytest.raises(InvalidParamsError):
        _raises(
            np.array([1.0, 1.0]),
            [
                om.LinearConstraint(
                    selector=lambda x: x[[0, 1]], weights=[1, 1], lower_bound=10
                )
            ],
        )


def test_non_psd_covariance_at_start_raises_invalid_params_error():
    with pytest.raises(InvalidParamsError):
        _raises(
            np.array([1.0, 2.0, 1.0]),
            [om.FlatCovConstraint(selector=lambda x: x[:3])],
        )


def test_non_psd_sdcorr_at_start_raises_invalid_params_error():
    """A correlation larger than one makes the implied matrix indefinite."""
    with pytest.raises(InvalidParamsError):
        _raises(
            np.array([1.0, 1.0, 2.0]),
            [om.FlatSDCorrConstraint(selector=lambda x: x[:3])],
        )


def test_negative_probability_at_start_raises_invalid_params_error():
    """The parameters sum to one but one of them is negative; this pins a different
    branch than the (sum does not equal one) probability test above.
    """
    with pytest.raises(InvalidParamsError):
        _raises(
            np.array([-0.1, 0.6, 0.5]),
            [om.ProbabilityConstraint(selector=lambda x: x[[0, 1, 2]])],
        )


def test_fix_that_differs_from_start_value_raises_invalid_params_error():
    """Uses the old dict interface because the new constraint objects do not have a
    value attribute.
    """
    with pytest.raises(InvalidParamsError):
        _raises(
            np.arange(3),
            [{"selector": lambda x: x[1], "value": 10, "type": "fixed"}],
        )


def test_covariance_probability_overlap_raises_invalid_constraint_error():
    with pytest.raises(InvalidConstraintError):
        _raises(
            np.arange(10),
            [
                om.FlatCovConstraint(selector=lambda x: x[[1, 0, 2]]),
                om.ProbabilityConstraint(selector=lambda x: x[[0, 1]]),
            ],
        )


def test_covariance_increasing_overlap_raises_invalid_constraint_error():
    with pytest.raises(InvalidConstraintError):
        _raises(
            np.arange(10),
            [
                om.FlatCovConstraint(selector=lambda x: x[[6, 3, 5, 2, 1, 4]]),
                om.IncreasingConstraint(selector=lambda x: x[[0, 1, 2]]),
            ],
        )


def test_too_many_linear_constraints_raise_invalid_constraint_error():
    """All three constraints are satisfied at the start params, so the error can only
    come from the rank check during consolidation.
    """
    with pytest.raises(InvalidConstraintError):
        _raises(
            np.array([1.0, 2.0]),
            [
                om.LinearConstraint(
                    selector=lambda x: x[[0, 1]], weights=[1, 0.5], value=2
                ),
                om.LinearConstraint(
                    selector=lambda x: x[[0, 1]], weights=[1, -1], value=-1
                ),
                om.LinearConstraint(
                    selector=lambda x: x[[0, 1]], weights=[1, 1], value=3
                ),
            ],
        )


def test_bound_on_probability_constrained_param_raises_invalid_constraint_error():
    with pytest.raises(InvalidConstraintError):
        _raises(
            np.array([0.3, 0.7]),
            [om.ProbabilityConstraint(selector=lambda x: x[[0, 1]])],
            bounds=om.Bounds(lower=np.full(2, -inf), upper=np.array([0.5, inf])),
        )


def test_fix_of_covariance_constrained_param_raises_invalid_constraint_error():
    """Fixing any but the first parameter of a covariance constraint is invalid (as
    long as the constraint cannot be simplified to bounds).
    """
    with pytest.raises(InvalidConstraintError):
        _raises(
            np.array([1.0, 0.1, 2.0, 0.2, 0.3, 3.0]),
            [
                om.FlatCovConstraint(selector=lambda x: x[:6]),
                om.FixedConstraint(selector=lambda x: x[2]),
            ],
        )


def test_duplicate_selection_raises_invalid_constraint_error():
    with pytest.raises(InvalidConstraintError):
        _raises(
            np.array([1.0, 2.0, 3.0]),
            [om.EqualityConstraint(selector=lambda x: x[[0, 0, 1]])],
        )


def test_invalid_selector_field_raises_invalid_constraint_error():
    """Loc selectors are not allowed for general pytree params."""
    with pytest.raises(InvalidConstraintError):
        _raises(
            {"a": 1.0, "b": 2.0},
            [{"type": "fixed", "loc": "a"}],
        )


def test_bound_on_covariance_constrained_param_raises_invalid_constraint_error():
    """Bounds on any but the first parameter of a covariance constraint are invalid (as
    long as the constraint cannot be simplified to bounds). This mirrors the fix-based
    test above for the bounds path.
    """
    with pytest.raises(InvalidConstraintError):
        _raises(
            np.array([1.0, 0.1, 2.0, 0.2, 0.3, 3.0]),
            [om.FlatCovConstraint(selector=lambda x: x[:6])],
            bounds=om.Bounds(
                lower=np.full(6, -inf),
                upper=np.array([inf, inf, 5.0, inf, inf, inf]),
            ),
        )


def test_fix_of_probability_constrained_param_raises_invalid_constraint_error():
    """Mirror of the bound-on-probability test above for the fix path."""
    with pytest.raises(InvalidConstraintError):
        _raises(
            np.array([0.3, 0.7]),
            [
                om.ProbabilityConstraint(selector=lambda x: x[[0, 1]]),
                om.FixedConstraint(selector=lambda x: x[0]),
            ],
        )


def test_contradictory_bounds_across_equality_set_raises_invalid_constraint_error():
    """The two parameters are equality constrained, so consolidating their bounds yields
    a lower bound (4) that exceeds the upper bound (2).
    """
    with pytest.raises(InvalidConstraintError):
        _raises(
            np.array([3.0, 3.0]),
            [om.EqualityConstraint(selector=lambda x: x[[0, 1]])],
            bounds=om.Bounds(lower=np.array([4.0, -inf]), upper=np.array([inf, 2.0])),
        )
