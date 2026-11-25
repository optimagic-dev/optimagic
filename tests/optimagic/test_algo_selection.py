from dataclasses import FrozenInstanceError

import pytest

from optimagic import algos
from optimagic.optimizers.scipy_optimizers import ScipyDifferentialEvolution


def test_dfols_is_present():
    assert hasattr(algos, "nag_dfols")
    assert hasattr(algos.Bounded, "nag_dfols")
    assert hasattr(algos.LeastSquares, "nag_dfols")
    assert hasattr(algos.Local, "nag_dfols")
    assert hasattr(algos.Bounded.Local.LeastSquares, "nag_dfols")
    assert hasattr(algos.Local.Bounded.LeastSquares, "nag_dfols")
    assert hasattr(algos.LeastSquares.Bounded.Local, "nag_dfols")


def test_scipy_cobyla_is_present():
    assert hasattr(algos, "scipy_cobyla")
    assert hasattr(algos.Local, "scipy_cobyla")
    assert hasattr(algos.NonlinearConstrained, "scipy_cobyla")
    assert hasattr(algos.GradientFree, "scipy_cobyla")
    assert hasattr(algos.Local.NonlinearConstrained, "scipy_cobyla")
    assert hasattr(algos.NonlinearConstrained.Local, "scipy_cobyla")
    assert hasattr(algos.GradientFree.NonlinearConstrained, "scipy_cobyla")
    assert hasattr(algos.GradientFree.NonlinearConstrained.Local, "scipy_cobyla")
    assert hasattr(algos.Local.GradientFree.NonlinearConstrained, "scipy_cobyla")
    assert hasattr(algos.NonlinearConstrained.GradientFree.Local, "scipy_cobyla")
    assert hasattr(algos.NonlinearConstrained.Local.GradientFree, "scipy_cobyla")
    assert hasattr(algos.Local.NonlinearConstrained.GradientFree, "scipy_cobyla")


def test_algorithm_lists():
    assert len(algos.All) >= len(algos.Available)
    assert len(algos.AllNames) == len(algos.All)
    assert len(algos.AvailableNames) == len(algos.Available)


def test_selection_cannot_be_modified():
    with pytest.raises(FrozenInstanceError):
        algos.scipy_lbfgsb = None  # ty:ignore[invalid-assignment]
    with pytest.raises(FrozenInstanceError):
        del algos.scipy_lbfgsb


def test_selections_of_the_same_categories_are_equal():
    assert algos.Bounded.Local == algos.Local.Bounded
    assert algos.Bounded != algos.Local
    assert len({algos.Bounded.Local, algos.Local.Bounded, algos.Local}) == 2


def test_selection_repr_lists_the_algorithms():
    selection = algos.Bounded.Global.NonlinearConstrained.Parallel.Scalar
    assert selection.All == [ScipyDifferentialEvolution]
    expected = (
        "BoundedGlobalNonlinearConstrainedParallelScalarAlgorithms("
        "scipy_differential_evolution)"
    )
    assert repr(selection) == expected
