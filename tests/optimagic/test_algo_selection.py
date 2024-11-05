from optimagic import algos


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
