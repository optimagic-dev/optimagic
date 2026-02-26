import numpy as np
import pytest

from optimagic.exceptions import NotInstalledError
from optimagic.optimizers.tranquilo import Tranquilo, TranquiloLS


@pytest.fixture()
def mock_problem():
    """Create a minimal mock of InternalOptimizationProblem."""

    class MockBounds:
        lower = np.array([-1.0, -1.0])
        upper = np.array([1.0, 1.0])

    class MockProblem:
        bounds = MockBounds()

        def batch_fun(self, xs):
            return [np.sum(x**2) for x in xs]

    return MockProblem()


def test_tranquilo_raises_if_version_too_old(monkeypatch, mock_problem):
    import optimagic.optimizers.tranquilo as tranquilo_mod

    monkeypatch.setattr(
        tranquilo_mod, "IS_TRANQUILO_VERSION_NEWER_OR_EQUAL_TO_0_1_0", False
    )

    algo = Tranquilo()
    x0 = np.array([0.5, 0.5])

    with pytest.raises(NotInstalledError, match="tranquilo"):
        algo._solve_internal_problem(mock_problem, x0)


def test_tranquilo_ls_raises_if_version_too_old(monkeypatch, mock_problem):
    import optimagic.optimizers.tranquilo as tranquilo_mod

    monkeypatch.setattr(
        tranquilo_mod, "IS_TRANQUILO_VERSION_NEWER_OR_EQUAL_TO_0_1_0", False
    )

    algo = TranquiloLS()
    x0 = np.array([0.5, 0.5])

    with pytest.raises(NotInstalledError, match="tranquilo"):
        algo._solve_internal_problem(mock_problem, x0)
