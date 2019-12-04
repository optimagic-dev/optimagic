import pytest


@pytest.fixture(autouse=True)
def no_database(monkeypatch):
    """Never create a database.

    The trick is to mock the function not where it is defined but where it is used.

    The mock is not available if multiple optimizations are spawned with
    multiprocessing.

    """

    def return_false(*args, **kwargs):
        return False

    monkeypatch.setattr(
        "estimagic.optimization.optimize.prepare_database", return_false
    )
