import pytest


@pytest.fixture(autouse=True)
def no_database(monkeypatch):
    """Never create a database.

    The trick is to mock the function not where it is defined but where it is used.

    """

    def return_false(*args, **kwargs):
        return False

    monkeypatch.setattr(
        "estimagic.optimization.optimize.prepare_database", return_false
    )
