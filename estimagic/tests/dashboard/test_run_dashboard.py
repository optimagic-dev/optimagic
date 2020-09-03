"""Test the functions to run the dashboard."""
from pathlib import Path

import pytest
from click.testing import CliRunner

from estimagic.cli import cli
from estimagic.dashboard.run_dashboard import _create_session_data
from estimagic.dashboard.run_dashboard import _process_database_paths


@pytest.fixture()
def database_paths():
    current_dir_path = Path(__file__).resolve().parent
    database_paths = [current_dir_path / "db1.db", current_dir_path / "db2.db"]
    return database_paths


@pytest.fixture()
def database_name_to_path(database_paths):
    name_to_path = {"db1": database_paths[0], "db2": database_paths[1]}
    return name_to_path


def test_process_dashboard_args_single_path():
    single_path = Path(__file__).resolve().parent / "db1.db"
    database_name_to_path = _process_database_paths(database_paths=single_path)
    assert database_name_to_path == {"db1": single_path}


def test_process_dashboard_args_two_paths(database_paths):
    database_name_to_path = _process_database_paths(database_paths=database_paths)
    assert database_name_to_path == {"db1": database_paths[0], "db2": database_paths[1]}


def test_process_dashboard_args_bad_path():
    with pytest.raises(TypeError):
        _process_database_paths(database_paths=394)


def test_create_session_data(database_paths, database_name_to_path):
    res = _create_session_data(database_name_to_path)
    expected = {
        "master_app": {},
        "db1": {
            "last_retrieved": 0,
            "database_path": database_paths[0],
            "callbacks": {},
        },
        "db2": {
            "last_retrieved": 0,
            "database_path": database_paths[1],
            "callbacks": {},
        },
    }
    assert res == expected


def test_dashboard_cli(monkeypatch):
    def fake_run_dashboard(
        database_paths, no_browser, port, rollover, jump, frequency, update_chunk
    ):
        assert len(database_paths) == 2
        assert no_browser is True
        assert port == 9999
        assert jump is True

    monkeypatch.setattr("estimagic.cli.run_dashboard", fake_run_dashboard)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "dashboard",
            str(Path(__file__).parent / "*.db"),
            "--no-browser",
            "--port",
            "9999",
            "--jump",
        ],
    )

    assert result.exit_code == 0


def test_dashboard_cli_duplicate_paths(monkeypatch):
    def fake_run_dashboard(
        database_paths, no_browser, port, rollover, jump, frequency, update_chunk
    ):
        assert len(database_paths) == 2
        assert no_browser is False
        assert port is None

    monkeypatch.setattr("estimagic.cli.run_dashboard", fake_run_dashboard)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "dashboard",
            str(Path(__file__).parent / "*.db"),
            str(Path(__file__).parent / "db1.db"),
            str(Path(__file__).parent / "db2.db"),
        ],
    )

    assert result.exit_code == 0


def test_dashboard_cli_recursively_search_directories(monkeypatch):
    def fake_run_dashboard(
        database_paths, no_browser, port, rollover, jump, frequency, update_chunk
    ):
        assert len(database_paths) == 2

    monkeypatch.setattr("estimagic.cli.run_dashboard", fake_run_dashboard)

    runner = CliRunner()
    result = runner.invoke(cli, ["dashboard", str(Path(__file__).parent)])

    assert result.exit_code == 0
