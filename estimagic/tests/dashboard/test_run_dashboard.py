"""Test the functions to run the dashboard."""
from pathlib import Path
from time import sleep

import pytest
from click.testing import CliRunner

from estimagic.cli import cli
from estimagic.dashboard import run_dashboard


@pytest.fixture()
def database_paths():
    current_dir_path = Path(__file__).resolve().parent
    database_paths = [current_dir_path / "db1.db", current_dir_path / "db2.db"]
    return database_paths


@pytest.fixture()
def database_name_to_path(database_paths):
    name_to_path = {"db1": database_paths[0], "db2": database_paths[1]}
    return name_to_path


def test_run_dashboard_in_separate_process(database_paths):
    # integration test
    p = run_dashboard.run_dashboard_in_separate_process(database_paths)
    sleep(5)
    p.terminate()


def test_process_dashboard_args_single_path():
    current_dir_path = Path(__file__).resolve().parent
    single_path = current_dir_path / "db1.db"
    database_name_to_path, no_browser, port = run_dashboard._process_dashboard_args(
        database_paths=single_path, no_browser=False, port=1000
    )
    assert database_name_to_path == {"db1": single_path}
    assert no_browser is False
    assert port == 1000


def test_process_dashboard_args_two_paths(database_paths):
    database_name_to_path, no_browser, port = run_dashboard._process_dashboard_args(
        database_paths=database_paths, no_browser=None, port=1000
    )
    assert database_name_to_path == {"db1": database_paths[0], "db2": database_paths[1]}
    assert no_browser is True
    assert port == 1000


def test_process_dashboard_args_bad_path():
    with pytest.raises(TypeError):
        run_dashboard._process_dashboard_args(
            database_paths=394, no_browser=True, port=1000
        )


def test_process_dashboard_args_wrong_port(database_paths):
    with pytest.raises(TypeError):
        run_dashboard._process_dashboard_args(
            database_paths=database_paths, no_browser=True, port="False"
        )


def test_create_session_data(database_paths, database_name_to_path):
    res = run_dashboard._create_session_data(database_name_to_path)
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
    def fake_run_dashboard(database_paths, no_browser, port):
        assert len(database_paths) == 2
        assert no_browser is True
        assert port == 9999

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
        ],
    )

    assert result.exit_code == 0


def test_dashboard_cli_duplicate_paths(monkeypatch):
    def fake_run_dashboard(database_paths, no_browser, port):
        assert len(database_paths) == 2
        assert no_browser is False
        assert port == 1234

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
    def fake_run_dashboard(database_paths, no_browser, port):
        assert len(database_paths) == 2

    monkeypatch.setattr("estimagic.cli.run_dashboard", fake_run_dashboard)

    runner = CliRunner()
    result = runner.invoke(cli, ["dashboard", str(Path(__file__).parent)])

    assert result.exit_code == 0
