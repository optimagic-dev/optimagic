"""Test the functions to run the dashboard."""
from pathlib import Path
from time import sleep

import pytest

from estimagic.dashboard import run_dashboard


@pytest.fixture()
def database_paths():
    current_dir_path = Path(__file__).resolve().parent
    database_paths = [current_dir_path / "db1.db", current_dir_path / "db2.db"]
    return database_paths


def test_run_dashboard_in_separate_process(database_paths):
    p = run_dashboard.run_dashboard_in_separate_process(database_paths)
    sleep(5)
    p.terminate()


def test_process_dashboard_args_single_path():
    current_dir_path = Path(__file__).resolve().parent
    single_path = current_dir_path / "db1.db"
    database_name_to_path, no_browser, port = run_dashboard._process_dashboard_args(
        database_paths=single_path, no_browser=True, port=1000
    )
    assert database_name_to_path == {"db1": single_path}
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
