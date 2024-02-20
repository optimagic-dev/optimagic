"""Test the functions to run the dashboard."""

from click.testing import CliRunner
from estimagic.cli import cli
from estimagic.config import EXAMPLE_DIR


def test_dashboard_cli(monkeypatch):
    def fake_run_dashboard(
        database_path,
        no_browser,
        port,
        updating_options,
    ):
        assert database_path == str(EXAMPLE_DIR / "db1.db")
        assert no_browser
        assert port == 9999
        assert updating_options["jump"]
        assert updating_options["stride"] == 1

    monkeypatch.setattr("estimagic.cli.run_dashboard", fake_run_dashboard)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "dashboard",
            str(EXAMPLE_DIR / "db1.db"),
            "--no-browser",
            "--port",
            "9999",
            "--jump",
        ],
    )

    assert result.exit_code == 0
