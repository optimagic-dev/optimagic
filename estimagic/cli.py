import click

from estimagic.dashboard.run_dashboard import run_dashboard

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option()
def cli():
    """Build, convert and upload a conda package."""
    pass


@cli.command()
@click.argument("database", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--port", "-p", default=1234, type=int, show_default=True)
@click.option("--no-browser", is_flag=True)
def dashboard(database, port, no_browser):
    run_dashboard(database, no_browser, port)
