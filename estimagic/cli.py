"""This module comprises all CLI capabilities of estimagic."""
import glob
from pathlib import Path

import click

from estimagic.dashboard.run_dashboard import run_dashboard

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option()
def cli():
    """Command-line interface for estimagic."""
    pass


@cli.command()
@click.argument("database", nargs=-1, required=True, type=click.Path())
@click.option(
    "--port",
    "-p",
    default=None,
    help="The port the dashboard server will listen on.",
    type=int,
    show_default=True,
)
@click.option(
    "--no-browser",
    is_flag=True,
    help="Don't open the dashboard in a browser after startup.",
)
@click.option(
    "--jump",
    is_flag=True,
    help="Jump to start the dashboard at the last rollover iterations.",
)
@click.option(
    "--rollover",
    default=10_000,
    help="After how many iterations convergence plots get truncated from the left.",
    type=int,
    show_default=True,
)
@click.option(
    "--update-frequency",
    default=1,
    help="Number of seconds to wait between checking for new entries in the database.",
    type=float,
    show_default=True,
)
@click.option(
    "--update-chunk",
    default=20,
    help="Upper limit how many new values are updated from the database at one update.",
    type=int,
    show_default=True,
)
def dashboard(
    database, port, no_browser, rollover, jump, update_frequency, update_chunk
):
    """Start the dashboard to visualize optimizations."""
    database_paths = []
    for path in database:
        # Search directories recursively for databases. "*" in is_dir() raises error.
        if "*" not in path and Path(path).is_dir():
            path = str(Path(path) / "**" / "*.db")

        database_paths.extend([Path(path) for path in glob.glob(path, recursive=True)])
    database_paths = list(set(database_paths))

    run_dashboard(
        database_paths=database_paths,
        no_browser=no_browser,
        port=port,
        rollover=rollover,
        jump=jump,
        update_frequency=update_frequency,
        update_chunk=update_chunk,
    )
