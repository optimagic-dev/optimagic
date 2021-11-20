"""This module comprises all CLI capabilities of estimagic."""
import click
from estimagic.dashboard.run_dashboard import run_dashboard

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option()
def cli():
    """Command-line interface for estimagic."""
    pass


@cli.command()
@click.argument("database", required=True, type=click.Path())
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
@click.option(
    "--stride",
    default=1,
    help=(
        "Plot every stride_th database row in the dashboard. Note that some database "
        "rows only contain gradient evaluations, thus for some values of stride the "
        "convergence plot of the criterion function can be empty."
    ),
    type=int,
    show_default=True,
)
def dashboard(
    database,
    port,
    no_browser,
    rollover,
    jump,
    update_frequency,
    update_chunk,
    stride,
):
    """Start the dashboard to visualize optimizations."""
    updating_options = {
        "rollover": int(rollover),
        "update_frequency": update_frequency,
        "update_chunk": update_chunk,
        "stride": stride,
        "jump": jump,
    }

    run_dashboard(
        database_path=database,
        no_browser=no_browser,
        port=port,
        updating_options=updating_options,
    )
