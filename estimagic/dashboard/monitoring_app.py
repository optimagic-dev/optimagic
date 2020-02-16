from bokeh.layouts import Column
from bokeh.layouts import Row
from bokeh.models import Panel
from bokeh.models import Tabs

from estimagic.dashboard_old.plotting_functions import plot_time_series
from estimagic.logging.create_database import load_database
from estimagic.logging.read_database import read_last_iterations


def monitoring_app(doc, database_path):
    database = load_database(database_path)
    # sachen extrahieren, die ich brauch
    #   lad db_options mit read_scalar_field
    #   lad params -> für "_internal_free" Parameter, Gruppe ... -> ANSCHAUEN

    # alle Tabellen auf einmal laden -> ColumnDataSources updaten!

    tab1 = _setup_convergence_tab(database=database)
    tabs = Tabs(tabs=[tab1])
    doc.add_root(tabs)


def _setup_convergence_tab(database):
    criterion_values = read_last_iterations(
        database=database, tables="criterion_history", n=-1, return_type="bokeh"
    )

    fitness_plot = plot_time_series(
        data=criterion_values, x_name="iteration", y_keys=["value"], title="Criterion",
    )
    plots = [Row(fitness_plot)]
    tab = Panel(child=Column(*plots), title="Convergence Tab")
    return tab
