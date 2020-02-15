import pandas as pd
from bokeh.layouts import Column
from bokeh.layouts import Row
from bokeh.models import Panel
from bokeh.models import Tabs

from estimagic.dashboard_old.plotting_functions import plot_time_series


def monitoring_app(doc, database):
    tab1 = _setup_convergence_tab(database=database)
    tabs = Tabs(tabs=[tab1])
    doc.add_root(tabs)


def _setup_convergence_tab(database):

    df = pd.DataFrame()
    df["x"] = [0, 1, 2, 3, 4, 5]
    df["y"] = [2, 4, 0, 1, 3, 2]

    fitness_plot = plot_time_series(
        data=df, x_name="x", y_keys=["y"], title="Trivial example"
    )
    plots = [Row(fitness_plot)]
    tab = Panel(child=Column(*plots), title="Convergence Tab")
    return tab
