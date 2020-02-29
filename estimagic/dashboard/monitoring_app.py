"""Show the development of one optimization's criterion and parameters over time."""
from functools import partial

from bokeh.layouts import Column
from bokeh.layouts import Row
from bokeh.models import ColumnDataSource
from bokeh.models import HoverTool
from bokeh.models import Panel
from bokeh.models import Tabs
from bokeh.models import Toggle

from estimagic.dashboard.utilities import create_wide_figure
from estimagic.dashboard.utilities import get_color_palette
from estimagic.logging.create_database import load_database
from estimagic.logging.read_database import read_new_iterations
from estimagic.logging.read_database import read_scalar_field
from estimagic.optimization.utilities import index_element_to_string


def monitoring_app(doc, database_name, session_data):
    """Create plots showing the development of the criterion and parameters until now.

    Options are loaded from the database. Supported options are:
        - rollover (int): How many iterations to keep before discarding.

    Args:
        doc (bokeh.Document): argument required by bokeh
        database_name (str): short and unique name of the database
        session_data (dict):
            infos to be passed between and within apps.
            Keys of this app's entry are:
            - last_retrieved (int): last iteration currently in the ColumnDataSource
            - database_path

    """
    database = load_database(session_data["database_path"])
    start_params = read_scalar_field(database, "start_params")
    dash_options = read_scalar_field(database, "dash_options")
    rollover = dash_options["rollover"]

    tables = ["criterion_history", "params_history"]
    criterion_history, params_history = _create_bokeh_data_sources(
        database=database, tables=tables
    )
    session_data["last_retrieved"] = 1

    # create initial bokeh elements without callbacks
    initial_convergence_plots = _create_initial_convergence_plots(
        criterion_history=criterion_history,
        params_history=params_history,
        start_params=start_params,
    )

    activation_button = Toggle(
        active=False,
        label="Start Updating from Database",
        button_type="danger",
        width=50,
        height=30,
        name="activation_button",
    )

    # add elements to bokeh Document
    bokeh_convergence_elements = [Row(activation_button)] + initial_convergence_plots
    convergence_tab = Panel(
        child=Column(*bokeh_convergence_elements), title="Convergence Tab"
    )
    tabs = Tabs(tabs=[convergence_tab])
    doc.add_root(tabs)

    # add callbacks
    activation_callback = _create_activation_callback(
        button=activation_button,
        doc=doc,
        database=database,
        session_data=session_data,
        rollover=rollover,
        tables=tables,
    )
    activation_button.on_change("active", activation_callback)


def _create_bokeh_data_sources(database, tables):
    """Load the first entry from the database to initialize the ColumnDataSources.

    Args:
        database (sqlalchemy.MetaData)
        tables (list): list of table names to load and convert to ColumnDataSources

    Returns:
        all_cds (list): list of ColumnDataSources

    """
    data_dict, _ = read_new_iterations(
        database=database,
        tables=tables,
        last_retrieved=0,
        limit=1,
        return_type="bokeh",
    )

    all_cds = []
    for tab, data in data_dict.items():
        cds = ColumnDataSource(data=data, name=f"{tab}_cds")
        all_cds.append(cds)
    return all_cds


def _create_initial_convergence_plots(criterion_history, params_history, start_params):
    """Create the initial convergence plots.

    Args:
        criterion_history (bokeh ColumnDataSource)
        params_history (bokeh ColumnDataSource)
        start_params (pd.DataFrame): params DataFrame that includes the "group" column.

    Returns:
        convergence_plots (list):
            List of bokeh Row elements, each containing one convergence plot.
    """
    criterion_plot = _plot_time_series(
        data=criterion_history,
        x_name="iteration",
        y_keys=["value"],
        y_names=["criterion"],
        title="Criterion",
    )
    convergence_plots = [criterion_plot]

    group_to_params = _map_groups_to_params(start_params)
    for g, group_params in group_to_params.items():
        param_group_plot = _plot_time_series(
            data=params_history, y_keys=group_params, x_name="iteration", title=g,
        )
        convergence_plots.append(Row(param_group_plot))
    return convergence_plots


def _plot_time_series(data, y_keys, x_name, title, y_names=None):
    """Plot time series linking the *y_keys* to a common *x_name* variable.

    Args:
        data (ColumnDataSource):
            data that contain the y_keys and x_name
        y_keys (list):
            list of the entries in the data that are to be plotted.
        x_name (str):
            name of the entry in the data that will be on the x axis.
        title (str):
            title of the plot.
        y_names (list):
            if given these replace the y keys for the names of the lines.

    Returns:
        plot (bokeh Figure)

    """
    if y_names is None:
        y_names = y_keys

    plot = create_wide_figure(title=title)

    colors = get_color_palette(nr_colors=len(y_keys))
    for color, y_key, y_name in zip(colors, y_keys, y_names):
        line_glyph = plot.line(
            source=data,
            x=x_name,
            y=y_key,
            line_width=2,
            legend_label=y_name,
            color=color,
            muted_color=color,
            muted_alpha=0.2,
        )
    tooltips = [(x_name, "@" + x_name)]
    tooltips += [("param_name", y_name), ("param_value", "@" + y_key)]
    hover = HoverTool(renderers=[line_glyph], tooltips=tooltips)
    plot.tools.append(hover)

    if len(y_key) == 1:
        plot.legend.visible = False
    else:
        plot.legend.click_policy = "mute"
        plot.legend.location = "top_left"

    return plot


def _map_groups_to_params(params):
    """Map the group name to the ColumnDataSource friendly parameter names.

    Args:
        params (pd.DataFrame):
            DataFrame with the parameter values and additional information such as the
            "group" column and Index.

    Returns:
        group_to_params (dict):
            Keys are the values of the "group" column. The values are lists with
            bokeh friendly strings of the index tuples identifying the parameters
            that belong to this group. Parameters where group is None are ignored.

    """
    group_to_params = {}
    for group in params["group"].unique():
        if group is not None:
            tup_params = params[params["group"] == group].index
            str_params = [index_element_to_string(tup) for tup in tup_params]
            group_to_params[group] = str_params
    return group_to_params


def _create_activation_callback(button, doc, database, session_data, rollover, tables):
    """Define the callback function that starts and resets the convergence plots.

    This effectively partials a lot of arguments as bokeh callbacks only support a
    very limited number of arguments (attr, old, new).

    Args:
        doc (bokeh.Document): argument required by bokeh
        database (sqlalchemy.MetaData)
        session_data (dict):
            this app's entry of infos to be passed between and within apps.
            The keys are:
            - last_retrieved (int): last iteration currently in the ColumnDataSource
            - database_path
        rollover (int): maximal number of points to show in the plot
        tables (list): list of table names to load and convert to ColumnDataSources


    Returns:
        activation_callback (func):
            function that starts the data updating callback when the button state is
            set to True and resets the convergence plots and stops their updating when
            the button state is set to False.

    """

    def activation_callback(
        attr,
        old,
        new,
        session_data=session_data,
        rollover=rollover,
        doc=doc,
        database=database,
        button=button,
        tables=tables,
    ):
        """Start and reset the convergence plots and their updating."""
        callback_dict = session_data["callbacks"]
        if new is True:
            plot_new_data = partial(
                _update_monitoring_tab,
                doc=doc,
                database=database,
                session_data=session_data,
                rollover=rollover,
                tables=tables,
            )
            callback_dict["plot_periodic_data"] = doc.add_periodic_callback(
                plot_new_data, period_milliseconds=200
            )
            # change the button color
            button.button_type = "success"
            button.label = "Reset Plot"
        else:
            doc.remove_periodic_callback(callback_dict["plot_periodic_data"])
            for table_name in ["criterion_history", "params_history"]:
                cds = doc.get_model_by_name(f"{table_name}_cds")
                column_names = cds.data.keys()
                cds.data = {name: [] for name in column_names}
            session_data["last_retrieved"] = 0
            # change the button color
            button.button_type = "danger"
            button.label = "Restart Plot"

    return activation_callback


def _update_monitoring_tab(doc, database, session_data, tables, rollover):
    """Callback to look up new entries in the database tables and plot them.

    Args:
        doc (bokeh.Document): argument required by bokeh
        database (sqlalchemy.MetaData)
        session_data (dict):
            infos to be passed between and within apps.
            Keys of this app's entry are:
            - last_retrieved (int): last iteration currently in the ColumnDataSource
            - database_path
        tables (list): list of table names to load and convert to ColumnDataSources
        rollover (int): maximal number of points to show in the plot

    """
    last_retrieved = session_data["last_retrieved"]
    new_data, new_last = read_new_iterations(
        database=database,
        tables=tables,
        last_retrieved=last_retrieved,
        return_type="bokeh",
        limit=20,
    )

    for table_name, to_add in new_data.items():
        cds = doc.get_model_by_name(f"{table_name}_cds")
        cds.stream(to_add, rollover=rollover)

    session_data["last_retrieved"] = new_last
