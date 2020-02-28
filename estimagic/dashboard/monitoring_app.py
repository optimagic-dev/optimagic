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

    Args:
        doc (bokeh.Document): argument required by bokeh
        database_name (str): short and unique name of the database
        session_data (dict):
            infos to be passed between and within apps.
            Keys of this app's entry are:
            - last_retrieved (int): last iteration currently in the ColumnDataSource
            - database_path

    """
    database = load_database(session_data[database_name]["database_path"])
    start_params = read_scalar_field(database, "start_params")
    dash_options = read_scalar_field(database, "dash_options")

    data_dict, new_last = read_new_iterations(
        database=database,
        tables=["criterion_history", "params_history"],
        last_retrieved=0,
        limit=1,
        return_type="bokeh",
    )
    session_data[database_name]["last_retrieved"] = new_last

    criterion_history = ColumnDataSource(
        data=data_dict["criterion_history"],
        name=f"{database_name}_criterion_history_cds",
    )
    params_history = ColumnDataSource(
        data=data_dict["params_history"], name=f"{database_name}_params_history_cds"
    )

    conv_tab = _setup_convergence_tab(
        doc=doc,
        database_name=database_name,
        database=database,
        criterion_history=criterion_history,
        params_history=params_history,
        start_params=start_params,
        session_data=session_data,
        **dash_options,
    )

    tabs = Tabs(tabs=[conv_tab])
    doc.add_root(tabs)


def _setup_convergence_tab(
    doc,
    database_name,
    database,
    criterion_history,
    params_history,
    start_params,
    session_data,
):
    """Create the figures and plot available time series of the criterion and parameters.

    Args:
        doc (bokeh.Document): argument required by bokeh
        database_name (str): short and unique name of the database
        database (sqlalchemy.MetaData)
        criterion_history (bokeh.ColumnDataSource):
            history of the criterion's values, loaded from the optimization's database.
        params_history (bokeh.ColumnDataSource):
            history of the parameters' values, loaded from the optimization's database.
        start_params (pd.DataFrame):
            DataFrame with the initial parameter values and additional columns,
            in particular the "group" column.
        session_data (dict):
            infos to be passed between and within apps.
            Keys of this app's entry are:
            - last_retrieved (int): last iteration currently in the ColumnDataSource
            - database_path

    Returns:
        tab (bokeh.Panel)
    """
    activation_button = _create_callback_button(
        doc=doc,
        database_name=database_name,
        database=database,
        session_data=session_data,
        reset_cds=True,
    )

    criterion_plot = _plot_time_series(
        data=criterion_history,
        x_name="iteration",
        y_keys=["value"],
        y_names=["criterion"],
        title="Criterion",
    )
    plots = [Row(activation_button), Row(criterion_plot)]
    group_to_params = _map_groups_to_params(start_params)
    for g, group_params in group_to_params.items():
        group_plot = _plot_time_series(
            data=params_history, y_keys=group_params, x_name="iteration", title=g,
        )
        plots.append(Row(group_plot))

    tab = Panel(child=Column(*plots), title="Convergence Tab")
    return tab


def _plot_time_series(data, y_keys, x_name, title, y_names=None):
    """Plot time series linking the *y_keys* to a common *x_name* variable.

    Args:
        data (ColumnDataSource):
            data that contain the y_keys and x_name
        y_keys (list):
            list of the entries in the data that are to be plotted
        x_name (str):
            name of the entry in the data that will be on the x axis
        title (str):
            title of the plot
        y_names (list):
            if given these replace the y keys for the names of the lines

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
    """
    group_to_params = {}
    for group in params["group"].unique():
        if group is not None:
            tup_params = params[params["group"] == group].index
            str_params = [index_element_to_string(tup) for tup in tup_params]
            group_to_params[group] = str_params
    return group_to_params


def _create_callback_button(doc, database_name, database, session_data, reset_cds):
    """Create a Button that changes color when clicked displaying its boolean state.

    Args:
        doc (bokeh Document): document to which add and remove the periodic callback
        database_name (str): name of the database
        database (sqlalchemy.MetaData)
        session_data (dict):
            infos to be passed between and within apps.
            Keys of this app's entry are:
            - last_retrieved (int): last iteration currently in the ColumnDataSource
            - database_path
        reset_cds (bool): whether to reset the ColumnDataSource when deactivating

    Returns:
        activation_button (bokeh Toggle)

    """
    callback_button = Toggle(
        active=False,
        label="Start Updating from Database" if reset_cds else "",
        button_type="danger",
        width=50,
        height=30,
        name="{}_button_{}".format(
            "activation" if reset_cds else "pause", database_name
        ),
    )

    def button_click_callback(attr, old, new, session_data=session_data):
        callback_dict = session_data[database_name]["callbacks"]
        if new is True:
            plot_new_data = partial(
                _update_monitoring_tab,
                doc=doc,
                database_name=database_name,
                database=database,
                session_data=session_data,
            )
            callback_dict["plot_periodic_data"] = doc.add_periodic_callback(
                plot_new_data, period_milliseconds=200
            )
            # change the color
            callback_button.button_type = "success"
            label = "Reset Plot" if reset_cds else "Stop Updating from Database"
            callback_button.label = label
        else:
            doc.remove_periodic_callback(callback_dict["plot_periodic_data"])
            # change the color
            callback_button.button_type = "danger"
            label = "Restart Plot" if reset_cds else "Resume Updating from Database"
            callback_button.label = label
            if reset_cds:
                for table_name in ["criterion_history", "params_history"]:
                    cds = doc.get_model_by_name(f"{database_name}_{table_name}_cds")
                    column_names = cds.data.keys()
                    cds.data = {name: [] for name in column_names}
                session_data[database_name]["last_retrieved"] = 0

    callback_button.on_change("active", button_click_callback)
    return callback_button


def _update_monitoring_tab(doc, database_name, database, session_data, rollover=500):
    """Callback to look up new entries in the database and plot them.

    Args:
        doc (bokeh.Document): argument required by bokeh
        database_name (str): short and unique name of the database
        database (sqlalchemy.MetaData)
        session_data (dict):
            infos to be passed between and within apps.
            Keys of this app's entry are:
            - last_retrieved (int): last iteration currently in the ColumnDataSource
            - database_path
        rollover (int): maximal number of points to show in the plot

    """
    last_retrieved = session_data[database_name]["last_retrieved"]
    new_data, new_last = read_new_iterations(
        database=database,
        tables=["criterion_history", "params_history"],
        last_retrieved=last_retrieved,
        return_type="bokeh",
        limit=20,
    )

    for table_name, to_add in new_data.items():
        cds = doc.get_model_by_name(f"{database_name}_{table_name}_cds")
        cds.stream(to_add, rollover=rollover)

    session_data[database_name]["last_retrieved"] = new_last
