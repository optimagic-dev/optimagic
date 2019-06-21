"""Functions for creating and styling the convergence tab."""
from bokeh.layouts import column
from bokeh.models import ColumnDataSource
from bokeh.models import Panel
from tornado import gen

from estimagic.dashboard.plotting_functions import plot_with_lines
from estimagic.optimization.utilities import index_element_to_string

X_NAME = "XxXxITERATIONxXxX"


def setup_convergence_tab(params_df, initial_fitness):
    """
    Setup the convergence plot for later updating.

    This function is called in _setup_dashboard.

    Args:
        params_df (pandas DataFrame):
            DataFrame with the initial parameter values, constraints etc.

        initial_fitness (pd.Series):
            criterion function evaluated at the initial parameters

    """
    data_dict = _make_data_dict(
        iteration=0, fitness=initial_fitness, param_values=params_df["value"]
    )
    conv_data = ColumnDataSource(data_dict)

    fitness_plot = plot_with_lines(
        data=conv_data, y_keys=["fitness"], x_name=X_NAME, title="Fitness"
    )

    plots = [fitness_plot] + _param_plots(params_df=params_df, data=conv_data)

    tab = Panel(
        child=column(children=plots, sizing_mode="scale_width"),
        title="Convergence Plots",
    )

    return conv_data, tab


@gen.coroutine
def update_convergence_data(new_fitness, new_params, data, rollover):
    """
    Update the convergence data with new parameters and a new fitness value.

    Args:
        new_fitness (float):
            fitness value of the new iteration

        new_params (Series):
            new parameter values

        data (ColumnDataSource):
            ColumnDataSource to stream to

        rollover (int):
            maximum number of entries to keep before dropping earlier entries
            to add new entries

    """
    iteration = max(data.data[X_NAME]) + 1
    to_add = _make_data_dict(
        iteration=iteration, fitness=new_fitness, param_values=new_params
    )
    data.stream(to_add, rollover)


def _make_data_dict(iteration, fitness, param_values):
    """
    Create a dictionary of the right format for streaming to ColumnDataSource.

    Args:
        iteration (int):
            iteration number.

        fitness (float):
            current value of the criterion function.

        param_values (pd.Series):
            parameter values at which the criterion function was evaluated.


    """
    data_dict = {X_NAME: [iteration], "fitness": [fitness]}
    data_dict.update(
        {
            index_element_to_string(name): [param_values[name]]
            for name in param_values.index
        }
    )
    return data_dict


def _param_plots(params_df, data):
    """
    Create the plots that will show the convergence of groups of parameters.

    Args:
        params_df (pd.DataFrame):
            See :ref:`params`.

        data (ColumnDataSource):
            ColumnDataSource that will be updated with new iterations
            of the parameter values and fitness.

    """
    group_to_params = _map_groups_to_params(params_df)
    plots = []
    for g, params in group_to_params.items():
        group_plot = plot_with_lines(data=data, y_keys=params, x_name=X_NAME, title=g)
        plots.append(group_plot)
    return plots


def _map_groups_to_params(params_df):
    """Map the group name to the ColumnDataSource friendly parameter names."""
    group_to_params = {}
    for group in params_df["group"].unique():
        if group is not None:
            tup_params = params_df[params_df["group"] == group].index
            str_params = [index_element_to_string(tup) for tup in tup_params]
            group_to_params[group] = str_params
    return group_to_params
