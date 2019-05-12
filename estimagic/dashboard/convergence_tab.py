"""Functions for creating and styling the convergence tab."""
from bokeh.layouts import column
from bokeh.models import ColumnDataSource
from bokeh.models import Panel
from pandas import MultiIndex
from tornado import gen

from estimagic.dashboard.plotting_functions import plot_with_lines

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
    conv_data = _convergence_data(params_df, initial_fitness)

    fitness_plot = plot_with_lines(
        data=conv_data, y_keys=["fitness"], x_name=X_NAME, title="Fitness"
    )

    plots = [fitness_plot]

    group_to_params = _map_groups_to_params(params_df)
    for g, params in group_to_params.items():
        group_plot = plot_with_lines(
            data=conv_data, y_keys=params, x_name=X_NAME, title=g
        )
        plots.append(group_plot)

    tab = Panel(child=column(plots), title="Convergence Plots")

    return conv_data, tab


@gen.coroutine
def update_convergence_data(new_fitness, new_params, data, rollover):
    iteration = max(data.data[X_NAME]) + 1
    to_add = {X_NAME: [iteration], "fitness": [new_fitness]}
    to_add.update({k: [new_params[k]] for k in new_params.index})
    data.stream(to_add, rollover)


def _convergence_data(params_df, initial_fitness):
    data_dict = {X_NAME: [0], "fitness": [initial_fitness]}
    params_dict = {p: [params_df["value"].loc[p]] for p in params_df.index}
    data_dict.update(params_dict)
    return ColumnDataSource(data=data_dict)


def _map_groups_to_params(params_df):
    if "param_group" not in params_df.columns:
        params_df = _add_parameter_groups(params_df)
    group_to_params = {}
    for group in params_df["param_group"].unique():
        if group is None:
            continue
        else:
            group_to_params[group] = params_df[params_df["param_group"] == group].index
    return group_to_params


def _add_parameter_groups(params_df):
    ind = params_df.index
    if type(ind) == MultiIndex:
        params_df["param_group"] = ind.get_level_values(ind.names[0])
    else:
        params_df["param_group"] = "Parameter Values"
    params_df["param_group"] = params_df["param_group"].where(
        ~params_df["fixed"], other=None
    )
    return params_df
