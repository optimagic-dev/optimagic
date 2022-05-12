import itertools
from copy import deepcopy

import numpy as np
import pandas as pd
import plotly.express as px
from estimagic.config import PLOTLY_PALETTE
from estimagic.config import PLOTLY_TEMPLATE
from plotly import graph_objects as go
from plotly.subplots import make_subplots


def combine_plots(
    plots_dict,
    parameter_mapping=None,
    plots_per_row=2,
    sharex=False,
    sharey=True,
    make_subplot_kwargs=None,
    layout_kwargs=None,
    legend_kwargs=None,
    title_kwargs=None,
    template=PLOTLY_TEMPLATE,
):
    """Combine individual plots into figure with subplots.
    Uses dictionary with plotly images as values to build plotly Figure with subplots.

    Args:
        plots_dict (dict): Dictionary with plots of univariate effects for each
            parameter.
        parameter_mapping (dict or NoneType): A dictionary with custom parameter names
            to display as axes labels.
        plots_per_row (int): Number of plots per row.
        make_subplot_kwargs (dict or NoneType): Dictionary of keyword arguments used
            to instantiate plotly Figure with multiple subplots. Is used to define
            properties such as, for example, the spacing between subplots. If None,
            default arguments defined in the function are used.
        sharex (bool): Whether to share the properties of x-axis across subplots.
            Default False.
        sharey (bool): Whether to share the properties ofy-axis across subplots.
            Default True.
        layout_kwargs (dict or NoneType): Dictionary of key word arguments used to
            update layout of plotly Figure object. If None, the default kwargs defined
            in the function will be used.
        legend_kwargs (dict or NoneType): Dictionary of key word arguments used to
            update position, orientation and title of figure legend. If None, default
            position and orientation will be used with no title.
        title_kwargs (dict or NoneType): Dictionary of key word arguments used to
            update properties of the figure title. Use {'text': '<desired title>'}
            to set figure title.
        template (str): Plotly layout template. Must be one of plotly.io.templates.

    Returns:
        fig (plotly.Figure): Plotly figure with subplots that combines individual
            slice plots.

    """
    plots_dict = deepcopy(plots_dict)

    params = list(plots_dict.keys())
    parameter_mapping = _process_params_mapping(parameter_mapping, params)
    make_subplot_kwargs, nrows = _get_make_subplot_kwargs(
        sharex, sharey, make_subplot_kwargs, plots_per_row, params
    )
    fig = make_subplots(**make_subplot_kwargs)
    layout_kwargs = _get_layout_kwargs(
        layout_kwargs, legend_kwargs, title_kwargs, template
    )
    for i, (row, col) in enumerate(
        itertools.product(np.arange(nrows), np.arange(plots_per_row))
    ):
        try:
            subfig = plots_dict[params[i]]
        except IndexError:
            subfig = go.Figure()
        if not (row == 0 and col == 0):
            for d in subfig.data:
                d.update({"showlegend": False})
                fig.add_trace(d, col=col + 1, row=row + 1)
        else:
            for d in subfig.data:
                fig.add_trace(
                    d,
                    col=col + 1,
                    row=row + 1,
                )
        if subfig.data:
            fig.update_xaxes(
                title_text=f"{parameter_mapping[params[i]]}", row=row + 1, col=col + 1
            )
        if col == 0:
            fig.update_yaxes(
                title_text="Criterion Value",
                row=row + 1,
                col=col + 1,
            )
    fig.update_layout(**layout_kwargs)
    return fig


def get_slice_plots(
    criterion,
    params,
    n_gridpoints=21,
    n_random_values=2,
    seed=5471,
    colorscale=PLOTLY_PALETTE,
    layout_kwargs=None,
    legend_kwargs=None,
    title_kwargs=None,
    template=PLOTLY_TEMPLATE,
):
    """Plot criterion along coordinates at given and random values.

    Args:
        criterion (callable): criterion function. Takes a DataFrame and
            returns a scalar value or dictionary with the entry "value".
        params (pandas.DataFrame): See :ref:`params`. Must contain finite
            lower and upper bounds for all parameters.
        n_gridpoints (int): Number of gridpoints on which the criterion
            function is evaluated. This is the number per plotted line.
        n_random_values (int): Number of random parameter vectors that
            are used as center of the plots.
        figure containing subplots for each factor pair or a dictionary
        of individual plots. Default True.
        template (str): The template for the figure. Default is "plotly_white".
        layout_kwargs (dict or NoneType): Dictionary of key word arguments used to
            update layout of plotly Figure object. If None, the default kwargs defined
            in the function will be used.
        legend_kwargs (dict or NoneType): Dictionary of key word arguments used to
            update position, orientation and title of figure legend. If None, default
            position and orientation will be used with no title.
        title_kwargs (dict or NoneType): Dictionary of key word arguments used to
            update properties of the figure title. Use {'text': '<desired title>'}
            to set figure title.
        template (str): Plotly layout template. Must be one of plotly.io.templates.

    Returns:
        plotly.Figure: The grid plot or dict of individual plots


    """
    params = params.copy(deep=True)

    np.random.seed(seed)
    if (
        "lower_bound" not in params.columns
        or not np.isfinite(params["lower_bound"]).all()
    ):
        raise ValueError("All parameters need a finite lower bound.")
    if (
        "upper_bound" not in params.columns
        or not np.isfinite(params["upper_bound"]).all()
    ):
        raise ValueError("All parameters need a finite upper bound.")

    if "name" not in params.columns:
        names = [_index_element_to_string(tup) for tup in params.index]
        params["name"] = names

    plot_data = _get_plot_data(
        params=params,
        use_random_value=False,
        value_identifier="start values",
        n_gridpoints=n_gridpoints,
    )
    to_concat = [plot_data]

    for i in range(n_random_values):
        to_concat.append(
            _get_plot_data(
                params=params,
                use_random_value=True,
                value_identifier=f"random value {i}",
                n_gridpoints=n_gridpoints,
            )
        )

    plot_data = pd.concat(to_concat).reset_index()

    arguments = []
    for _, row in plot_data.iterrows():
        p = params.copy(deep=True)
        p["value"] = row[params.index].astype(float)
        arguments.append(p)

    function_values = [criterion(arg) for arg in arguments]
    if isinstance(function_values[0], dict):
        function_values = [val["value"] for val in function_values]

    plot_data["Criterion Value"] = function_values

    layout_kwargs = _get_layout_kwargs(
        layout_kwargs, legend_kwargs, title_kwargs, template
    )
    plots_dict = {}
    for par_name in plot_data["name"].unique():
        df = plot_data[plot_data["name"] == par_name]
        subfig = px.line(
            df,
            y="Criterion Value",
            x="Parameter Value",
            color="value_identifier",
            color_discrete_sequence=colorscale,
        )
        subfig.update_layout(**layout_kwargs)
        subfig.update_xaxes(title={"text": par_name})
        subfig.update_yaxes(title={"text": "Criterion"})
        plots_dict[par_name] = subfig

    return plots_dict


def _get_plot_data(params, use_random_value, value_identifier, n_gridpoints):
    params = params.copy(deep=True)
    if use_random_value:
        params = params.copy()
        params["value"] = np.random.uniform(
            params["lower_bound"], params["upper_bound"]
        )

    to_concat = []
    for loc in params.index:
        param_name = params.loc[loc, "name"]
        lb = params.loc[loc, "lower_bound"]
        ub = params.loc[loc, "upper_bound"]
        df = pd.DataFrame(
            data=[params["value"].to_numpy()] * n_gridpoints,
            columns=params.index,
        )
        grid = np.linspace(lb, ub, n_gridpoints)
        df[loc] = grid
        df["Parameter Value"] = grid
        df["name"] = param_name
        df["value_identifier"] = value_identifier
        to_concat.append(df)

    plot_data = pd.concat(to_concat).reset_index()
    return plot_data


def _index_element_to_string(element, separator="_"):
    if isinstance(element, (tuple, list)):
        as_strings = [str(entry).replace("-", "_") for entry in element]
        res_string = separator.join(as_strings)
    else:
        res_string = str(element)
    return res_string


def _get_make_subplot_kwargs(sharex, sharey, kwrgs, plots_per_row, params):
    """Define and update keywargs for instantiating figure with subplots."""
    nrows = int(np.ceil(len(params) / plots_per_row))
    default_kwargs = {
        "rows": nrows,
        "cols": plots_per_row,
        "start_cell": "top-left",
        "print_grid": False,
        "shared_yaxes": sharey,
        "shared_xaxes": sharex,
        "vertical_spacing": 0.2,
    }
    if kwrgs:
        default_kwargs.update(kwrgs)
    return default_kwargs, nrows


def _process_params_mapping(params_mapping, params):
    """Get dictionary mappping old parameter names to new ones."""
    if params_mapping is None:
        params_mapping = {par: par for par in params}
    else:
        for par in params:
            if par not in params_mapping:
                params_mapping[par] = par
    return params_mapping


def _get_layout_kwargs(layout_kwargs, legend_kwargs, title_kwargs, template):
    """Define and update default kwargs for update_layout.
    Defines some default keyword arguments to update figure layout, such as
    title and legend.

    """
    default_kwargs = {
        "template": template,
        "xaxis_showgrid": False,
        "yaxis_showgrid": False,
        "legend": {},
        "title": {},
    }
    if title_kwargs:
        default_kwargs["title"] = title_kwargs
    if legend_kwargs:
        default_kwargs["legend"].update(legend_kwargs)
    if layout_kwargs:
        default_kwargs.update(layout_kwargs)
    return default_kwargs
