import numpy as np
import pandas as pd
import plotly.express as px
from estimagic.config import PLOTLY_PALETTE
from estimagic.config import PLOTLY_TEMPLATE
from estimagic.visualization.plotting_utilities import get_layout_kwargs


def slice_plots(
    criterion,
    params,
    param_name_mapping=None,
    n_gridpoints=21,
    n_random_values=2,
    seed=5471,
    share_yrange=True,
    y_expand=0.02,
    colorscale=PLOTLY_PALETTE,
    template=PLOTLY_TEMPLATE,
    showlegend=True,
    layout_kwargs=None,
    legend_kwargs=None,
    title_kwargs=None,
):
    """Plot criterion along coordinates at given and random values.

    Args:
        criterion (callable): criterion function. Takes a DataFrame and returns a
            scalar value or dictionary with the entry "value".
        params (pandas.DataFrame): See :ref:`params`. Must contain finite lower and
            upper bounds for all parameters.
        param_name_mapping (dict or NoneType): Dictionary mapping old parameter names
            to new ones.
        n_gridpoins (int): Number of gridpoints on which the criterion function is
            evaluated. This is the number per plotted line.
        n_random_values (int): Number of random parameter vectors that are used as
            center of the plots.
        seed (int): Numpy randoms seed used when generating the random values.
        share_yrange (bool): If True, the individual plots share the scale on the yaxis.
        y_expand (float): The ration by which to expand the range of the (shared) y
            axis, such that the axis is not cropped at exactly max of Criterion Value.
        colorscale: The coloring palette for traces. Default is "qualitative.Set2".
        template (str): The template for the figure. Default is "plotly_white".
        showlegend (bool): If True, show legend.
        layout_kwargs (dict or NoneType): Dictionary of key word arguments used to
            update layout of plotly Figure object. If None, the default kwargs defined
            in the function will be used.
        legend_kwargs (dict or NoneType): Dictionary of key word arguments used to
            update position, orientation and title of figure legend. If None, default
            position and orientation will be used with no title.
        title_kwargs (dict or NoneType): Dictionary of key word arguments used to
            update properties of the figure title. Use {'text': '<desired title>'}
            to set figure title.

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
    param_names = plot_data["name"].unique()
    param_name_mapping = _process_names_mapping(param_name_mapping, param_names)
    arguments = []
    for _, row in plot_data.iterrows():
        p = params.copy(deep=True)
        p["value"] = row[params.index].astype(float)
        arguments.append(p)

    function_values = [criterion(arg) for arg in arguments]
    if isinstance(function_values[0], dict):
        function_values = [val["value"] for val in function_values]

    plot_data["Criterion Value"] = function_values
    lb = plot_data["Criterion Value"].min()
    ub = plot_data["Criterion Value"].max()
    y_range = ub - lb
    yaxis_ub = ub + y_range * y_expand
    yaxis_lb = lb - y_range * y_expand
    layout_kwargs = get_layout_kwargs(
        layout_kwargs, legend_kwargs, title_kwargs, template, showlegend
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
        subfig.update_xaxes(title={"text": param_name_mapping[par_name]})
        subfig.update_yaxes(title={"text": "Criterion Value"})
        if share_yrange is True:
            subfig.update_yaxes(range=[yaxis_lb, yaxis_ub])
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


def _process_names_mapping(params_mapping, old_names):
    """Get dictionary mappping old parameter names to new ones."""
    if params_mapping is None:
        params_mapping = {par: par for par in old_names}
    else:
        for par in old_names:
            if par not in params_mapping:
                params_mapping[par] = par
    return params_mapping
