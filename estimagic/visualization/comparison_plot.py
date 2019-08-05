"""
Draw an interactive comparison plot of named result dictionaries.

The plot can plot many results for large numbers of parameters
against each other.

The plot can answer the following questions:

1. How are the parameters distributed?

2. How large are the differences in parameter estimates between results
    compared to the uncertainty around the parameter estimates?

3. Are parameters of groups of results clustered?

Example Usage: see tutorials/example_comparison_plot.ipynb

"""
import warnings

import numpy as np
import pandas as pd
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource
from bokeh.models import HoverTool
from bokeh.models import Range1d
from bokeh.models import TapTool
from bokeh.models.callbacks import CustomJS
from bokeh.models.widgets import CheckboxGroup
from bokeh.plotting import figure
from bokeh.plotting import show
from numba import jit

from estimagic.optimization.utilities import index_element_to_string


def comparison_plot(
    data_dict, color_dict=None, marker_dict=None, height=None, width=None
):
    """Make a comparison plot either from a data_dict.

    Args:
        data_dict (dict): The keys are the names of different models.
            The values is a dictinoary with the following keys and values:
                - result_df (pd.DataFrame):
                    params_df returned by estimagic.optimization.maximize or
                    estimagic.optimization.minimize.
                - model_class (str, optional):
                    name of the model class to which the model belongs.
                    This determines the color and checkbox entries with
                    which model classes can be selected and unselected

        color_dict (dict):
            maps model_class to color string that is understood by bokeh.

        marker_dict (dict):
            maps model_class to a marker string that is understood by bokeh scatter.

        height (int):
            height of the (entire) plot.

        width (int):
            width of the (entire) plot.

    """
    df, param_groups_and_heights, width, scatter_size = _process_inputs(
        data_dict=data_dict,
        color_dict=color_dict,
        marker_dict=marker_dict,
        height=height,
        width=width,
    )

    source = ColumnDataSource(df)

    plots = []
    for param_group_name, group_height in param_groups_and_heights:
        df_slice = df[df["group"] == param_group_name]
        to_plot = sorted(df_slice["full_name"].unique(), reverse=True)

        # create the "canvas"
        param_group_plot = figure(
            title="Comparison Plot of {} Parameters".format(param_group_name.title()),
            y_range=to_plot,
            plot_height=group_height,
            plot_width=width,
        )

        # add scatterplot representing the parameter value
        point_estimate_glyph = param_group_plot.scatter(
            source=source,
            x="final_value",
            y="name_with_dodge",
            size=scatter_size,
            color="color",
            selection_color="color",
            nonselection_color="color",
            alpha=0.5,
            selection_alpha=0.7,
            nonselection_alpha=0.3,
            marker="marker",
        )

        # add the confidence_intervals as hbars
        # horizontal whiskers not supported in bokeh 1.0.4
        if "conf_int_lower" in df.columns and "conf_int_upper" in df.columns:
            param_group_plot.hbar(
                source=source,
                y="name_with_dodge",
                left="conf_int_lower",
                right="conf_int_upper",
                height=0.01,
                alpha=0.0,
                selection_alpha=0.7,
                nonselection_fill_alpha=0.0,
                line_alpha=0.0,
                selection_line_alpha=0.7,
                nonselection_line_alpha=0.0,
                color="color",
                selection_color="color",
                nonselection_color="color",
            )

        _add_hover_tool(param_group_plot, point_estimate_glyph, df)

        _add_tap_tool(source, param_group_plot, point_estimate_glyph)

        _style_plot(param_group_plot, df_slice)

        plots.append(param_group_plot)

    if "model_class" in df.columns:
        cb_group = _create_checkbox(
            widget_labels=sorted(df["model_class"].unique()), source=source
        )
        plots = [cb_group] + plots

    grid = gridplot(plots, toolbar_location="right", ncols=1)
    show(grid)
    return grid, plots


def _process_inputs(data_dict, color_dict, marker_dict, height, width):
    """
    Convert a dictionary mapping model names to the optimization results to a DataFrame.

    Args:
        data_dict (dict): The keys are the names of different models.
            The values is a dictinoary with the following keys and values:
                - result_df (pd.DataFrame):
                    params_df returned by estimagic.optimization.maximize or
                    estimagic.optimization.minimize.
                - model_class (str, optional):
                    name of the model class to which the model belongs.
                    This determines the color and checkbox entries with
                    which model classes can be selected and unselected

        color_dict (dict):
            maps model_class to color string that is understood by bokeh.
            This is generated from the Category20 palette if not given.

        marker_dict (dict):
            maps model_class to a marker string that is understood by bokeh scatter.

        height (int):
            height of the (entire) plot.

        width (int):
            width of the (entire) plot.

    Returns:
        df (pd.DataFrame): DataFrame with the following columns:
            - *model* (str): model name
            - *value* (float): point estimate of the parameter value
            - *name* (str): name of the parameter (excluding its group)
            - *group* (str): name of the parameter group
                (used for grouping parameters in plots).
            - *color* (str): color

            if they were supplied by at least some of the result_dfs:
                - *conf_int_lower* (float): lower end of the confidence interval
                - *conf_int_upper* (float): upper end of the confidence interval
            if they were supplied by at least some data_dicts:
                - *model_class* (str): groups that can be filtered through the widget
        param_groups (list): list of paramater_groups
        group_and_heights (list):
            list of of tuples of the name of the group and height for each group plot
        width (int): width of the plot

    """
    color_dict, marker_dict = _build_or_check_option_dicts(
        color_dict, marker_dict, data_dict
    )
    df = _build_df_from_data_dict(data_dict, color_dict, marker_dict)
    _check_groups_and_names_compatible(df)

    if width is None:
        width = 600
    group_and_heights = _create_group_and_heights(df, height)

    df, scatter_size = _determine_dodge_and_scatter_size(
        df=df, param_groups_and_heights=group_and_heights, width=width
    )

    return df, group_and_heights, width, scatter_size


def _build_or_check_option_dicts(color_dict, marker_dict, data_dict):
    model_classes = {
        d["model_class"] for d in data_dict.values() if "model_class" in d.keys()
    }

    if color_dict is None:
        color_dict = {m: "#035096" for m in model_classes}
    else:
        assert set(model_classes).issubset(color_dict.keys()), (
            "Your color_dict does not map every model class "
            + "in your data_dict to a color."
        )

    if marker_dict is None:
        marker_dict = {m: "circle" for m in list(model_classes)}
    else:
        assert set(model_classes).issubset(marker_dict.keys()), (
            "Your marker_dict does not map every model class "
            + "in your data_dict to a marker."
        )

    return color_dict, marker_dict


def _build_df_from_data_dict(data_dict, color_dict, marker_dict):
    df = pd.DataFrame(
        columns=["conf_int_lower", "conf_int_upper", "group", "model_class"]
    )

    for model, mod_dict in data_dict.items():
        ext_param_df = mod_dict["result_df"]
        name_cols = [x for x in ["group", "name"] if x in ext_param_df.columns]
        ext_param_df["full_name"] = ext_param_df[name_cols].apply(
            lambda x: index_element_to_string(tuple(x)), axis=1
        )
        ext_param_df["model"] = model
        if "model_class" in mod_dict.keys():
            model_class = mod_dict["model_class"]
            ext_param_df["model_class"] = model_class
            ext_param_df["color"] = color_dict[model_class]
            ext_param_df["marker"] = marker_dict[model_class]

        else:
            # the standard color is mediumelectricblue
            ext_param_df["model_class"] = "no class"
            ext_param_df["color"] = "#035096"
            ext_param_df["marker"] = "circle"

        df = df.append(ext_param_df, sort=False)

    # reset index as safety measure to make sure the index
    # gives the position in the arrays
    # that the source data dictionary points to

    # keep as much information from the index as possible
    if type(df.index) is pd.MultiIndex:
        for name in df.index.names:
            drop = name in df.columns
            df.reset_index(name, drop=drop, inplace=True)
    else:
        drop = df.index.name in df.columns
        df.reset_index(drop=drop, inplace=True)

    if "group" not in df.columns:
        df["group"] = "all"

    return df


def _check_groups_and_names_compatible(df):
    name_to_group = {}
    for model in df["model"].unique():
        small_df = df[df["model"] == model]
        for name in small_df["name"].unique():
            current_group = set(df[df["name"] == name]["group"])
            assert (
                len(current_group) == 1
            ), "{} is assigned to several groups: {}".format(name, current_group)
            if name not in name_to_group.keys() or name_to_group[name] is None:
                name_to_group[name] = current_group
            else:
                supposed = name_to_group[name]
                if current_group is None:
                    pass
                else:
                    assert (
                        supposed == current_group
                    ), "{} was assigned to group {} before but now {}".format(
                        name, supposed, current_group
                    )


def _create_group_and_heights(df, height):
    param_groups = [x for x in df["group"].unique() if x is not None]
    nr_params = len(df[df["group"].isin(param_groups)]["name"].unique())
    nr_models = len(df["model"].unique())
    if height is None:
        model_param = max(min(nr_models, 60), 10)
        height = 8 * model_param * nr_params
    group_and_heights = []
    for group_name in param_groups:
        nr_group_params = len(df[df["group"] == group_name]["name"].unique())
        plot_height = int(nr_group_params / nr_params * height)
        group_and_heights.append((group_name, plot_height))
    return group_and_heights


def _determine_dodge_and_scatter_size(df, param_groups_and_heights, width):
    for scatter_size in [12, 9, 6, 3]:
        df["name_with_dodge"] = np.nan
        df["dodge"] = np.nan
        for group, height in param_groups_and_heights:
            group_df = df[df["group"] == group]
            param_names = group_df["full_name"].unique()

            height_points_per_param = height / len(param_names)
            dodge_unit = 1.5 * scatter_size / height_points_per_param
            x_range = group_df["final_value"].max() - group_df["final_value"].min()
            critical_dist = 1.5 * scatter_size * x_range / width

            for p in param_names:
                param_slice = df[df["full_name"] == p]
                values = param_slice["final_value"].sort_values()
                ind = values.index
                dist_to_left_neighbor = values.diff()
                needs_dodge = dist_to_left_neighbor < critical_dist
                dodge = increment_with_reset(needs_dodge.to_numpy()) * dodge_unit
                df.loc[ind, "dodge"] = dodge
        if df["dodge"].max() < 0.9:
            df["name_with_dodge"] = df.apply(
                lambda x: (x["full_name"], x["dodge"]), axis=1
            )
            return df, scatter_size
    prob_param_names = df[df["dodge"] >= 0.9]["full_name"].tolist()
    warnings.warn(
        "Points of "
        + ", ".join(prob_param_names)
        + " are stacked so high "
        + "that it is hard to distinguish to which parameter a point belongs. "
        + "Switch to a histogram, KDE plot or increase the plot height to avoid this."
    )
    df["name_with_dodge"] = df.apply(lambda x: (x["full_name"], x["dodge"]), axis=1)
    return df, scatter_size


@jit
def increment_with_reset(bool_arr):
    res = []
    for x in bool_arr:
        if x is False:
            res.append(0)
        else:
            res.append(res[-1] + 1)
    return np.array(res)


def _add_tap_tool(source, param_group_plot, point_estimate_glyph):
    tap_js_kwargs = {"source": source}
    tap_js_code = """
    // adapted from https://stackoverflow.com/a/44996422

    var chosen = source.selected.indices;
    if (typeof(chosen) == "number"){
        var chosen = [chosen]
    };

    var chosen_models = [];

    for (var i = 0; i < chosen.length; ++ i){
        chosen_models.push(source.data['model'][chosen[i]])
    };

    var chosen_models_indices = [];

    for (var i = 0; i < source.data['index'].length; ++ i){
        if (chosen_models.includes(source.data['model'][i])){
            chosen_models_indices.push(i)
        };
    };

    source.selected.indices = chosen_models_indices;
    source.change.emit();"""
    tap_callback = CustomJS(args=tap_js_kwargs, code=tap_js_code)
    # point_estimate_glyph as only renderer assures that when a point is chosen
    # only that point's model is chosen
    # this makes it impossible to choose models based on clicking confidence bands
    tap = TapTool(renderers=[point_estimate_glyph], callback=tap_callback)
    param_group_plot.tools.append(tap)


def _add_hover_tool(param_group_plot, point_estimate_glyph, df):
    top_cols = ["model", "full_name", "start_value", "final_value"]
    dont_display = ["color", "marker", "group", "name"]
    cols_sorted_by_missing = (
        (df.isnull().mean() + (df == None).mean())
        .sort_values(ascending=True)
        .index  # noqa
    )
    to_add = top_cols + [
        x for x in cols_sorted_by_missing if x not in top_cols and x not in dont_display
    ]

    tooltips = [(col, "@" + col) for col in to_add]
    hover = HoverTool(renderers=[point_estimate_glyph], tooltips=tooltips)
    param_group_plot.tools.append(hover)


def _create_checkbox(widget_labels, source):
    widget_js_kwargs = {"source": source, "group_list": widget_labels}
    widget_js_code = """
    // https://stackoverflow.com/a/36145278

    var chosen_inds = cb_obj.active;

    var chosen_widget_groups = [];

    for (var i = 0; i < group_list.length; ++ i){
        if (chosen_inds.includes(i)){
            chosen_widget_groups.push(group_list[i])
        };
    };

    to_select_inds = []

    for (var i = 0; i < source.data['index'].length; ++ i){
        if (chosen_widget_groups.includes(source.data['model_class'][i])){
            to_select_inds.push(i)
        };
    };

    source.selected.indices = to_select_inds;
    source.change.emit();"""
    widget_callback = CustomJS(args=widget_js_kwargs, code=widget_js_code)
    cb_group = CheckboxGroup(
        labels=widget_labels,
        active=[0] * len(widget_labels),
        callback=widget_callback,
        inline=True,
    )
    return cb_group


def _style_plot(fig, df_slice):
    fig.xgrid.grid_line_color = None
    fig.yaxis.major_tick_line_color = None
    fig.yaxis.axis_line_color = "white"
    fig.outline_line_color = None

    top = df_slice[["conf_int_upper", "final_value"]].max(axis=1).max()
    bottom = df_slice[["conf_int_lower", "final_value"]].min(axis=1).min()
    border_buffer = 0.07 * (top - bottom)
    fig.x_range = Range1d(bottom - border_buffer, top + border_buffer)
