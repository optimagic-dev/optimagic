"""
Draw an interactive comparison plot of named result dictionaries.

The plot can plot many results for large numbers of parameters
against each other.

The plot can answer the following questions:

1. How are the parameters distributed?

2. How large are the differences in parameter estimates between results
    compared to the uncertainty around the parameter estimates?

3. Are parameters of groups of results clustered?

"""
import numpy as np
import pandas as pd
from bokeh.layouts import gridplot
from bokeh.models import BoxSelectTool
from bokeh.models import ColumnDataSource
from bokeh.models import HoverTool
from bokeh.models import Range1d
from bokeh.models import TapTool
from bokeh.models.annotations import BoxAnnotation
from bokeh.models.callbacks import CustomJS
from bokeh.models.widgets import CheckboxGroup
from bokeh.plotting import figure
from bokeh.plotting import show

from estimagic.optimization.utilities import index_element_to_string


def comparison_plot(
    data_dict, color_dict=None, height=None, width=600, point_estimate_plot_kwargs=None
):
    """Make a comparison plot either from a data_dict.

    Args:
        data_dict (dict): The keys are the names of different models.
            Each value is a dictinoary with the following keys and values:
                - result_df (pd.DataFrame):
                    params returned by estimagic.optimization.maximize or
                    estimagic.optimization.minimize.
                - model_class (str, optional):
                    name of the model class to which the model belongs.
                    This determines the color and checkbox entries with
                    which model classes can be selected and unselected.

        color_dict (dict):
            maps model_class to color string that is understood by bokeh.

        height (int):
            height of the (entire) plot.

        width (int):
            width of the (entire) plot.

    """

    df = _build_df_from_data_dict(data_dict, color_dict)
    param_groups_and_heights = _create_group_and_heights(df, height)
    df = _add_rectangle_specs_to_df(df, param_groups_and_heights, width)
    source = ColumnDataSource(df)

    plots = []
    for param_group_name, group_height in param_groups_and_heights:
        df_slice = df[df["group"] == param_group_name]
        y_range = sorted(df_slice["full_name"].unique(), reverse=True)

        param_group_plot = figure(
            title="Comparison Plot of {} Parameters".format(param_group_name.title()),
            y_range=y_range,
            plot_height=group_height,
            plot_width=width,
            tools="reset,save",
        )

        point_estimate_glyph = param_group_plot.rect(
            source=source,
            x="binned_x",
            y="name_with_dodge",
            width="rect_width",
            height="rect_height",
            color="color",
            selection_color="color",
            nonselection_color="color",
            alpha=0.5,
            selection_alpha=0.7,
            nonselection_alpha=0.3,
        )

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

        _add_select_tools(source, param_group_plot, point_estimate_glyph)

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


def _build_df_from_data_dict(data_dict, color_dict):
    df = pd.DataFrame(
        columns=["conf_int_lower", "conf_int_upper", "group", "model_class"]
    )
    for model, model_dict in data_dict.items():
        result_df = model_dict["result_df"].reset_index()
        result_df["model"] = model
        name_cols = [x for x in ["group", "name"] if x in result_df.columns]
        result_df["full_name"] = result_df[name_cols].apply(
            lambda x: index_element_to_string(tuple(x)), axis=1
        )
        if "model_class" in model_dict.keys():
            model_class = model_dict["model_class"]
            result_df["model_class"] = model_class
            if model_class in color_dict.keys():
                result_df["color"] = color_dict[model_class]
            else:
                result_df["color"] = "#035096"
        else:
            result_df["model_class"] = "no class"
            result_df["color"] = "#035096"

        df = df.append(result_df, sort=False)

    if "group" not in df.columns:
        df["group"] = "all"

    return df.reset_index(drop=True)


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


def _add_rectangle_specs_to_df(df, width):
    df["rect_width"] = np.nan
    df["rect_height"] = np.nan
    df["rect_angle"] = 0.0
    df["needs_dodge"] = np.nan
    df["binned_x"] = np.nan
    df["dodge"] = np.nan
    for group in df["group"].unique():
        group_df = df[df["group"] == group]
        group_ind = group_df.index
        x_range = group_df["final_value"].max() - group_df["final_value"].min()
        min_rect_width = x_range / 80
        no_overlap_rect_width = _smallest_diff_btw_params(group_df, x_range)

        if no_overlap_rect_width >= min_rect_width:
            rect_width = min(x_range / 20, no_overlap_rect_width)
            df.loc[group_ind, "rect_width"] = rect_width
            df.loc[group_ind, "rect_height"] = 0.25
            df.loc[group_ind, "needs_dodge"] = False
            df.loc[group_ind, "binned_x"] = df.loc[group_ind, "final_value"]
            df.loc[group_ind, "dodge"] = 0
        else:
            df.loc[group_ind, "rect_width"] = min_rect_width
            param_names = group_df["full_name"].unique()
            for p in param_names:
                param_slice = df[df["full_name"] == p]
                sorted_values = param_slice["final_value"].sort_values()
                ind = sorted_values.index
                dist_to_left_neighbor = sorted_values.diff()
                needs_dodge = dist_to_left_neighbor < 1.25 * min_rect_width
                df.loc[ind, "needs_dodge"] = needs_dodge
                new_xs, dodge = _create_x_and_dodge(sorted_values, needs_dodge)
                df.loc[ind, "binned_x"] = new_xs
                rect_height = 0.4 * min(0.25, 1 / max(1, 0.5 * max(np.abs(dodge))))
                df.loc[ind, "rect_height"] = rect_height
                df.loc[ind, "dodge"] = 0.55 * rect_height * dodge
    df["name_with_dodge"] = df.apply(lambda x: (x["full_name"], x["dodge"]), axis=1)
    return df


def _smallest_diff_btw_params(group_df, x_range):
    param_names = group_df["full_name"].unique()
    min_dist = x_range
    for p in param_names:
        param_slice = group_df[group_df["full_name"] == p]
        values = param_slice["final_value"].sort_values()
        dist_to_left_neighbor = values.diff()
        min_dist = min(min_dist, dist_to_left_neighbor.min())
    return min_dist


def _create_x_and_dodge(val_arr, bool_arr):
    new_xs, dodge, stored_x = [], [], []
    for old_x, needs_dodge in zip(val_arr, bool_arr):
        _update_dodge(dodge, needs_dodge)
        _update_x(old_x, needs_dodge, new_xs, stored_x)

    _catch_up_x(new_xs, stored_x)
    return np.array(new_xs), np.array(dodge, dtype=int)


def _update_dodge(dodge, needs_dodge):
    if not needs_dodge:
        dodge.append(0)
    elif len(dodge) == 0:
        dodge.append(1)
    elif dodge[-1] == 0:
        dodge[-1] = 1
        dodge.append(-1)
    else:
        last_was_pos = dodge[-1] > 0
        if last_was_pos:
            dodge.append(-dodge[-1])
        else:
            dodge.append(-dodge[-1] + 1)


def _update_x(old_x, needs_dodge, new_xs, stored_x):
    if not needs_dodge:
        _catch_up_x(new_xs, stored_x)
    stored_x.append(old_x)


def _catch_up_x(new_xs, stored_x):
    if len(stored_x) != 0:
        new_xs += [np.mean(stored_x)] * len(stored_x)
        del stored_x[:]


def _add_select_tools(source, param_group_plot, point_estimate_glyph):
    select_js_kwargs = {"source": source}
    select_js_code = """
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
    select_callback = CustomJS(args=select_js_kwargs, code=select_js_code)
    # point_estimate_glyph as only renderer assures that when a point is chosen
    # only that point's model is chosen
    # this makes it impossible to choose models based on clicking confidence bands
    tap = TapTool(renderers=[point_estimate_glyph], callback=select_callback)
    param_group_plot.tools.append(tap)
    boxselect = BoxSelectTool(
        renderers=[point_estimate_glyph],
        callback=select_callback,
        overlay=BoxAnnotation(fill_alpha=0.2, fill_color="gray"),
    )
    param_group_plot.tools.append(boxselect)


def _add_hover_tool(param_group_plot, point_estimate_glyph, df):
    top_cols = ["model", "full_name", "final_value", "model_class"]
    if "conf_int_lower" in df.columns and "conf_int_upper" in df.columns:
        top_cols += ["conf_int_lower", "conf_int_upper"]
    tooltips = [(col, "@" + col) for col in top_cols]
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
