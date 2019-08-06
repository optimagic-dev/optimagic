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
from bokeh.models import TapTool
from bokeh.models import Title
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
    plot_specs = _plot_specs(df, width, height)
    _add_rectangle_specs_to_df(df, plot_specs)
    source = ColumnDataSource(df)

    groups = plot_specs.keys()
    name_to_source = {k: {} for k in groups}
    name_to_figure = {k: {} for k in groups}
    name_to_point_glyph = {k: {} for k in groups}
    name_to_ci_glyphs = {k: {} for k in groups}
    finished_plot_list = []

    for param_group in groups:
        group_specs = plot_specs[param_group]
        rect_width = group_specs["rect_width"]
        param_names = sorted(df[df["group"] == param_group]["full_name"].unique())
        for param in param_names:
            param_src = ColumnDataSource(df[df["full_name"] == param])
            name_to_source[param_group][param] = param_src
            param_plot = figure(
                title=param,
                title_location="left",
                y_axis_location="right",
                plot_height=int(group_specs["plot_height"]),
                plot_width=width,
                tools="reset,save",
                x_range=[group_specs["lower"], group_specs["upper"]],
            )
            name_to_figure[param_group][param] = param_plot

            point_glyph = param_plot.rect(
                source=param_src,
                x="binned_x",
                y="dodge",
                width=rect_width,
                height=1,
                color="color",
                selection_color="color",
                nonselection_color="color",
                alpha=0.5,
                selection_alpha=0.7,
                nonselection_alpha=0.3,
            )
            name_to_point_glyph[param_group][param] = point_glyph

            name_to_ci_glyphs[param_group][param] = []
            if "conf_int_lower" in df.columns and "conf_int_upper" in df.columns:
                ci_glyph = param_plot.hbar(
                    source=param_src,
                    y="dodge",
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
                name_to_ci_glyphs[param_group][param].append(ci_glyph)

            _add_hover_tool(param_plot, point_glyph, df)
            hide_x_axis = param != param_names[-1]
            _style_plot(param_plot, hide_x_axis=hide_x_axis)

    for param_group in groups:
        param_names = sorted(df[df["group"] == param_group]["full_name"].unique())
        for param in param_names:
            param_src = name_to_source[param_group][param]
            param_plot = name_to_figure[param_group][param]
            point_glyph = name_to_point_glyph[param_group][param]
            other_src = []
            for g in groups:
                for p, src in name_to_source[g].items():
                    if p != param:
                        other_src.append(src)
            _add_select_tools(
                current_src=param_src,
                full_src=source,
                other_src=other_src,
                param_plot=param_plot,
                point_glyph=point_glyph,
            )

            if param == param_names[0]:
                param_plot.add_layout(
                    Title(
                        text="Comparison Plot of {} Parameters".format(
                            param_group.title()
                        ),
                        align="center",
                    ),
                    "above",
                )
            finished_plot_list.append(param_plot)

    all_src = [param_src] + other_src
    if "model_class" in df.columns:
        cb_group = _create_checkbox(
            widget_labels=sorted(df["model_class"].unique()), all_src=all_src
        )
        finished_plot_list = [cb_group] + finished_plot_list

    grid = gridplot(finished_plot_list, toolbar_location="right", ncols=1)
    show(grid)
    return df, grid, finished_plot_list


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


def _plot_specs(df, figure_width, figure_height):
    figure_height = _determine_figure_height(df, figure_height)
    nr_groups = len(df["group"].unique())
    available_height = figure_height - nr_groups * 50
    nr_params = len(df["full_name"].unique())
    plot_specs = {}

    group_names = df["group"].unique()
    for group in group_names:
        plot_specs[group] = {}
        group_df = df[df["group"] == group]
        param_names = group_df["full_name"].unique()
        lower, upper = _determine_lower_and_upper_bound(group_df)
        plot_specs[group]["lower"] = lower
        plot_specs[group]["upper"] = upper
        plot_specs[group]["rect_width"] = (upper - lower) / 50
        plot_height = (len(param_names) / nr_params) * available_height
        plot_specs[group]["plot_height"] = plot_height
    return plot_specs


def _determine_figure_height(df, figure_height):
    if figure_height is None:
        nr_models = len(df["model"].unique())
        nr_params = len(df["full_name"].unique())
        figure_height = 8 * max(min(nr_models, 60), 10) * nr_params
    return figure_height


def _determine_lower_and_upper_bound(group_df):
    if "conf_int_upper" in group_df.columns:
        max_value = group_df["conf_int_upper"].max()
    else:
        max_value = group_df["final_value"].max()
    if "conf_int_lower" in group_df.columns:
        min_value = group_df["conf_int_lower"].min()
    else:
        min_value = group_df["final_value"].min()

    return min_value, max_value


def _add_rectangle_specs_to_df(df, plot_specs):
    for group in plot_specs.keys():
        rect_width = plot_specs[group]["rect_width"]
        param_names = df[df["group"] == group]["full_name"].unique()
        for param in param_names:
            _add_dodge_and_binned_x(df, param, rect_width)


def _add_dodge_and_binned_x(df, param, rect_width):
    param_df = df[df["full_name"] == param]
    values = param_df["final_value"]

    bins = np.arange(
        start=values.min() - 2 * rect_width,
        stop=values.max() + 2 * rect_width,
        step=1.1 * rect_width,
    )
    hist, edges = np.histogram(values, bins)
    for lower, upper, nr_points in zip(edges[:-1], edges[1:], hist):
        if nr_points > 1:
            point_df = param_df[param_df["final_value"].between(lower, upper)]
            if "model_class" in point_df.columns:
                ind = point_df.sort_values("model_class").index
            else:
                ind = point_df.sort_values("final_value").index
            df.loc[ind, "dodge"] = 0.5 + np.arange(len(ind))
    df.loc[param_df.index, "dodge"].fillna(0.5, inplace=True)

    df.loc[param_df.index, "lower_edges"] = param_df["final_value"].apply(
        lambda x: _find_nearest_lower(bins, x)
    )
    df.loc[param_df.index, "upper_edges"] = param_df["final_value"].apply(
        lambda x: _find_nearest_upper(bins, x)
    )
    df.loc[param_df.index, "binned_x"] = (
        df.loc[param_df.index, "upper_edges"] + df.loc[param_df.index, "lower_edges"]
    ) / 2


def _find_nearest_lower(array, value):
    # adapted from https://stackoverflow.com/a/2566508
    candidates = array[array <= value]
    idx = (np.abs(candidates - value)).argmin()
    return candidates[idx]


def _find_nearest_upper(array, value):
    # adapted from https://stackoverflow.com/a/2566508
    candidates = array[array >= value]
    idx = (np.abs(candidates - value)).argmin()
    return candidates[idx]


def _add_select_tools(current_src, full_src, other_src, param_plot, point_glyph):
    select_js_kwargs = {
        "full_src": full_src,
        "current_src": current_src,
        "other_src": other_src,
    }
    select_js_code = """
    // adapted from https://stackoverflow.com/a/44996422

    var chosen = current_src.selected.indices;
    if (typeof(chosen) == "number"){
        var chosen = [chosen]
    };

    var chosen_models = [];

    for (var i = 0; i < chosen.length; ++ i){
        chosen_models.push(current_src.data['model'][chosen[i]])
    };

    var chosen_models_indices = [];
    for (var i = 0; i < current_src.data['index'].length; ++ i){
        if (chosen_models.includes(current_src.data['model'][i])){
            chosen_models_indices.push(i)
        };
    };
    current_src.selected.indices = chosen_models_indices;
    current_src.change.emit();

    var chosen_models_indices = [];
    for (var i = 0; i < full_src.data['index'].length; ++ i){
        if (chosen_models.includes(full_src.data['model'][i])){
            chosen_models_indices.push(i)
        };
    };
    full_src.selected.indices = chosen_models_indices;
    full_src.change.emit();

    for (var i = 0; i < other_src.length; ++i){
        var chosen_models_indices = [];
        for (var j = 0; j < other_src[i].data['index'].length; ++ j){
            if (chosen_models.includes(other_src[i].data['model'][j])){
                chosen_models_indices.push(j)
            };
        };
        other_src[i].selected.indices = chosen_models_indices;
        other_src[i].change.emit();
    };

    """
    select_callback = CustomJS(args=select_js_kwargs, code=select_js_code)
    # point_glyph as only renderer assures that when a point is chosen
    # only that point's model is chosen
    # this makes it impossible to choose models based on clicking confidence bands
    tap = TapTool(renderers=[point_glyph], callback=select_callback)
    param_plot.tools.append(tap)
    boxselect = BoxSelectTool(renderers=[point_glyph], callback=select_callback)
    param_plot.tools.append(boxselect)


def _add_hover_tool(plot, point_glyph, df):
    top_cols = ["model", "full_name", "final_value", "model_class"]
    if "conf_int_lower" in df.columns and "conf_int_upper" in df.columns:
        top_cols += ["conf_int_lower", "conf_int_upper"]
    tooltips = [(col, "@" + col) for col in top_cols]
    hover = HoverTool(renderers=[point_glyph], tooltips=tooltips)
    plot.tools.append(hover)


def _create_checkbox(widget_labels, all_src):
    widget_js_kwargs = {"all_src": all_src, "group_list": widget_labels}
    widget_js_code = """
    // https://stackoverflow.com/a/36145278

    var chosen_inds = cb_obj.active;

    var chosen_widget_groups = [];

    for (var i = 0; i < group_list.length; ++ i){
        if (chosen_inds.includes(i)){
            chosen_widget_groups.push(group_list[i])
        };
    };

    for (var j = 0; j < all_src.length; ++ j){

        to_select_inds = []

        for (var i = 0; i < all_src[j].data['index'].length; ++ i){
            if (chosen_widget_groups.includes(all_src[j].data['model_class'][i])){
                to_select_inds.push(i)
            };
        };

        all_src[j].selected.indices = to_select_inds;
        all_src[j].change.emit();
    };

    """
    widget_callback = CustomJS(args=widget_js_kwargs, code=widget_js_code)
    cb_group = CheckboxGroup(
        labels=widget_labels,
        active=[0] * len(widget_labels),
        callback=widget_callback,
        inline=True,
    )
    return cb_group


def _style_plot(fig, hide_x_axis):
    fig.title.vertical_align = "middle"
    fig.title.align = "center"
    fig.title.offset = 0
    fig.outline_line_color = None
    fig.xgrid.visible = False
    fig.yaxis.minor_tick_line_color = None
    fig.xaxis.minor_tick_line_color = None
    fig.yaxis.axis_line_color = None
    fig.yaxis.major_tick_line_color = None
    if hide_x_axis:
        fig.xaxis.visible = False
    else:
        fig.xaxis.axis_line_color = None
    xmin = fig.x_range.start
    xmax = fig.x_range.end
    fig.line([xmin, xmax], [0, 0], line_color="black")
    fig.sizing_mode = "scale_width"
