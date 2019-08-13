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

MEDIUMELECTRICBLUE = "#035096"


def comparison_plot(
    res_dict, color_dict=None, height=None, width=500, point_estimate_plot_kwargs=None
):
    """Make a comparison plot from a dictionary containing optimization results.

    Args:
        res_dict (dict): The keys are the names of different models.
            Each value is a dictinoary with the following keys and values:
                - result_df (pd.DataFrame):
                    params DataFrame returned by maximize or minimize.
                - model_class (str, optional):
                    name of the model class to which the model belongs.
                    This determines the color and checkbox entries with
                    which model classes can be selected and unselected.

        color_dict (dict):
            mapping from the model class names to colors.

        height (int):
            height of the plot.

        width (int):
            width of the plot (in pixels).
    """
    df = _df_with_all_results(res_dict)
    lower, upper, rect_widths = _create_bounds_and_rect_widths(df)
    df = _df_with_plot_specs(df, rect_widths, lower, upper, color_dict)
    heights = _determine_plot_heights(df, height)

    source_dict, figure_dict, glyph_dict = _create_comparison_plot_components(
        df=df,
        heights=heights,
        lower=lower,
        upper=upper,
        rect_widths=rect_widths,
        width=width,
    )

    plots_with_callbacks = _add_callbacks(
        source_dict=source_dict,
        figure_dict=figure_dict,
        glyph_dict=glyph_dict,
        model_classes=sorted(df["model_class"].unique()),
    )

    grid = gridplot(plots_with_callbacks, toolbar_location="right", ncols=1)
    show(grid)
    return df, grid, plots_with_callbacks


# ===========================================================================
# DATA PREP FUNCTIONS
# ===========================================================================


def _df_with_all_results(res_dict):
    """
    Build the DataFrame combining all DataFrames from the results dictionary.

    Args:
        res_dict (dict): see comparsion_plot docstring
    """
    df = pd.DataFrame()
    for model, model_dict in res_dict.items():
        result_df = _prep_result_df(model_dict, model)
        df = df.append(result_df, sort=False, ignore_index=True)

    if "conf_int_upper" not in df.columns:
        df["conf_int_upper"] = np.nan
    if "conf_int_lower" not in df.columns:
        df["conf_int_lower"] = np.nan
    return df


def _prep_result_df(model_dict, model):
    result_df = model_dict["result_df"].copy(deep=True)
    for index_name in result_df.index.names:
        if index_name in result_df.columns:
            ind_values = result_df.index.get_level_values(index_name)
            if (result_df[index_name] == ind_values).all():
                result_df.drop(columns=[index_name], inplace=True)
            else:
                raise ValueError(
                    "The index level {} of the result DataFrame of {}".format(
                        index_name, model
                    ),
                    "has different values than the column of the same name.",
                )
    result_df.reset_index(inplace=True)
    result_df["model"] = model
    result_df["model_class"] = model_dict.get("model_class", "no class")
    return result_df


def _create_bounds_and_rect_widths(df):
    """Calculate lower and upper bounds and the width of the "histogram" bars."""
    grouped = df.groupby("group")
    upper = grouped[["conf_int_upper", "value"]].max().max(axis=1)
    lower = grouped[["conf_int_lower", "value"]].min().min(axis=1)
    rect_widths = 0.02 * (upper - lower)

    return lower, upper, rect_widths


def _determine_plot_heights(df, figure_height):
    grouped = df.groupby("group")
    figure_height = _determine_figure_height(df, figure_height)
    n_params = grouped["name"].unique().apply(len)
    height_shares = n_params / n_params.sum()
    return (height_shares * figure_height).astype(int)


def _determine_figure_height(df, figure_height):
    if figure_height is None:
        n_models = len(df["model"].unique())
        n_params = len(df["name"].unique())
        height_per_plot = max(min(n_models, 60), 20)
        figure_height = height_per_plot * n_params
    return figure_height


def _df_with_plot_specs(df, rect_widths, lower, upper, color_dict):
    """Add color column, dodge column and binned_x column to *df*."""
    df = _df_with_color_column(df, color_dict)
    df["dodge"] = 0.5
    for group in df["group"].unique():
        rect_width = rect_widths.loc[group]
        bins = np.arange(
            start=lower.loc[group] - 2 * rect_width,
            stop=upper.loc[group] + 2 * rect_width,
            step=rect_width,
        )
        param_names = df[df["group"] == group]["name"].unique()
        for param in param_names:
            _add_dodge_and_binned_x(df, group, param, bins)
    df["binned_x"].fillna(df["value"], inplace=True)
    return df


def _df_with_color_column(df, color_dict):
    models = df["model_class"].unique()
    if color_dict is None:
        color_dict = {}
    color_dict = {m: color_dict.get(m, "#035096") for m in models}

    for m in models:
        if m not in color_dict.keys():
            color_dict[m] = MEDIUMELECTRICBLUE
    df["color"] = df["model_class"].replace(color_dict)
    return df


def _add_dodge_and_binned_x(df, group, param, bins):
    """For one parameter calculate for each estimate its dodge and bin."""
    param_df = df[(df["group"] == group) & (df["name"] == param)].copy(deep=True)
    param_df.sort_values(["model_class", "value"], inplace=True)
    values = param_df["value"]
    param_ind = param_df.index
    hist, edges = np.histogram(param_df["value"], bins)
    df.loc[param_ind, "lower_edge"] = values.apply(lambda x: _find_next_lower(bins, x))
    df.loc[param_ind, "upper_edge"] = values.apply(lambda x: _find_next_upper(bins, x))
    for lower, upper, n_points in zip(bins[:-1], bins[1:], hist):
        if n_points > 1:
            need_dodge = values[(lower <= values) & (values < upper)]
            ind = need_dodge.index
            df.loc[ind, "dodge"] = 0.5 + np.arange(len(ind))

    edge_cols = ["upper_edge", "lower_edge"]
    df.loc[param_ind, "binned_x"] = df.loc[param_ind, edge_cols].mean(
        axis=1, skipna=False
    )


def _find_next_lower(array, value):
    # adapted from https://stackoverflow.com/a/2566508
    candidates = array[array <= value]
    if len(candidates) > 0:
        idx = (np.abs(candidates - value)).argmin()
        return candidates[idx]
    else:
        return np.nan


def _find_next_upper(array, value):
    # adapted from https://stackoverflow.com/a/2566508
    candidates = array[array > value]
    if len(candidates) > 0:
        idx = (np.abs(candidates - value)).argmin()
        return candidates[idx]
    else:
        return np.nan


# ===========================================================================
# PLOTTING FUNCTIONS
# ===========================================================================


def _create_comparison_plot_components(df, heights, lower, upper, rect_widths, width):
    groups = heights.index.tolist()
    source_dict = {k: {} for k in groups}
    figure_dict = {k: {} for k in groups}
    glyph_dict = {k: {} for k in groups}

    for param_group in groups:
        group_df = df[df["group"] == param_group]
        param_names = sorted(group_df["name"].unique())
        for param in param_names:
            param_src = ColumnDataSource(group_df[group_df["name"] == param])
            param_plot = figure(
                title=param,
                plot_height=int(heights.loc[param_group]),
                plot_width=width,
                tools="reset,save",
                y_axis_location="right",
                x_range=[lower.loc[param_group], upper.loc[param_group]],
            )

            point_glyph = param_plot.rect(
                source=param_src,
                x="binned_x",
                y="dodge",
                width=rect_widths.loc[param_group],
                height=1,
                color="color",
                selection_color="color",
                nonselection_color="color",
                alpha=0.5,
                selection_alpha=0.7,
                nonselection_alpha=0.3,
            )
            _add_hover_tool(param_plot, point_glyph, df)

            param_plot.hbar(
                source=param_src,
                y="dodge",
                left="conf_int_lower",
                right="conf_int_upper",
                height=0.01,
                alpha=0.0,
                selection_alpha=0.7,
                nonselection_alpha=0.0,
                color="color",
                selection_color="color",
                nonselection_color="color",
            )

            _style_plot(param_plot, param, param_names, param_group)

            figure_dict[param_group][param] = param_plot
            source_dict[param_group][param] = param_src
            glyph_dict[param_group][param] = point_glyph

    return source_dict, figure_dict, glyph_dict


def _add_hover_tool(plot, point_glyph, df):
    top_cols = ["model", "name", "value", "model_class"]
    if "conf_int_lower" in df.columns and "conf_int_upper" in df.columns:
        top_cols += ["conf_int_lower", "conf_int_upper"]
    tooltips = [(col, "@" + col) for col in top_cols]
    hover = HoverTool(renderers=[point_glyph], tooltips=tooltips)
    plot.tools.append(hover)


def _add_callbacks(source_dict, figure_dict, glyph_dict, model_classes):
    """Add checkbox for selecting model classes and tap tools."""
    all_src = _flatten_dict(source_dict)
    plots_with_callbacks = [
        _create_checkbox(widget_labels=model_classes, all_src=all_src)
    ]

    for group, param_to_source in source_dict.items():
        for param, param_src in param_to_source.items():
            param_plot = figure_dict[group][param]
            point_glyph = glyph_dict[group][param]
            other_src = _flatten_dict(source_dict, param)
            _add_select_tools(
                current_src=param_src,
                other_src=other_src,
                param_plot=param_plot,
                point_glyph=point_glyph,
            )
            plots_with_callbacks.append(param_plot)

    return plots_with_callbacks


def _flatten_dict(nested_dict, exclude_key=None):
    flattened = []
    for inner_dict in nested_dict.values():
        for inner_key, inner_val in inner_dict.items():
            if exclude_key is None or inner_key != exclude_key:
                flattened.append(inner_val)
    return flattened


def _add_select_tools(current_src, other_src, param_plot, point_glyph):
    select_js_kwargs = {"current_src": current_src, "other_src": other_src}
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


def _create_checkbox(widget_labels, all_src):
    widget_js_kwargs = {"all_src": all_src, "group_list": widget_labels}
    widget_js_code = """
    // adapted from https://stackoverflow.com/a/36145278

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


def _style_plot(fig, param, param_names, group):
    if param == param_names[0]:
        group_title = Title(
            text="Comparison Plot of {} Parameters".format(group.title()),
            align="center",
        )
        fig.add_layout(group_title, "above")

    _prettify_x_axis(fig, param, param_names)

    fig.title.vertical_align = "middle"
    fig.title.align = "center"
    fig.title.offset = 0
    fig.outline_line_color = None
    fig.xgrid.visible = False
    fig.yaxis.minor_tick_line_color = None
    fig.xaxis.minor_tick_line_color = None
    fig.yaxis.axis_line_color = None
    fig.yaxis.major_tick_line_color = None
    fig.sizing_mode = "scale_width"
    fig.title_location = "left"


def _prettify_x_axis(fig, param, param_names):
    if param != param_names[-1]:
        fig.xaxis.visible = False
    else:
        fig.xaxis.axis_line_color = None
    xmin = fig.x_range.start
    xmax = fig.x_range.end
    fig.line([xmin, xmax], [0, 0], line_color="black")
