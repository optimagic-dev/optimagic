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

from estimagic.visualization.comparison_plot_data_preparation import (
    comparison_plot_inputs,
)


def comparison_plot(
    results,
    color_dict=None,
    height=None,
    width=500,
    axis_for_every_parameter=False,
    x_padding=0.1,
    num_bins=50,
):
    """Make a comparison plot from a dictionary containing optimization results.

    Args:
        results (list): List of estimagic optimization results where the info
            can have been extended with 'model' and 'model_name'
        color_dict (dict):
            mapping from the model class names to colors.
        height (int):
            height of the plot.
        width (int):
            width of the plot (in pixels).
        axis_for_every_parameter (bool):
            if False the x axis is only shown once for every group of parameters.
        x_padding (float): the x_range is extended on each side by x_padding
            times the range of the data
        num_bins (int): number of bins

    Returns:
        source_dfs, grid
    """
    source_dfs, plot_info = comparison_plot_inputs(
        results=results,
        x_padding=x_padding,
        num_bins=num_bins,
        color_dict=color_dict,
        fig_height=height,
    )

    source_dict, figure_dict, glyph_dict = _create_comparison_plot_components(
        source_dfs=source_dfs,
        plot_info=plot_info,
        axis_for_every_parameter=axis_for_every_parameter,
        width=width,
    )

    model_classes = sorted({res.info["model_class"] for res in results})

    plots_with_callbacks = _add_callbacks(
        source_dict=source_dict,
        figure_dict=figure_dict,
        glyph_dict=glyph_dict,
        model_classes=model_classes,
    )

    grid = gridplot(plots_with_callbacks, toolbar_location="right", ncols=1)
    show(grid)
    return source_dfs, grid


def _create_comparison_plot_components(
    source_dfs, plot_info, width, axis_for_every_parameter
):

    source_dict = {k: {} for k in source_dfs.keys()}
    figure_dict = {k: {} for k in source_dfs.keys()}
    glyph_dict = {k: {} for k in source_dfs.keys()}

    for group, param_to_df in source_dfs.items():
        group_info = plot_info["group_info"][group]
        title_fig = figure(
            title=Title(
                text="Comparison Plot of " + group.title() + " Parameters",
                align="center",
                text_font_size="15pt",
            ),
            plot_height=50,
            plot_width=width,
            tools="reset,save",
        )
        _style_title_fig(title_fig)

        figure_dict[group]["__title__"] = title_fig

        for i, (param, df) in enumerate(param_to_df.items()):
            param_src = ColumnDataSource(df)
            param_plot = figure(
                title=df["name"].unique()[0],
                plot_height=plot_info["plot_height"],
                plot_width=width,
                tools="reset,save",
                y_axis_location="left",
                x_range=group_info["x_range"],
                y_range=plot_info["y_range"],
            )

            point_glyph = param_plot.rect(
                source=param_src,
                x="binned_x",
                y="dodge",
                width=group_info["width"],
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

            is_last = i == len(param_to_df)
            _style_plot(
                fig=param_plot,
                last=is_last,
                axis_for_every_parameter=axis_for_every_parameter,
            )

            figure_dict[group][param] = param_plot
            source_dict[group][param] = param_src
            glyph_dict[group][param] = point_glyph

    return source_dict, figure_dict, glyph_dict


def _add_hover_tool(plot, point_glyph, df):
    top_cols = ["model", "name", "value"]
    optional_cols = ["model_class", "conf_int_lower", "conf_int_upper"]
    for col in optional_cols:
        if len(df[col].unique()) > 1:
            top_cols.append(col)
    tooltips = [(col, "@" + col) for col in top_cols]
    hover = HoverTool(renderers=[point_glyph], tooltips=tooltips)
    plot.tools.append(hover)


def _add_callbacks(source_dict, figure_dict, glyph_dict, model_classes):
    """Add checkbox for selecting model classes and tap tools."""
    all_src = _flatten_dict(source_dict)
    plots_with_callbacks = [
        _create_checkbox(widget_labels=model_classes, all_src=all_src)
    ]

    for group, param_to_figure in figure_dict.items():
        for param, param_plot in param_to_figure.items():
            if param != "__title__":
                param_src = source_dict[group][param]
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


def _style_title_fig(fig):
    fig.line([], [])  # add renderer to avoid warning
    fig.ygrid.visible = False
    fig.xgrid.visible = False
    fig.outline_line_color = None
    fig.yaxis.axis_line_color = None
    fig.xaxis.axis_line_color = None


def _style_plot(fig, last, axis_for_every_parameter):
    _style_x_axis(fig=fig, last=last, axis_for_every_parameter=axis_for_every_parameter)
    _style_y_axis(fig=fig)

    fig.title.vertical_align = "top"
    fig.title.text_alpha = 70
    fig.title.text_font_style = "normal"
    fig.outline_line_color = None
    fig.min_border_top = 20
    fig.min_border_bottom = 20
    fig.xgrid.visible = False
    fig.ygrid.visible = False
    fig.sizing_mode = "scale_width"


def _style_x_axis(fig, last, axis_for_every_parameter):
    if not axis_for_every_parameter:
        if last:
            fig.xaxis.visible = False
        else:
            fig.xaxis.axis_line_color = None
        xmin = fig.x_range.start
        xmax = fig.x_range.end
        fig.line([xmin, xmax], [0, 0], line_color="black")
    fig.xaxis.minor_tick_line_color = None


def _style_y_axis(fig):
    fig.yaxis.minor_tick_line_color = None
    fig.yaxis.axis_line_color = None
    fig.yaxis.major_tick_line_color = None
