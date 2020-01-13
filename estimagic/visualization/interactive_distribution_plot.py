"""
Draw interactive distribution plots that allow to identify particular obseservations.

One can think of the interactive distribution plot as a clickable histogram.
The main difference to a histogram is that in this type of plot
every bar is a stack of bricks where each brick identifies a particular observation.
By hovering or clicking on a particular brick you can learn more about that observation
making it easy to identify and analyze patterns.

Estimagic uses interactive distribution plots for two types of visualizations:
1. Parameter comparison plots
2. Loglikelihood contribution plots

"""
import warnings

import numpy as np
from bokeh.layouts import gridplot
from bokeh.models import CDSView
from bokeh.models import ColumnDataSource
from bokeh.models import Title
from bokeh.models.callbacks import CustomJS
from bokeh.models.widgets import CheckboxGroup
from bokeh.models.widgets import RangeSlider
from bokeh.plotting import figure
from bokeh.plotting import show

from estimagic.visualization.add_histogram_columns_to_tidy_df import (
    add_histogram_columns_to_tidy_df,
)
from estimagic.visualization.interactive_distribution_plot_callbacks import (
    add_single_plot_tools,
)
from estimagic.visualization.interactive_distribution_plot_callbacks import (
    add_value_slider_in_front,
)
from estimagic.visualization.interactive_distribution_plot_callbacks import (
    create_filters,
)


def interactive_distribution_plot(
    df,
    value_col,
    id_col,
    group_cols=None,
    subgroup_col=None,
    lower_bound_col=None,
    upper_bound_col=None,
    figure_height=None,
    width=500,
    x_padding=0.1,
    num_bins=50,
    axis_for_every_parameter=False,
):
    """Create an interactive distribution plot from a tidy DataFrame.

    Args:
        df (pd.DataFrame):
            Tidy DataFrame.
            see: http://vita.had.co.nz/papers/tidy-data.pdf
        value_col (str):
            Name of the column for which to draw the histogram.
            In case of a parameter comparison plot this would be the "value" column
            of the params DataFrame returned by maximize or minimize.
        id_col (str):
            Name of the column that identifies
            which values belong to the same observation.
            In case of a parameter comparison plot
            this would be the "model_name" column.
        group_cols (list):
            Name of the columns that identify groups that will be plotted together.
            In case of a parameter comparison plot this would be the parameter group
            and parameter name.
        subgroup_col (str, optional):
            Name of a column according to whose values individual bricks will be
            color coded. The selection which column is the subgroup_col
            can be changed in the plot from a dropdown menu.
        lower_bound_col (str, optional):
            Name of the column identifying the lower bound of the whisker.
        upper_bound_col (str, optional):
            Name of the column identifying the upper bound of the whisker.
        figure_height (int, optional):
            height of the figure (i.e. of all plots together, in pixels).
        width (int, optional):
            width of the figure (in pixels).
        x_padding (float, optional):
            the x_range is extended on each side by this factor of the range of the data
        num_bins (int, optional):
            number of bins
        axis_for_every_parameter (bool, optional):
            if False the x axis is only shown once for every group of parameters.

    Returns:
        source (bokeh.models.ColumnDataSource): data underlying the plots
        gridplot (bokeh.layouts.Column): grid of the distribution plots.

    """
    if group_cols is None:
        group_cols = []
    elif type(group_cols) == str:
        group_cols = [group_cols]

    hist_data = add_histogram_columns_to_tidy_df(
        df=df,
        group_cols=group_cols,
        value_col=value_col,
        subgroup_col=subgroup_col,
        id_col=id_col,
        num_bins=num_bins,
        x_padding=x_padding,
    )

    if figure_height is None:
        plot_height = 200
    else:
        plot_height = _determine_plot_height(
            figure_height=figure_height, data=hist_data, group_cols=group_cols,
        )

    source, plots = _create_plots(
        df=hist_data,
        value_col=value_col,
        group_cols=group_cols,
        subgroup_col=subgroup_col,
        id_col=id_col,
        lower_bound_col=lower_bound_col,
        upper_bound_col=upper_bound_col,
        plot_height=plot_height,
        width=width,
        axis_for_every_parameter=axis_for_every_parameter,
    )
    grid = gridplot(plots, toolbar_location="right", ncols=1)
    show(grid)
    return source, grid


# =====================================================================================
#                                    PLOT FUNCTIONS
# =====================================================================================


def _create_plots(
    df,
    value_col,
    group_cols,
    subgroup_col,
    id_col,
    lower_bound_col,
    upper_bound_col,
    plot_height,
    width,
    axis_for_every_parameter,
):

    source = ColumnDataSource(df)

    if subgroup_col is not None:
        widgets = _create_group_widgets(
            df=df,
            source=source,
            subgroup_col=subgroup_col,
            lower_bound_col=lower_bound_col,
            upper_bound_col=upper_bound_col,
        )
    else:
        widgets = (None, None)

    plots = [wid for wid in widgets if wid is not None]

    if len(group_cols) == 0:
        plots.append(title_fig(group_type="All Parameters", group_name=""))
        gb = [(None, df)]
        old_group_tup = None

    else:
        gb = df.groupby(group_cols)
        old_group_tup = tuple([np.nan] * len(group_cols))
        if len(group_cols) == 1:
            plots.append(title_fig(group_type=group_cols[0], group_name=""))

    for group_tup, group_df in gb:
        plots, new_group, old_group_tup = _check_and_handle_group_switch(
            group_cols=group_cols,
            old_group_tup=old_group_tup,
            group_tup=group_tup,
            df=df,
            plots=plots,
        )

        plots = _add_param_plot(
            plots=plots,
            df=df,
            id_col=id_col,
            group_cols=group_cols,
            subgroup_col=subgroup_col,
            lower_bound_col=lower_bound_col,
            upper_bound_col=upper_bound_col,
            source=source,
            group_tup=group_tup,
            group_df=group_df,
            new_group=new_group,
            plot_height=plot_height,
            widgets=widgets,
            width=width,
            axis_for_every_parameter=axis_for_every_parameter,
        )

    plots = add_value_slider_in_front(
        df=df,
        value_col=value_col,
        lower_bound_col=lower_bound_col,
        upper_bound_col=upper_bound_col,
        plots=plots,
    )
    return source, plots


def _create_group_widgets(df, source, lower_bound_col, upper_bound_col, subgroup_col):
    sr = df[subgroup_col]
    checkboxes = None
    slider = None
    if sr.dtype == float:
        sorted_uniques = np.array(sorted(sr.unique()[::-1]))
        min_dist_btw_vals = (sorted_uniques[1:] - sorted_uniques[:-1]).min()
        slider = RangeSlider(
            start=sr.min(),
            end=sr.max(),
            value=(sr.min(), sr.max()),
            step=min_dist_btw_vals,
            title=subgroup_col.title(),
        )
        slider.js_on_change(
            "value", CustomJS(code="source.change.emit();", args={"source": source})
        )
    elif sr.dtype == object:
        checkbox_labels = sr.unique().tolist()
        checkboxes = CheckboxGroup(
            labels=checkbox_labels, active=list(range(len(checkbox_labels)))
        )
        checkboxes.js_on_change(
            "active", CustomJS(code="source.change.emit();", args={"source": source})
        )

    return (
        checkboxes,
        slider,
    )


def _check_and_handle_group_switch(group_cols, old_group_tup, group_tup, df, plots):
    if len(group_cols) < 2:
        new_group = False
    else:
        new_group = old_group_tup[:-1] != group_tup[:-1]
        if new_group:
            plots = _write_titles(
                plots=plots,
                group_cols=group_cols,
                old_group_tup=old_group_tup,
                group_tup=group_tup,
            )
            old_group_tup = group_tup
    return plots, new_group, old_group_tup


def _add_param_plot(
    plots,
    df,
    id_col,
    group_cols,
    subgroup_col,
    lower_bound_col,
    upper_bound_col,
    source,
    group_tup,
    group_df,
    new_group,
    plot_height,
    widgets,
    width,
    axis_for_every_parameter,
):
    x_range = _calculate_x_range(
        df=df,
        lower_bound_col=lower_bound_col,
        upper_bound_col=upper_bound_col,
        group_cols=group_cols,
        group_tup=group_tup,
    )
    filters = create_filters(
        source=source,
        group_df=group_df,
        subgroup_col=subgroup_col,
        id_col=id_col,
        widgets=widgets,
    )
    view = CDSView(source=source, filters=filters)

    if len(group_cols) == 0:
        plot_title = ""
    elif len(group_cols) == 1:
        plot_title = str(group_tup)
    else:
        plot_title = str(group_tup[-1])

    param_plot = _create_base_plot(
        title=plot_title,
        group_df=group_df,
        source=source,
        view=view,
        x_range=x_range,
        plot_height=plot_height,
        width=width,
        id_col=id_col,
    )

    param_plot = _add_ci_bars_if_present(
        param_plot=param_plot,
        source=source,
        view=view,
        lower_bound_col=lower_bound_col,
        upper_bound_col=upper_bound_col,
    )

    _style_plot(
        fig=param_plot,
        new_group=new_group,
        axis_for_every_parameter=axis_for_every_parameter,
    )
    plots.append(param_plot)
    return plots


def _calculate_x_range(df, lower_bound_col, upper_bound_col, group_cols, group_tup):
    if len(group_cols) < 2:
        whole_group_df = df
    elif len(group_cols) == 2:
        whole_group_df = df[df[group_cols[0]] == group_tup[0]]
    else:
        whole_group_df = df[(df[group_cols[:-1]] == group_tup[:-1]).all(axis=1)]
    rect_width = whole_group_df["rect_width"].unique()[0]
    group_min = whole_group_df["binned_x"].min() - rect_width
    group_max = whole_group_df["binned_x"].max() + rect_width
    if lower_bound_col is not None:
        group_min = min(group_min, whole_group_df[lower_bound_col].min())
    if upper_bound_col is not None:
        group_max = max(group_max, whole_group_df[upper_bound_col].max())

    assert np.isfinite(group_min) and np.isfinite(group_max), "{}".format(
        group_tup[:-1]
    )
    return group_min, group_max


def _create_base_plot(
    title, group_df, source, view, x_range, plot_height, width, id_col
):
    param_plot = figure(
        title=str(title).title(),
        plot_height=plot_height,
        plot_width=width,
        tools="reset,save",
        y_axis_location="left",
        x_range=x_range,
    )
    point_glyph = param_plot.rect(
        source=source,
        view=view,
        x="binned_x",
        width="rect_width",
        y="dodge",
        height=1,
        color="color",
        selection_color="color",
        nonselection_color="color",
        alpha=0.5,
        selection_alpha=0.7,
        nonselection_alpha=0.1,
    )

    param_plot = add_single_plot_tools(param_plot, point_glyph, source, id_col)
    return param_plot


def _add_ci_bars_if_present(param_plot, source, view, lower_bound_col, upper_bound_col):
    if lower_bound_col is not None and upper_bound_col is not None:
        param_plot.hbar(
            source=source,
            view=view,
            y="dodge",
            left=lower_bound_col,
            right=upper_bound_col,
            height=0.01,
            alpha=0.0,
            selection_alpha=0.7,
            nonselection_alpha=0.0,
            color="color",
            selection_color="color",
            nonselection_color="color",
        )
    return param_plot


# =====================================================================================
#                                   STYLING FUNCTIONS
# =====================================================================================


def _determine_plot_height(figure_height, data, group_cols):
    """Calculate the height alloted to each plot in pixels.

    Args:
        figure_height (int): height of the entire figure in pixels
        data (pd.DataFrame): DataFrame of the

    Returns:
        plot_height (int): Plot height in pixels.

    """
    if figure_height is None:
        figure_height = int(max(min(30 * data["dodge"].max(), 1000), 100))

    if len(group_cols) > 1:
        n_groups = len(data.groupby(group_cols[:-1]))
        n_plots = len(data.groupby(group_cols))
    elif len(group_cols) == 1:
        n_groups = 0
        n_plots = len(data.groupby(group_cols))
    else:
        n_groups = 0
        n_plots = 1
    space_of_titles = n_groups * 50
    available_space = figure_height - space_of_titles
    plot_height = int(available_space / n_plots)
    if plot_height < 20:
        warnings.warn(
            "The figure height you specified results in very small ({}) ".format(
                plot_height
            )
            + "plots which may not render well. Adjust the figure height "
            "to a larger value or set it to None to get a larger plot. "
            "Alternatively, you can click on the Reset button "
            "on the right of the plot and your plot should render correctly."
        )
    return plot_height


def _style_plot(fig, new_group, axis_for_every_parameter):
    _style_x_axis(
        fig=fig, new_group=new_group, axis_for_every_parameter=axis_for_every_parameter
    )
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


def _style_x_axis(fig, new_group, axis_for_every_parameter):
    if not axis_for_every_parameter:
        if new_group:
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


def title_fig(group_type, group_name, width=500, level=0):
    title = Title(
        text="{group_type} {group_name}".format(
            group_type=str(group_type).title(), group_name=str(group_name).title()
        ),
        align="center",
        text_font_size="{}pt".format(15 - 2 * level),
    )
    fig = figure(title=title, plot_height=50, plot_width=width, tools="reset,save",)
    fig.line([], [])  # add renderer to avoid warning
    fig.ygrid.visible = False
    fig.xgrid.visible = False
    fig.outline_line_color = None
    fig.yaxis.axis_line_color = None
    fig.xaxis.axis_line_color = None

    return fig


def _write_titles(plots, group_cols, old_group_tup, group_tup):
    for level in range(len(group_cols) - 1):
        old_name = old_group_tup[level]
        new_name = group_tup[level]
        if old_name != new_name:
            plots.append(title_fig(group_cols[level], new_name, level=level))
    return plots
