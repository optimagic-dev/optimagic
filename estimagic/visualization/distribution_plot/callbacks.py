"""Callback functions for the interactive distribution plot."""
import bokeh
import numpy as np
from bokeh.layouts import row
from bokeh.models import BoxSelectTool
from bokeh.models import CDSView
from bokeh.models import HoverTool
from bokeh.models import IndexFilter
from bokeh.models import TapTool
from bokeh.models.callbacks import CustomJS
from bokeh.models.filters import CustomJSFilter
from bokeh.models.widgets import CheckboxGroup
from bokeh.models.widgets import Div
from bokeh.models.widgets import RangeSlider


# =====================================================================================
# Create Group Widgets
# =====================================================================================


def create_group_widgets(source, subgroup_col):
    if subgroup_col is None:
        return []
    slider = _make_slider(source=source, subgroup_col=subgroup_col)
    checkbox_group = _make_checkbox_group(source=source, subgroup_col=subgroup_col)
    if slider is not None:
        return [slider]
    elif checkbox_group is not None:
        return [checkbox_group]
    else:
        raise AttributeError("dtype of the subgroup column must be object or float.")


def _make_slider(source, subgroup_col):
    arr = source.data[subgroup_col]
    if arr.dtype == float:
        sorted_uniques = np.array(sorted(set(arr)))
        min_dist_btw_vals = (sorted_uniques[1:] - sorted_uniques[:-1]).min()
        slider = RangeSlider(
            start=arr.min() - 0.05 * arr.min(),
            end=arr.max() + 0.05 * arr.max(),
            value=(arr.min(), arr.max()),
            step=min_dist_btw_vals,
            title=subgroup_col.title(),
            name="subgroup_slider",
        )
        slider.js_on_change(
            "value", CustomJS(code="source.change.emit();", args={"source": source})
        )
        return slider


def _make_checkbox_group(source, subgroup_col):
    arr = source.data[subgroup_col]
    if arr.dtype == object:
        checkbox_title = Div(text=str(subgroup_col).title() + ": ")
        checkbox_labels = sorted(set(arr))
        actives = list(range(len(checkbox_labels)))
        checkboxes = CheckboxGroup(
            labels=checkbox_labels,
            active=actives,
            inline=True,
            name="subgroup_checkbox",
        )
        checkboxes.js_on_change(
            "active", CustomJS(code="source.change.emit();", args={"source": source})
        )
        return row(checkbox_title, checkboxes)


# =====================================================================================


def create_view(source, group_df, subgroup_col, widget):
    filters = [IndexFilter(group_df.index)]
    if type(widget) == bokeh.models.layouts.Row:
        checkboxes = widget.children[1]
        group_slider = None
    elif type(widget) == bokeh.models.widgets.RangeSlider:
        checkboxes = None
        group_slider = widget
    else:
        # no widget
        checkboxes = None
        group_slider = None
    if checkboxes is not None:
        checkbox_filter = _create_checkbox_filter(checkboxes, source, subgroup_col)
        filters.append(checkbox_filter)
    if group_slider is not None:
        group_slider_filter = _create_slider_filter(group_slider, source, subgroup_col)
        filters.append(group_slider_filter)
    view = CDSView(source=source, filters=filters)
    return view


def add_single_plot_tools(param_plot, point_glyph, source, id_col):
    param_plot = _add_hover_tool(param_plot, point_glyph, source)
    param_plot = _add_select_tools(param_plot, point_glyph, source, id_col)
    return param_plot


def add_value_slider_in_front(df, value_col, lower_bound_col, upper_bound_col, plots):
    val_min = df[value_col].min()
    val_max = df[value_col].max()
    if lower_bound_col is not None:
        val_min = min(val_min, df[lower_bound_col].min())
    if upper_bound_col is not None:
        val_max = max(val_max, df[upper_bound_col].max())
    x_range = val_max - val_min
    value_column_slider = RangeSlider(
        start=val_min - 0.02 * x_range,
        end=val_max + 0.02 * x_range,
        value=(val_min, val_max),
        step=x_range / 500,
        title=value_col.title(),
    )

    code = """
        var lower_end = cb_obj.value[0]
        var upper_end = cb_obj.value[1]

        for (var i = 0; i < plots.length; ++ i){
            plots[i].x_range.start = lower_end;
            plots[i].x_range.end = upper_end;
        }
    """

    callback = CustomJS(args={"plots": plots[1:]}, code=code)
    value_column_slider.js_on_change("value", callback)
    return [value_column_slider] + plots


def _create_checkbox_filter(checkboxes, source, subgroup_col):
    code = (
        """
    let selected = checkboxes.active.map(i=>checkboxes.labels[i]);
    let indices = [];
    let column = source.data."""
        + subgroup_col
        + """;
    for(let i=0; i<column.length; i++){
        if(selected.includes(column[i])){
            indices.push(i);
        }
    }
    return indices;
    """
    )

    checkbox_filter = CustomJSFilter(
        code=code, args={"checkboxes": checkboxes, "source": source}
    )

    return checkbox_filter


def _create_slider_filter(slider, source, column):
    code = (
        """
    let lower_bound = slider.value[0];
    let upper_bound = slider.value[1];
    let indices = [];
    let column = source.data."""
        + column
        + """;
    for(let i=0; i<column.length; i++){
        if(lower_bound <= column[i]){
            if (column[i] <= upper_bound){
                indices.push(i);
            }
        }
    }
    return indices;
    """
    )

    slider_filter = CustomJSFilter(code=code, args={"slider": slider, "source": source})

    return slider_filter


def _add_hover_tool(param_plot, point_glyph, source):
    skip = ["dodge", "color", "binned_x", "level_0", "index"] + [
        x for x in source.column_names if x.startswith("index__")
    ]
    to_display = [
        col
        for col in source.column_names
        if len(set(source.data[col])) > 1 and col not in skip
    ]
    tooltips = [(col, "@" + col) for col in to_display]
    hover = HoverTool(renderers=[point_glyph], tooltips=tooltips)
    param_plot.tools.append(hover)
    return param_plot


def _add_select_tools(param_plot, point_glyph, source, id_col):
    select_kwargs = {"source": source}
    select_code = (
        """
    // adapted from https://stackoverflow.com/a/44996422

    var chosen = source.selected.indices;
    if (typeof(chosen) == "number"){
        var chosen = [chosen]
    };

    var chosen_ids = [];
    for (var i = 0; i < chosen.length; ++ i){
        chosen_ids.push(source.data['"""
        + id_col
        + """'][chosen[i]])
    };

    var chosen_ids_indices = [];
    for (var i = 0; i < source.data['index'].length; ++ i){
        if (chosen_ids.includes(source.data['"""
        + id_col
        + """'][i])){
            chosen_ids_indices.push(i)
        };
    };
    source.selected.indices = chosen_ids_indices;
    source.change.emit();
    """
    )
    select_callback = CustomJS(args=select_kwargs, code=select_code)
    # point_glyph as only renderer assures that when a point is chosen
    # only that brick's id is chosen
    # this makes it impossible to choose ids based on clicking confidence bands
    tap = TapTool(renderers=[point_glyph], callback=select_callback)
    param_plot.tools.append(tap)
    boxselect = BoxSelectTool(renderers=[point_glyph], callback=select_callback)
    param_plot.tools.append(boxselect)
    return param_plot
