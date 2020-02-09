"""Callback functions for the interactive distribution plot."""
import numpy as np
from bokeh.models import BoxSelectTool
from bokeh.models import CDSView
from bokeh.models import HoverTool
from bokeh.models import IndexFilter
from bokeh.models import TapTool
from bokeh.models.callbacks import CustomJS
from bokeh.models.filters import CustomJSFilter
from bokeh.models.layouts import Row
from bokeh.models.widgets import CheckboxGroup
from bokeh.models.widgets import Div
from bokeh.models.widgets import RangeSlider


# =====================================================================================
# Create widgets
# =====================================================================================


def value_slider(source, plots):
    val_min = source.data["xmin"].min()
    val_max = source.data["xmax"].max()
    if "ci_lower" in source.column_names:
        val_min = min(val_min, source.data["ci_lower"].min())
    if "ci_upper" in source.column_names:
        val_max = max(val_max, source.data["ci_upper"].max())
    x_range = val_max - val_min
    value_column_slider = RangeSlider(
        start=val_min - 0.02 * x_range,
        end=val_max + 0.02 * x_range,
        value=(val_min, val_max),
        step=x_range / 500,
        title="Value",
        name="value_slider",
    )

    code = """
        var lower_end = cb_obj.value[0]
        var upper_end = cb_obj.value[1]

        for (var i = 0; i < plots.length; ++ i){
            plots[i].x_range.start = lower_end;
            plots[i].x_range.end = upper_end;
        }
    """

    callback = CustomJS(args={"plots": plots}, code=code)
    value_column_slider.js_on_change("value", callback)
    return value_column_slider


def create_group_widget(source, subgroup_col):
    if subgroup_col is None:
        return Div(name="group_widget_placeholder")
    else:
        value_type = source.data[subgroup_col].dtype
        if value_type == float:
            slider = _subgroup_slider(source=source, subgroup_col=subgroup_col)
            return slider
        elif value_type == object:
            checkbox_group = _checkbox_group(source=source, subgroup_col=subgroup_col)
            return checkbox_group
        else:
            raise AttributeError(
                "dtype of the subgroup column must be object or float."
            )


def _subgroup_slider(source, subgroup_col):
    sorted_uniques = np.array(sorted(set(source.data[subgroup_col])))
    min_val, max_val = sorted_uniques[0], sorted_uniques[-1]
    min_dist_btw_vals = (sorted_uniques[1:] - sorted_uniques[:-1]).min()
    slider = RangeSlider(
        start=min_val - 0.05 * min_val,
        end=max_val + 0.05 * max_val,
        value=(min_val, max_val),
        step=min_dist_btw_vals,
        title=subgroup_col.title(),
        name="subgroup_widget",
    )
    slider.js_on_change(
        "value", CustomJS(code="source.change.emit();", args={"source": source})
    )
    return slider


def _checkbox_group(source, subgroup_col):
    checkbox_title = Div(text=str(subgroup_col).title() + ": ")
    checkbox_labels = sorted(set(source.data[subgroup_col]))
    actives = list(range(len(checkbox_labels)))
    checkboxes = CheckboxGroup(
        labels=checkbox_labels, active=actives, inline=True, name="subgroup_checkbox",
    )
    checkboxes.js_on_change(
        "active", CustomJS(code="source.change.emit();", args={"source": source})
    )
    return Row(checkbox_title, checkboxes, name="subgroup_widget")


# =====================================================================================
# create view from index and subgroup filters
# =====================================================================================


def create_view(source, group_index, subgroup_col, widget):
    filters = [IndexFilter(group_index)]
    if isinstance(widget, Row):  # this means we have checkbox groups
        checkboxes = widget.children[1]
        checkbox_filter = _checkbox_filter(checkboxes, source, subgroup_col)
        filters.append(checkbox_filter)
    elif isinstance(widget, RangeSlider):
        group_slider_filter = _slider_filter(widget, source, subgroup_col)
        filters.append(group_slider_filter)
    view = CDSView(source=source, filters=filters)
    return view


def _checkbox_filter(checkboxes, source, subgroup_col):
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


def _slider_filter(slider, source, column):
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


# =====================================================================================
# Hover and Select Tools
# =====================================================================================


def add_hover_tool(fig, point_glyph, source, group_cols):
    skip = (
        group_cols
        + ["dodge", "color", "binned_x", "level_0", "rect_width", "xmin", "xmax"]
        + [x for x in source.column_names if x.startswith("index")]
    )
    to_display = [
        col
        for col in source.column_names
        if len(set(source.data[col])) > 1 and col not in skip
    ]
    tooltips = [(col, "@" + col) for col in to_display]
    hover = HoverTool(renderers=[point_glyph], tooltips=tooltips)
    fig.tools.append(hover)
    return fig


def add_select_tools(fig, point_glyph, source):
    select_kwargs = {"source": source}
    select_code = """
    // adapted from https://stackoverflow.com/a/44996422

    var chosen = source.selected.indices;
    if (typeof(chosen) == "number"){
        var chosen = [chosen]
    };

    var chosen_ids = [];
    for (var i = 0; i < chosen.length; ++ i){
        chosen_ids.push(source.data['id'][chosen[i]])
    };

    var chosen_ids_indices = [];
    for (var i = 0; i < source.data['index'].length; ++ i){
        if (chosen_ids.includes(source.data['id'][i])){
            chosen_ids_indices.push(i)
        };
    };
    source.selected.indices = chosen_ids_indices;
    source.change.emit();
    """
    select_callback = CustomJS(args=select_kwargs, code=select_code)
    # point_glyph as only renderer assures that when a point is chosen
    # only that brick's id is chosen
    # this makes it impossible to choose ids based on clicking confidence bands
    tap = TapTool(renderers=[point_glyph], callback=select_callback)
    fig.tools.append(tap)
    boxselect = BoxSelectTool(renderers=[point_glyph], callback=select_callback)
    fig.tools.append(boxselect)
    return fig
