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
import pandas as pd
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource
from bokeh.models import HoverTool
from bokeh.models import TapTool
from bokeh.models.callbacks import CustomJS
from bokeh.models.widgets import CheckboxGroup
from bokeh.plotting import figure
from bokeh.plotting import show

from estimagic.dashboard.plotting_functions import get_color_palette
from estimagic.optimization.utilities import index_element_to_string


def comparison_plot(data_dict, color_dict=None, height=None, width=None):
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
            This is generated from the Category20 palette if not given.

        height (int):
            height of the (entire) plot.

        width (int):
            width of the (entire) plot.

    """
    df, color_dict, param_groups, heights, width = _process_inputs(
        data_dict=data_dict, color_dict=color_dict, height=height, width=width
    )

    source = ColumnDataSource(df)

    plots = []
    for param_group_name, group_height in zip(param_groups, heights):
        to_plot = df[df["group"] == param_group_name]["full_name"].unique()

        # create the "canvas"
        param_group_plot = figure(
            title="Comparison Plot of {} Parameters".format(param_group_name.title()),
            y_range=to_plot,
            plot_height=group_height,
            plot_width=width,
        )

        # add circles representing the parameter value
        point_estimate_glyph = param_group_plot.circle(
            source=source,
            x="value",
            y="full_name",
            size=12,
            color="color",
            selection_color="color",
            nonselection_color="color",
            alpha=0.5,
            selection_alpha=0.8,
            nonselection_alpha=0.2,
        )

        # add the confidence_intervals as hbars
        # horizontal whiskers not supported in bokeh 1.0.4
        if "conf_int_lower" in df.columns and "conf_int_upper" in df.columns:
            param_group_plot.hbar(
                source=source,
                y="full_name",
                left="conf_int_lower",
                right="conf_int_upper",
                height=0.3,
                alpha=0.0,
                selection_alpha=0.25,
                nonselection_fill_alpha=0.0,
                line_alpha=0.1,
                color="color",
                selection_color="color",
                nonselection_color="color",
            )

        _add_hover_tool(param_group_plot, point_estimate_glyph)

        _add_tap_tool(source, param_group_plot, point_estimate_glyph)

        plots.append(param_group_plot)

    if "model_class" in df.columns:
        cb_group = _create_checkbox(
            widget_labels=sorted(df["model_class"].unique()), source=source
        )
        plots = [cb_group] + plots

    grid = gridplot(plots, toolbar_location="right", ncols=1)
    show(grid)
    return grid, plots


def _process_inputs(data_dict, color_dict, height, width):
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

    """
    color_dict = _build_or_check_color_dict(color_dict, data_dict)
    df = _build_df_from_data_dict(data_dict, color_dict)
    _check_groups_and_names_compatible(df)

    param_groups = [x for x in df["group"].unique() if x is not None]
    nr_params = len(df[df["group"].isin(param_groups)]["name"].unique())
    if height is None:
        heights = [300 for group_name in param_groups]
    else:
        heights = []
        for group_name in param_groups:
            nr_group_params = len(df[df["group"] == group_name]["name"].unique())
            heights.append(int(nr_group_params / nr_params * height))

    if width is None:
        width = 600
    return df, color_dict, param_groups, heights, width


def _build_or_check_color_dict(color_dict, data_dict):
    model_classes = []
    for d in data_dict.values():
        if "model_class" in d.keys():
            model_classes.append(d["model_class"])

    if color_dict is None:
        colors = get_color_palette(len(model_classes))
        color_dict = {m: c for m, c in zip(model_classes, colors)}
    else:
        assert set(model_classes).issubset(color_dict.keys()), (
            "Your color_dict does not map every model class "
            + "in your data_dict to a color."
        )
    return color_dict


def _build_df_from_data_dict(data_dict, color_dict):
    df = pd.DataFrame(columns=["value", "model", "name"])

    for model, mod_dict in data_dict.items():
        ext_param_df = mod_dict["result_df"]
        name_cols = [x for x in ["group", "name"] if x in ext_param_df.columns]
        ext_param_df["full_name"] = ext_param_df[name_cols].apply(
            lambda x: index_element_to_string(tuple(x)), axis=1
        )
        ext_param_df["model"] = model
        if "model_class" in mod_dict.keys():
            ext_param_df["model_class"] = mod_dict["model_class"]
            ext_param_df["color"] = color_dict[mod_dict["model_class"]]
        else:
            # the standard color is mediumelectricblue
            ext_param_df["color"] = "#035096"

        df = df.append(ext_param_df, sort=False)

    # reset index as safety measure to make sure the index
    # gives the position in the arrays
    # that the source data dictionary points to

    # keep as much information from the index as possible
    if type(df.index) is pd.MultiIndex:
        for name in df.index.names:
            drop = name not in df.columns
            df.reset_index(name, drop=drop, inplace=True)
    else:
        drop = df.index.name not in df.columns
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


def _add_hover_tool(param_group_plot, point_estimate_glyph):
    tooltips = [("parameter value", "@value"), ("model", "@model")]
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
        labels=widget_labels, active=[0, 1], callback=widget_callback
    )
    return cb_group
