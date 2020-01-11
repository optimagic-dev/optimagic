"""
Plot the parameters from a list of optimization result params DataFrames.
The plot shows you how robust the estimated parameter values are across your models.

In particular, the plot can answer the following questions:

1. How are the parameters distributed?

2. How large are the differences in parameter estimates between results
compared to the uncertainty around the parameter estimates?

3. Are parameters of groups of results clustered?

The plot is basically a clickable histogram where each individual observation
(in this case the parameter estimate of a particular model) is represented as a brick
in the stackt that correspond to one bar of the histogram.
By hovering or clicking on a particular brick you can learn more about that observation
making it easy to identify and analyze patterns.

"""
import pandas as pd

from estimagic.visualization.interactive_distribution_plot import (
    interactive_distribution_plot,
)


def parameter_distribution_plot(
    results,
    height=None,
    width=500,
    axis_for_every_parameter=False,
    x_padding=0.1,
    num_bins=50,
):
    """Make a comparison plot from a dictionary containing optimization results.

    Args:
        results (list): List of estimagic optimization results where the info
            can have been extended with 'model_class' and 'model_name'
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
        source (bokeh.models.ColumnDataSource): data underlying the plots
        gridplot (bokeh.layouts.Column): grid of the distribution plots.
    """
    df = _tidy_df_from_results(results)

    source, grid = interactive_distribution_plot(
        df=df,
        value_col="value",
        id_col="model_name",
        group_cols=["group", "name"],
        subgroup_col="model_class" if "model_class" in df.columns else None,
        height=height,
        width=width,
        axis_for_every_parameter=axis_for_every_parameter,
        x_padding=x_padding,
        num_bins=num_bins,
    )
    return source, grid


def _tidy_df_from_results(results):
    results = _add_model_names_if_missing(results)
    df = pd.concat(results, sort=True)
    keep = [x for x in df.columns if not x.startswith("_")]
    df = df[keep]
    if "model_class" in df.columns:
        df["model_class"].fillna("None", inplace=True)
    return df


def _add_model_names_if_missing(results):
    if not any("model_name" in df.columns for df in results):
        results = [df.copy() for df in results]
        for i, df in enumerate(results):
            df["model_name"] = str(i)
    elif all("model_name" in df.columns for df in results):
        assert all(len(df["model_name"].unique()) == 1 for df in results), (
            """The model name must be the same for all parameters """
            + """in one results DataFrame."""
        )
        unique_model_names = {df["model_name"].unique()[0] for df in results}
        assert len(unique_model_names) == len(
            results
        ), """The model names are not unique."""
    return results
