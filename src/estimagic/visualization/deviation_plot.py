import pandas as pd
import plotly.express as px

from estimagic.benchmarking.process_benchmark_results import (
    create_convergence_histories,
)
from estimagic.config import PLOTLY_TEMPLATE


def deviation_plot(
    problems,
    results,
    *,
    distance_measure="criterion",
    monotone=True,
    stopping_criterion="y",
    x_precision=1e-4,
    y_precision=1e-4,
    template=PLOTLY_TEMPLATE,
):
    """Plot convergence of optimizers for a set of problems.

    This creates a grid of plots, showing the convergence of the different
    algorithms on each problem. The faster a line falls, the faster the algorithm
    improved on the problem. The algorithm converged where its line reaches 0
    (if normalize_distance is True) or the horizontal blue line labeled "true solution".

    Each plot shows on the x axis the runtime_measure, which can be walltime or number
    of evaluations. Each algorithm's convergence is a line in the plot. Convergence can
    be measured by the criterion value of the particular time/evaluation. The
    convergence can be made monotone (i.e. always taking the bast value so far) or
    normalized such that the distance from the start to the true solution is one.

    Args:
        problems (dict): estimagic benchmarking problems dictionary. Keys are the
            problem names. Values contain information on the problem, including the
            solution value.
        results (dict): estimagic benchmarking results dictionary. Keys are
            tuples of the form (problem, algorithm), values are dictionaries of the
            collected information on the benchmark run, including 'criterion_history'
            and 'time_history'.

        distance_measure (str): One of "criterion", "parameter_distance".
        monotone (bool): If True the best found criterion value so far is plotted.
            If False the particular criterion evaluation of that time is used.
        normalize_distance (bool): If True the progress is scaled by the total distance
            between the start value and the optimal value, i.e. 1 means the algorithm
            is as far from the solution as the start value and 0 means the algorithm
            has reached the solution value.
        normalize_runtime (bool): If True the runtime each algorithm needed for each
            problem is scaled by the time the fastest algorithm needed. If True, the
            resulting plot is what Moré and Wild (2009) called data profiles.
        stopping_criterion (str): "x_and_y", "x_or_y", "x", "y" or None. If None, no
            clipping is done.
        x_precision (float or None): how close an algorithm must have gotten to the
            true parameter values (as percent of the Euclidean distance between start
            and solution parameters) before the criterion for clipping and convergence
            is fulfilled.
        y_precision (float or None): how close an algorithm must have gotten to the
            true criterion values (as percent of the distance between start
            and solution criterion value) before the criterion for clipping and
            convergence is fulfilled.

        template (str): The template for the figure. Default is "plotly_white".

    Returns:
        plotly.Figure

    """

    df, _ = create_convergence_histories(
        problems=problems,
        results=results,
        stopping_criterion=stopping_criterion,
        x_precision=x_precision,
        y_precision=y_precision,
    )

    outcome = f"{'monotone_' if monotone else ''}" + distance_measure + "_normalized"
    deviations = (
        df.set_index(["problem", "algorithm", "n_evaluations"])[outcome]
        .reindex(
            pd.MultiIndex.from_product(
                [
                    df["problem"].unique(),
                    df["algorithm"].unique(),
                    range(df["n_evaluations"].min(), df["n_evaluations"].max() + 1),
                ],
                names=["problem", "algorithm", "n_evaluations"],
            )
        )
        .fillna(method="ffill")
        .reset_index()
    )
    average_deviations = (
        deviations.groupby(["algorithm", "n_evaluations"]).mean()[outcome].reset_index()
    )
    fig = px.line(average_deviations, x="n_evaluations", y=outcome, color="algorithm")

    y_labels = {
        "criterion_normalized": "Share of Function Distance to Optimum<br>"
        "Missing From Current Criterion Value",
        "monotone_criterion_normalized": "Share of Function Distance to Optimum<br>"
        "Missing From Best So Far",
        "parameter_distance_normalized": "Share of Parameter Distance to Optimum<br>"
        "Missing From Current Parameters",
        "monotone_parameter_distance_normalized": "Share of the Parameter Distance "
        "to Optimum<br> Missing From the Best Parameters So Far",
    }
    fig.update_layout(
        xaxis_title="Number of Function Evaluations",
        yaxis_title=y_labels[outcome],
        title=None,
        height=300,
        width=500,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        template=template,
    )

    return fig
