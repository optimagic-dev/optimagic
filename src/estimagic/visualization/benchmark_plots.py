import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from estimagic.visualization.colors import get_colors

plt.rcParams.update(
    {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)


def convergence_plot(
    problems,
    results,
    n_cols=2,
    distance_measure="criterion",
    monotone=True,
    normalize=True,
    runtime_measure="n_evaluations",
    stopping_criterion=None,
    x_precision=1e-4,
    y_precision=1e-4,
):
    """Plot convergence of a numerical optimizer.

    Args:
        problems (dict): estimagic benchmarking problems dictionary.
            Keys are the problem names. Values contain information on the problem,
            including the solution value.
        results (dict): estimagic benchmarking results dictionary. Keys are tuples of
            the form (problem, algorithm), values are dictionaries of the collected
            information on the benchmark run, including 'criterion_history' and
            'time_history'.
        n_cols (int): number of columns in the plot of grids. The number of rows is
            determined automatically.
        distance_measure (str): One of "criterion", "parameter_distance".
        monotone (bool): If True the best found criterion value is plotted.
            If False the particular criterion evaluation of that time is found.
        normalize (bool): If True the progress is scaled by the total distance between
            the start value and the optimal value, i.e. 1 means the algorithm is as far
            from the solution as the start value and 0 means the algorithm has reached
            the solution value.
        runtime_measure (str): "n_evaluations" or "walltime".
        stopping_criterion (str): "x_and_y", "x_or_y", "x", "y" or None.
            If None, no clipping is done.
        x_precision (float or None): Default is 1e-4.
        y_precision (float or None): Default is 1e-4.

    Returns:
        fig, axes

    """
    n_rows = int(np.ceil(len(problems) / n_cols))
    figsize = (n_cols * 6, n_rows * 4)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)
    algorithms = set(tup[1] for tup in results.keys())
    palette = get_colors("categorical", number=len(algorithms))

    df = create_performance_df(
        problems=problems,
        results=results,
        stopping_criterion=stopping_criterion,
        x_precision=x_precision,
        y_precision=y_precision,
    )

    outcome = (
        f"{'monotone_' if monotone else ''}"
        + distance_measure
        + f"{'_normalized' if normalize else ''}"
    )

    y_labels = {
        "criterion": "Current Function Value",
        "monotone_criterion": "Best Function Value Found So Far",
        "criterion_normalized": "Share of Function Distance to Optimum\nMissingFrom Current Criterion Value",
        "monotone_criterion_normalized": "Share of Function Distance to Optimum\nMissingFrom Best So Far",
    }
    x_labels = {
        "n_evaluations": "Number of Function Evaluations",
        "walltime": "Elapsed Time",
    }

    for ax, prob_name in zip(axes.flatten(), problems.keys()):
        ax.set_ylabel(y_labels[outcome])
        ax.set_xlabel(x_labels[runtime_measure])
        ax.set_title(prob_name.replace("_", " ").title())

        to_plot = df.loc[prob_name].reset_index()
        sns.lineplot(
            data=to_plot,
            x=runtime_measure,
            y=outcome,
            hue="algorithm",
            lw=2.5,
            alpha=0.7,
            ax=ax,
            palette=palette,
        )

        ax.legend(title=None)
        if distance_measure == "criterion" and not normalize:
            f_opt = problems[prob_name]["solution"]["value"]
            ax.axhline(f_opt, label="true solution", lw=2.5)

    fig.tight_layout()
    return fig, axes


def create_performance_df(
    problems, results, stopping_criterion, x_precision, y_precision
):
    """Create DataFrame with all information needed for the benchmarking plots.

    Args:
        problems (dict): estimagic benchmarking problems dictionary.
        results (dict): estimagic benchmarking results dictionary.

    Returns:
        pandas.DataFrame: index levels are ['problem', 'algorithm', 'evaluation'].
    """
    # build df from results
    time_sr = get_history_as_stacked_sr_from_results(results, "time_history")
    time_sr.name = "walltime"
    criterion_sr = get_history_as_stacked_sr_from_results(results, "criterion_history")
    df = pd.concat([time_sr, criterion_sr], axis=1)

    # normalizations
    f_0 = df.query("evaluation == 1").groupby("problem")["criterion"].mean()
    f_opt = pd.Series(
        {name: prob["solution"]["value"] for name, prob in problems.items()}
    )
    df["criterion_normalized"] = calculate_share_of_improvement_missing(
        sr=df["criterion"], start_values=f_0, target_values=f_opt
    )

    # monotone versions
    df["monotone_criterion"] = get_lowest_so_far(df, "criterion")
    df["monotone_criterion_normalized"] = get_lowest_so_far(df, "criterion_normalized")

    ### distance measures on the parameter dimension are missing!

    ### CLIPPING IS MISSING !!!
    if stopping_criterion is not None:
        raise NotImplementedError("Clipping still missing at the moment.")

    df.index = df.index.rename({"evaluation": "n_evaluations"})
    return df


def get_history_as_stacked_sr_from_results(results, key):
    """Get history as stacked Series from results.

    Args:
        results (dict): estimagic benchmarking results dictionary.
        key (str): name of the history for which to build the Series, e.g.
            criterion_history.

    Returns:
        pandas.Series: index levels are 'problem', 'algorithm' and 'evaluation'.
            the name is the key with '_history' stripped off.

    """
    histories = {tup: res[key] for tup, res in results.items()}
    sr = pd.concat(histories)
    sr.index.names = ["problem", "algorithm", "evaluation"]
    sr.name = key.replace("_history", "")
    return sr


def get_lowest_so_far(df, col):
    """Create Series with lowest value so far.

    Args:
        df (pandas.DataFrame): index levels are ['problem', 'algorithm', 'evaluation'].
            The columns include **col**.
        col (str): name of the column for which to calculate the lowest value so far.

    Returns:
        pandas. Series: index is the same as that of df. Values are the same as in df[col] but monotone increasing

    """
    values = df.unstack(["problem", "algorithm"])[col].sort_index()
    only_lowest = values[values.diff() < 0].reindex(values.index)
    # put first row back in. This was NaN from the first differencing
    only_lowest.loc[0] = values.loc[0]
    nan_filled = only_lowest.fillna(method="ffill")
    stacked = nan_filled.stack(["problem", "algorithm"])
    with_correct_index_order = stacked.reorder_levels(
        ["problem", "algorithm", "evaluation"]
    )
    shortened = with_correct_index_order.loc[df.index]
    return shortened


def calculate_share_of_improvement_missing(sr, start_values, target_values):
    """Calculate the share of improvement still missing relative to the start point.

    Args:
        sr (pandas.Series): index levels are ["problem", "algorithm", "evaluation"].
            Values are the current values, e.g. criterion values.
        start_values (pandas.Series): index are the problems, values are the start values
        target_values (pandas.Series): index are the problems, values are the target values.

    Returns:
        pandas.Series: index is the same as that of sr. The lower the value the closer the
            current value is to the target value. 0 means the target value has been reached.
            1 means the current value is as far from the target value as the start value.

    """
    total_improvements = start_values - target_values
    current_value = sr.unstack("problem")
    missing_improvement = current_value - target_values
    share_missing = missing_improvement / total_improvements
    share_missing.columns.name = "problem"  # this got missing in between
    stacked = share_missing.stack("problem")
    correct_index_levels = stacked.reorder_levels(
        ["problem", "algorithm", "evaluation"]
    ).loc[sr.index]
    return correct_index_levels
