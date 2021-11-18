"""Create the benchmark plots.

Future To-Dos:
    - allow parameter distance as distance measure
    -

"""
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


def create_convergence_plots(
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
    """Plot convergence of algorithms for a set of problems.

    This creates a plot for each problem that shows the convergence of the different
    algorithms. The faster a line falls, the faster the algorithm improved on the
    problem. The algorithm converged where its line reaches 0 (if normalize is True) or
    the horizontal blue line labeled "true solution".

    Each plot shows on the x axis the runtime_measure, which can be walltime or number
    of evaluations. Each algorithm's convergence is a line in the plot. Convergence can
    be measured by the criterion
    value of the particular time/evaluation. The convergence can be made monotone (i.e.
    always taking the bast value so far) or normalized such that the distance from the
    start to the true solution is one.

    Args:
        problems (dict): estimagic benchmarking problems dictionary. Keys are the
            problem names. Values contain information on the problem, including the solution
            value.
        results (dict): estimagic benchmarking results dictionary. Keys are
            tuples of the form (problem, algorithm), values are dictionaries of the
            collected information on the benchmark run, including 'criterion_history' and
            'time_history'.
        n_cols (int): number of columns in the plot of grids. The number
            of rows is determined automatically. distance_measure (str): One of "criterion",
            "parameter_distance".
        monotone (bool): If True the best found criterion value is
            plotted. If False the particular criterion evaluation of that time is used.
        normalize (bool): If True the progress is scaled by the total distance between
            the start value and the optimal value, i.e. 1 means the algorithm is as far from
            the solution as the start value and 0 means the algorithm has reached the
            solution value.
        runtime_measure (str): "n_evaluations" or "walltime".
        stopping_criterion (str): "x_and_y", "x_or_y", "x", "y" or None. If None, no
            clipping is done.
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

    df = create_performance_df(problems=problems, results=results)

    if stopping_criterion is not None:
        solutions = get_solutions(problems)
        df, converged_info = clip_histories(
            df=df,
            stopping_criterion=stopping_criterion,
            x_precision=x_precision,
            y_precision=y_precision,
            solutions=solutions,
        )
    ### Use converged_info: maybe to augment legend entry for not converged algos?

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
            alpha=1.0,
            ax=ax,
            palette=palette,
        )

        ax.legend(title=None)
        if distance_measure == "criterion" and not normalize:
            f_opt = problems[prob_name]["solution"]["value"]
            ax.axhline(f_opt, label="true solution", lw=2.5)

    fig.tight_layout()
    return fig, axes


def create_data_profile_plot(
    problems,
    results,
    ### what we call runtime_measure here, i.e. runtime until desired convergence,
    ### is called performance measure by Moré and Wild (2009).
    runtime_measure="n_evaluations",
    stopping_criterion="y",
    x_precision=1e-4,
    y_precision=1e-4,
):
    """Plot data profiles as proposed by Moré and Wild (2009).

    Data profiles answer the question: What percentage of problems can each
    algorithm solve within a certain runtime budget?

    The runtime budget is plotted an the x axis and given in multiples of the best
    performing algorithm's required runtime to solve each problem.

    Looking at x=1.0 for example gives for each algorithm the share of problems that
    the algorithm solved in the shortest time. On the other hand looking at x=5.0 gives
    the share of problems each algorithm solved within 5 times the runtime of the
    fastest algorithm on each problem.

    Thus, algorithms that are very specialized and perform well on
    some share of problems but are not able to solve more problems with a larger
    computational budget will have flat lines. Algorithms that are robust but slow,
    will start at low shares but have a large slope.

    Note that failing to converge according to the given stopping_criterion and
    precisions is scored as needing in an infinite computational budget.

    Args:
        problems (dict): estimagic benchmarking problems dictionary.
        results (dict): estimagic benchmarking results dictionary.
        runtime_measure (str): "n_evaluations" or "walltime"
        x_precision (float): default 1e-4
        y_precision (float): default 1e-4
        stopping_criterion (str): one of "x_and_y", "x_or_y", "x", "y".

    Returns:
        fig, ax

    """
    df = create_performance_df(problems=problems, results=results)

    solutions = get_solutions(problems)
    if stopping_criterion is None:
        raise ValueError(
            "You must specify a stopping criterion for the performance plot. "
            "Without it is not possible to determine convergence."
        )

    df, converged_info = clip_histories(
        df=df,
        stopping_criterion=stopping_criterion,
        x_precision=x_precision,
        y_precision=y_precision,
        solutions=solutions,
    )

    solution_times = create_solution_times(
        df, runtime_measure=runtime_measure, converged_info=converged_info
    )
    performance_ratios = solution_times.divide(solution_times.min(axis=1), axis=0)
    # set again to inf because no inf Timedeltas were allowed but we want inf here.
    performance_ratios[~converged_info] = np.inf

    alphas = determine_alpha_grid(performance_ratios)

    for_each_alpha = pd.concat(
        {alpha: performance_ratios < alpha for alpha in alphas},
        names=["alpha"],
    )
    performance_profiles = for_each_alpha.groupby("alpha").mean().stack().reset_index()

    fig, ax = plt.subplots(figsize=(8, 6))
    n_algos = len(performance_ratios.columns)
    sns.lineplot(
        data=performance_profiles,
        x="alpha",
        y=0,
        hue="algorithm",
        ax=ax,
        lw=2.5,
        alpha=1.0,
        palette=get_colors("categorical", n_algos),
    )
    ax.set_xlabel("Multiple of Fastest Algorithm's Runtime")
    ax.set_ylabel("Share of Problems Solved")
    return fig, ax


def create_solution_times(df, runtime_measure, converged_info):
    """Find the solution time for each algorithm and problem in walltime and criterion evaluations.

    Args:
        df (pandas.DataFrame): index levels are ['problem', 'algorithm', 'evaluation'].
        runtime_measure (str): 'walltime' or 'n_evaluations'.
        converged_info (pandas.DataFrame): columns are the algorithms, index are the
            problems. The values are boolean and True when the algorithm arrived at
            the solution with the desired precision.

    Returns:
        solution_times (pandas.DataFrame): columns are algorithms, index are problems.
            The values is either the number of evaluations or the walltime each
            algorithm needed to arrive at the required precision that was specified
            in the creation of the performance_df. ###

    """
    if runtime_measure == "walltime":
        solution_times = df.groupby(["problem", "algorithm"])["walltime"].max()
        solution_times = solution_times.unstack()
        # inf not allowed for timedeltas
        solution_times[~converged_info] = pd.Timedelta(weeks=1000)
    elif runtime_measure == "n_evaluations":
        solution_times = df.groupby(["problem", "algorithm"]).size()
        solution_times.name = "n_evaluations"
        solution_times = solution_times.unstack()
        solution_times[~converged_info] = np.inf
    else:
        raise ValueError(
            "Only 'walltime' or 'n_evaluations' are allowed as "
            f"runtime_measure. You specified {runtime_measure}"
        )

    return solution_times


def determine_alpha_grid(performance_ratios):
    """Determine the alphas at which to calculate the performance profile.

    Args:
        performance_ratios (pandas.DataFrame): columns are the names of the algorithms,
            the index are the problem. Values are performance as multiple of the best
            algorithm. For example, if the criterion is runtime in walltime and there
            are two algorithms, one which needed 20 seconds and one that needed 30 for
            a problem. Then the first algorithm has a performance ratio of 1.0 and the
            other of 1.5.

    Returns:
        list: sorted switching points plus one point slightly to the right
    """
    switch_points = np.unique(performance_ratios.values)
    finite_switch_points = switch_points[np.isfinite(switch_points)]
    point_to_right = finite_switch_points[-1] * 1.05
    alphas = np.append(finite_switch_points, point_to_right)
    alphas = sorted(alphas)
    return alphas


def create_performance_df(problems, results):
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
        pandas. Series: index is the same as that of df. Values are the same as in
            df[col] but monotone decreasing.

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


def clip_histories(df, stopping_criterion, x_precision, y_precision, solutions):
    """Shorten the DataFrame to just the evaluations until each algorithm converged.

    Args:
        df (pandas.DataFrame): index levels are ['problem', 'algorithm', 'evaluation'].
            Columns must include "monotone_criterion".
        stopping_criterion (str): one of "x_and_y", "x_or_y", "x", "y".
        x_precision (float): when an algorithm's parameters are closer than this to the
            true solution's parameters, the algorithm is counted as having converged.
        y_precision (float): when an algorithm's criterion value is closer than this to
            the solution value, the algorithm is counted as having converged.
        solutions (pandas.DataFrame): index are the problem names. There's one column for the
            optimal criterion and one for the optimal parameters

    Returns:
        shortened (pandas.DataFrame): the entered DataFrame with all histories
            shortened to stop once conversion according to the given criteria is
            reached.
        converged_info (pandas.DataFrame): columns are the algorithms, index are the
            problems. The values are boolean and True when the algorithm arrived at
            the solution with the desired precision.

    """
    if stopping_criterion in ["y"]:
        criterion_by_problem = df.unstack("problem")["monotone_criterion"]
        f_opt = solutions.loc[criterion_by_problem.columns, "criterion"]
        converged = criterion_by_problem < f_opt + y_precision
        shortened = criterion_by_problem[~converged]
        shortened_stacked = shortened.stack().reorder_levels(
            ["problem", "algorithm", "n_evaluations"]
        )
        shortened_indices = shortened_stacked.dropna().index
        shortened = df.loc[shortened_indices]

        converged_info = converged.unstack("algorithm").any().unstack("algorithm")
    else:
        raise NotImplementedError("Only 'y' is supported as stopping_criterion so far")
    return shortened, converged_info


def get_solutions(problems):
    """Extract problem solutions from problems.

    Args:
        problems (dict): estimagic benchmarking problems dictionary. Keys are the
            problem names. Values contain information on the problem, including the solution
            value.

    Returns:
        pandas.DataFrame: index are the problem names. There's one column for the
            optimal criterion and one for the optimal parameters

    """
    f_opt = pd.Series(
        {name: prob["solution"]["value"] for name, prob in problems.items()},
        name="criterion",
    )
    opt_params = pd.Series(
        {name: prob["solution"]["params"] for name, prob in problems.items()},
        name="params",
    )

    solutions = pd.concat([f_opt, opt_params], axis=1)
    return solutions
