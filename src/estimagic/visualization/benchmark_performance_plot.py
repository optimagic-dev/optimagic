import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from estimagic.examples.process_benchmark_results import create_performance_df
from estimagic.visualization.colors import get_colors

plt.rcParams.update(
    {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)


def create_performance_plot_over_full_problem_set(
    problems,
    results,
    runtime_measure="n_evaluations",
    stopping_criterion="y",
    x_precision=1e-4,
    y_precision=1e-4,
):
    """Compare optimizers over full problem set as proposed by Moré and Wild (2009).

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
        problems (dict): estimagic benchmarking problems dictionary. Keys are the
            problem names. Values contain information on the problem, including the
            solution value.
        results (dict): estimagic benchmarking results dictionary. Keys are
            tuples of the form (problem, algorithm), values are dictionaries of the
            collected information on the benchmark run, including 'criterion_history'
            and 'time_history'.
        runtime_measure (str): "n_evaluations" or "walltime".
            This is the runtime until the desired convergence was reached by an
            algorithm. This is called performance measure by Moré and Wild (2009).
        stopping_criterion (str): one of "x_and_y", "x_or_y", "x", "y". Determines
            how convergence is determined from the two precisions.
        x_precision (float or None): how close an algorithm must have gotten to the
            true parameter values (as percent of the Euclidean distance between start
            and solution parameters) before the criterion for clipping and convergence
            is fulfilled.
        y_precision (float or None): how close an algorithm must have gotten to the
            true criterion values (as percent of the distance between start
            and solution criterion value) before the criterion for clipping and
            convergence is fulfilled.

    Returns:
        fig, ax

    """
    if stopping_criterion is None:
        raise ValueError(
            "You must specify a stopping criterion for the performance plot. "
        )

    df, converged_info = create_performance_df(
        problems=problems,
        results=results,
        stopping_criterion=stopping_criterion,
        x_precision=x_precision,
        y_precision=y_precision,
    )

    solution_times = _create_solution_times(
        df, runtime_measure=runtime_measure, converged_info=converged_info
    )
    performance_ratios = solution_times.divide(solution_times.min(axis=1), axis=0)
    # set again to inf because no inf Timedeltas were allowed.
    performance_ratios[~converged_info] = np.inf

    alphas = _determine_alpha_grid(performance_ratios)

    for_each_alpha = pd.concat(
        {alpha: performance_ratios <= alpha for alpha in alphas},
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
    if runtime_measure == "n_evaluations":
        ax.set_xlabel(
            "Multiple of Minimal Number of Function Evaluations\n"
            "Needed to Solve the Problem"
        )
    elif runtime_measure == "walltime":
        ax.set_xlabel(
            "Multiple of Minimal Number of Wall Time\nNeeded to Solve the Problem"
        )

    ax.set_ylabel("Share of Problems Solved")
    return fig, ax


def _create_solution_times(df, runtime_measure, converged_info):
    """Find the solution time for each algorithm and problem.

    Args:
        df (pandas.DataFrame): contains 'problem', 'algorithm', 'n_evaluation' in the
            columns.
        runtime_measure (str): 'walltime' or 'n_evaluations'.
        converged_info (pandas.DataFrame): columns are the algorithms, index are the
            problems. The values are boolean and True when the algorithm arrived at
            the solution with the desired precision.

    Returns:
        solution_times (pandas.DataFrame): columns are algorithms, index are problems.
            The values are either the number of evaluations or the walltime each
            algorithm needed to achieve the desired precision. If the desired precision
            was not achieved the value is set to np.inf (for n_evaluations) or 7000 days
            (for walltime since there no infinite value is allowed).

    """
    solution_times = df.groupby(["problem", "algorithm"])[runtime_measure].max()
    solution_times = solution_times.unstack()

    # inf not allowed for timedeltas so put something very large
    if runtime_measure == "walltime":
        inf_value = pd.Timedelta(weeks=1000)
    elif runtime_measure == "n_evaluations":
        inf_value = np.inf
    else:
        raise ValueError(
            "Only 'walltime' or 'n_evaluations' are allowed as "
            f"runtime_measure. You specified {runtime_measure}."
        )

    solution_times[~converged_info] = inf_value
    return solution_times


def _determine_alpha_grid(performance_ratios):
    """Determine the alphas at which to calculate the performance profile.

    Args:
        performance_ratios (pandas.DataFrame): columns are the names of the algorithms,
            the index are the problems. Values are performance as multiple of the best
            algorithm. For example, if the criterion is runtime in walltime and there
            are two algorithms, one which needed 20 seconds and one that needed 30 for
            a problem. Then the first algorithm has a performance ratio of 1.0 and the
            other of 1.5.

    Returns:
        list: sorted switching points plus one point slightly to the right
    """
    switch_points = np.unique(performance_ratios.values) + 1e-10
    finite_switch_points = switch_points[np.isfinite(switch_points)]
    point_to_right = finite_switch_points[-1] * 1.05
    alphas = np.append(finite_switch_points, point_to_right)
    alphas = sorted(alphas)
    return alphas
