import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from estimagic.benchmarking.process_benchmark_results import (
    create_convergence_histories,
)
from estimagic.utilities import propose_alternatives
from estimagic.visualization.colors import get_colors

plt.rcParams.update(
    {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)


def convergence_plot(
    problems=None,
    results=None,
    problem_subset=None,
    algorithm_subset=None,
    n_cols=2,
    distance_measure="criterion",
    monotone=True,
    normalize_distance=True,
    runtime_measure="n_evaluations",
    stopping_criterion="y",
    x_precision=1e-4,
    y_precision=1e-4,
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
        problem_subset (list, optional): List of problem names. These must be a subset
            of the keys of the problems dictionary. If provided the convergence plot is
            only created for the problems specified in this list.
        algorithm_subset (list, optional): List of algorithm names. These must be a
            subset of the keys of the optimizer_options passed to run_benchmark. If
            provided only the convergence of the given algorithms are shown.
        n_cols (int): number of columns in the plot of grids. The number
            of rows is determined automatically.
        distance_measure (str): One of "criterion", "parameter_distance".
        monotone (bool): If True the best found criterion value so far is plotted.
            If False the particular criterion evaluation of that time is used.
        normalize_distance (bool): If True the progress is scaled by the total distance
            between the start value and the optimal value, i.e. 1 means the algorithm
            is as far from the solution as the start value and 0 means the algorithm
            has reached the solution value.
        runtime_measure (str): "n_evaluations" or "walltime".
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

    Returns:
        fig

    """
    df, _ = create_convergence_histories(
        problems=problems,
        results=results,
        stopping_criterion=stopping_criterion,
        x_precision=x_precision,
        y_precision=y_precision,
    )

    # handle string provision for single problems / algorithms
    if isinstance(problem_subset, str):
        problem_subset = [problem_subset]
    if isinstance(algorithm_subset, str):
        algorithm_subset = [algorithm_subset]

    _check_only_allowed_subset_provided(problem_subset, df["problem"], "problem")
    _check_only_allowed_subset_provided(algorithm_subset, df["algorithm"], "algorithm")

    if problem_subset is not None:
        df = df[df["problem"].isin(problem_subset)]
    if algorithm_subset is not None:
        df = df[df["algorithm"].isin(algorithm_subset)]

    # plot configuration
    outcome = (
        f"{'monotone_' if monotone else ''}"
        + distance_measure
        + f"{'_normalized' if normalize_distance else ''}"
    )

    # create plots
    remaining_problems = df["problem"].unique()
    n_rows = int(np.ceil(len(remaining_problems) / n_cols))
    figsize = (n_cols * 6, n_rows * 4)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)

    if algorithm_subset is None:
        algorithms = {tup[1] for tup in results.keys()}
    else:
        algorithms = algorithm_subset
    palette = get_colors("categorical", number=len(algorithms))

    for ax, prob_name in zip(axes.flatten(), remaining_problems):
        to_plot = df[df["problem"] == prob_name]
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
        ax.set_title(prob_name.replace("_", " ").title())
        if distance_measure == "criterion" and not normalize_distance:
            f_opt = problems[prob_name]["solution"]["value"]
            ax.axhline(f_opt, label="true solution", lw=2.5)

    # style plots
    y_labels = {
        "criterion": "Current Function Value",
        "monotone_criterion": "Best Function Value Found So Far",
        "criterion_normalized": "Share of Function Distance to Optimum\n"
        + "Missing From Current Criterion Value",
        "monotone_criterion_normalized": "Share of Function Distance to Optimum\n"
        + "Missing From Best So Far",
        "parameter_distance": "Distance Between Current and Optimal Parameters",
        "parameter_distance_normalized": "Share of the Parameter Distance to Optimum\n"
        + "Missing From Current Parameters",
        "monotone_parameter_distance_normalized": "Share of the Parameter Distance "
        + "to Optimum\n Missing From the Best Parameters So Far",
        "monotone_parameter_distance": "Distance Between the Best Parameters So Far\n"
        "and the Optimal Parameters",
    }
    x_labels = {
        "n_evaluations": "Number of Function Evaluations",
        "walltime": "Elapsed Time",
    }
    for ax in axes.flatten():
        ax.set_ylabel(y_labels[outcome])
        ax.set_xlabel(x_labels[runtime_measure])
        ax.legend(title=None)

    # make empty plots invisible
    n_empty_plots = len(axes.flatten()) - len(remaining_problems)
    if n_empty_plots > 0:
        for ax in axes.flatten()[-n_empty_plots:]:
            ax.set_visible(False)
    fig.tight_layout()
    return fig


def _check_only_allowed_subset_provided(subset, allowed, name):
    """Check if all entries of a proposed subset are in a Series.

    Args:
        subset (iterable or None): If None, no checks are performed. Else a ValueError
            is raised listing all entries that are not in the provided Series.
        allowed (iterable): allowed entries.
        name (str): name of the provided entries to use for the ValueError.

    Raises:
        ValueError

    """
    allowed = set(allowed)
    if subset is not None:
        missing = [entry for entry in subset if entry not in allowed]
        if missing:
            missing_msg = ""
            for entry in missing:
                proposed = propose_alternatives(entry, allowed)
                missing_msg += f"Invalid {name}: {entry}. Did you mean {proposed}?\n"
            raise ValueError(missing_msg)
