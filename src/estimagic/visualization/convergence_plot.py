import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from estimagic.benchmarking.process_benchmark_results import create_performance_df
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
        fig, axes

    """
    n_rows = int(np.ceil(len(problems) / n_cols))
    figsize = (n_cols * 6, n_rows * 4)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)
    algorithms = {tup[1] for tup in results.keys()}
    palette = get_colors("categorical", number=len(algorithms))

    df, _ = create_performance_df(
        problems=problems,
        results=results,
        stopping_criterion=stopping_criterion,
        x_precision=x_precision,
        y_precision=y_precision,
    )

    outcome = (
        f"{'monotone_' if monotone else ''}"
        + distance_measure
        + f"{'_normalized' if normalize_distance else ''}"
    )

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

    for ax, prob_name in zip(axes.flatten(), problems.keys()):
        ax.set_ylabel(y_labels[outcome])
        ax.set_xlabel(x_labels[runtime_measure])
        ax.set_title(prob_name.replace("_", " ").title())

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

        ax.legend(title=None)
        if distance_measure == "criterion" and not normalize_distance:
            f_opt = problems[prob_name]["solution"]["value"]
            ax.axhline(f_opt, label="true solution", lw=2.5)

    fig.tight_layout()
    return fig, axes
