import numpy as np
import pandas as pd


def create_convergence_histories(
    problems, results, stopping_criterion, x_precision, y_precision
):
    """Create tidy DataFrame with all information needed for the benchmarking plots.

    Args:
        problems (dict): estimagic benchmarking problems dictionary. Keys are the
            problem names. Values contain information on the problem, including the
            solution value.
        results (dict): estimagic benchmarking results dictionary. Keys are
            tuples of the form (problem, algorithm), values are dictionaries of the
            collected information on the benchmark run, including 'criterion_history'
            and 'time_history'.
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
        pandas.DataFrame: tidy DataFrame with the following columns:
            - problem
            - algorithm
            - n_evaluations
            - walltime
            - criterion
            - criterion_normalized
            - monotone_criterion
            - monotone_criterion_normalized
            - parameter_distance
            - parameter_distance_normalized
            - monotone_parameter_distance
            - monotone_parameter_distance_normalized

    """
    # get solution values for each problem
    f_opt = pd.Series(
        {name: prob["solution"]["value"] for name, prob in problems.items()}
    )
    x_opt = {
        name: prob["solution"]["params"]["value"] for name, prob in problems.items()
    }

    # build df from results
    time_sr = _get_history_as_stacked_sr_from_results(results, "time_history")
    time_sr.name = "walltime"
    criterion_sr = _get_history_as_stacked_sr_from_results(results, "criterion_history")
    x_dist_sr = _get_history_of_the_parameter_distance(results, x_opt)
    df = pd.concat([time_sr, criterion_sr, x_dist_sr], axis=1)

    df.index = df.index.rename({"evaluation": "n_evaluations"})
    df = df.sort_index().reset_index()

    first_evaluations = df.query("n_evaluations == 0").groupby("problem")
    f_0 = first_evaluations["criterion"].mean()
    x_0_dist = first_evaluations["parameter_distance"].mean()
    x_opt_dist = {name: 0 for name in problems}

    # normalizations
    df["criterion_normalized"] = _normalize(
        df=df, col="criterion", start_values=f_0, target_values=f_opt
    )
    df["parameter_distance_normalized"] = _normalize(
        df=df,
        col="parameter_distance",
        start_values=x_0_dist,
        target_values=x_opt_dist,
    )
    # create monotone versions of columns
    df["monotone_criterion"] = _make_history_monotone(df, "criterion")
    df["monotone_parameter_distance"] = _make_history_monotone(df, "parameter_distance")
    df["monotone_criterion_normalized"] = _make_history_monotone(
        df, "criterion_normalized"
    )
    df["monotone_parameter_distance_normalized"] = _make_history_monotone(
        df, "parameter_distance_normalized"
    )

    if stopping_criterion is not None:
        df, converged_info = _clip_histories(
            df=df,
            stopping_criterion=stopping_criterion,
            x_precision=x_precision,
            y_precision=y_precision,
        )
    else:
        converged_info = None

    return df, converged_info


def _get_history_as_stacked_sr_from_results(results, key):
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


def _get_history_of_the_parameter_distance(results, x_opt):
    """Calculate the history of the distances to the optimal parameters.

    Args:
        results (dict): estimagic benchmarking results dictionary. Keys are
            tuples of the form (problem, algorithm), values are dictionaries of the
            collected information on the benchmark run, including 'params_history'.
        x_opt (dict): the keys are the problems, the values are pandas.Series with the
            optimal parameters for the respective problem.

    Returns:
        pandas.Series: index levels are "problem", "algorithm", "evaluation". The name
            is "parameter_distance".

    """
    x_dist_history = {}
    for (prob, algo), res in results.items():
        param_history = res["params_history"]
        x_dist_history[(prob, algo)] = pd.Series(
            np.linalg.norm(param_history - x_opt[prob], axis=1)
        )

    sr = pd.concat(x_dist_history)
    sr.index.names = ["problem", "algorithm", "evaluation"]
    sr.name = "parameter_distance"
    return sr


def _make_history_monotone(df, target_col, direction="minimize"):
    """Create a monotone Series, i.e. the best so far instead of the current evaluation.

    Args:
        df (pandas.Dataframe): must contain ["problem", "algorithm", "n_evaluations"]
            and the target_col as columns.
        target_col (str): column of which to create the monotone version.
        direction (str): "minimize" or "maximize". "minimize" makes the history
            monotonically decreasing, "maximize" means the history will be monotonically
            increasing.

    Retruns:
        pd.Series: target column where all values that are not weak improvements are
            replaced with the best value so far. Index is the same as that of df.

    """
    sorted_df = df.sort_values(["problem", "algorithm", "n_evaluations"])
    grouped = sorted_df.groupby(["problem", "algorithm"])[target_col]

    if direction == "minimize":
        out = grouped.apply(np.minimum.accumulate)
    elif direction == "maximize":
        out = grouped.apply(np.maximum.accumulate)
    else:
        raise ValueError("Only maximize and minimize are allowed as directions.")

    return out


def _normalize(df, col, start_values, target_values):
    """Normalize the values in **col** relative to the total possible improvement.

    We normalize the values of **col** by calculating the share of the distance between
    the start and target values that is still missing.

    Note: This is correct whether we have a minimization or a maximization problem
    because in the case of a maximization both the sign of the numerator and denominator
    in the line where normalized is defined would be switched, i.e. that cancels out.
    (In the case of a maximization the total improvement would be target - start values
    and the currently still missing improvement would be target - current values)

    Args:
        df (pandas.DataFrame): contains the columns **col** and "problem".
        col (str): name of the column to normalize
        start_values (pandas.Series): index are problems, values are start values
        target_values (pandas.Series): index are problems, values are target values.

    Returns:
        pandas.Series: index is the same as that of sr. The lower the value the closer
            the current value is to the target value. 0 means the target value has been
            reached. 1 means the current value is as far from the target value as the
            start value.

    """
    # expand start and target values to the length of the full DataFrame
    start_values = df["problem"].map(start_values)
    target_values = df["problem"].map(target_values)

    normalized = (df[col] - target_values) / (start_values - target_values)
    return normalized


def _clip_histories(df, stopping_criterion, x_precision, y_precision):
    """Shorten the DataFrame to just the evaluations until each algorithm converged.

    Args:
        df (pandas.DataFrame): index levels are ['problem', 'algorithm', 'evaluation'].
            Columns must include "monotone_criterion_normalized" if stopping_criterion
            includes y and "monotone_parameter_distance_normalized" if x is in
            the stopping_criterion.
        stopping_criterion (str): one of "x_and_y", "x_or_y", "x", "y".
        x_precision (float): when an algorithm's parameters are closer than this to the
            true solution's parameters, the algorithm is counted as having converged.
        y_precision (float): when an algorithm's criterion value is closer than this to
            the solution value, the algorithm is counted as having converged.

    Returns:
        shortened (pandas.DataFrame): the entered DataFrame with all histories
            shortened to stop once conversion according to the given criteria is
            reached.
        converged_info (pandas.DataFrame): columns are the algorithms, index are the
            problems. The values are boolean and True when the algorithm arrived at
            the solution with the desired precision.

    """
    # drop problems with no known solution
    if "x" in stopping_criterion:
        df = df[df["monotone_parameter_distance_normalized"].notnull()]
    if "y" in stopping_criterion:
        df = df[df["monotone_criterion_normalized"].notnull()]

    # determine convergence in the known problems
    if "x" in stopping_criterion:
        x_converged = df["monotone_parameter_distance_normalized"] < x_precision
    if "y" in stopping_criterion:
        y_converged = df["monotone_criterion_normalized"] < y_precision

    # determine converged function evaluations
    if stopping_criterion == "y":
        converged = y_converged
    elif stopping_criterion == "x":
        converged = x_converged
    elif stopping_criterion == "x_and_y":
        converged = y_converged & x_converged
    elif stopping_criterion == "x_or_y":
        converged = y_converged | x_converged
    else:
        raise NotImplementedError(
            f"You specified {stopping_criterion} as stopping_criterion but only the "
            "following are allowed: 'x_and_y', 'x_or_y', 'x', or 'y'."
        )

    first_converged = _find_first_converged(converged, df)

    # keep first converged and non-converged
    shortened = df[~converged | first_converged]

    # create converged_info
    converged.index = pd.MultiIndex.from_frame(df[["problem", "algorithm"]])
    grouped = converged.groupby(["problem", "algorithm"])
    converged_info = grouped.any().unstack("algorithm")

    return shortened, converged_info


def _find_first_converged(converged, df):
    """Identify the first converged entry for each problem run.

    Args:
        converged (pandas.Series): same index as df, True where an algorithm has gotten
            sufficiently close to the solution.
        df (pandas.DataFrame): contains the "problem", "algorithm" and "n_evaluations"
            columns.

    Returns:
        pandas.Series: same index as converged. Only True for the first converged entry
            for each problem run, i.e. problem and algorithm combination.

    """
    # this function can probably be implemented much quicker and easier by shifting
    # the converged Series to identify the first converged entries

    converged_with_multi_index = converged.copy(deep=True)
    multi_index = pd.MultiIndex.from_frame(
        df[["problem", "algorithm", "n_evaluations"]]
    )
    converged_with_multi_index.index = multi_index

    only_converged = converged_with_multi_index[converged_with_multi_index]
    first_true_indices = only_converged.groupby(["problem", "algorithm"]).idxmin()
    first_trues = pd.Series(
        converged_with_multi_index.index.isin(first_true_indices.values),
        index=converged.index,
    )
    return first_trues
