import numpy as np
import pandas as pd


def create_performance_df(
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
            - distance_to_optimal_params
            - distance_to_optimal_params_normalized
            - monotone_distance_to_optimal_params
            - monotone_distance_to_optimal_params_normalized

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
    x_dist_sr = _get_history_of_the_distance_to_optimal_params(results, x_opt)
    df = pd.concat([time_sr, criterion_sr, x_dist_sr], axis=1)

    # make tidy
    df.index = df.index.rename({"evaluation": "n_evaluations"})
    df = df.sort_index().reset_index()

    first_evaluations = df.query("n_evaluations == 0").groupby("problem")
    f_0 = first_evaluations["criterion"].mean()
    x_0_dist = first_evaluations["distance_to_optimal_params"].mean()
    x_opt_dist = {name: 0 for name in problems}

    # normalizations
    df["criterion_normalized"] = _normalize(
        df=df, col="criterion", start_values=f_0, target_values=f_opt
    )
    df["distance_to_optimal_params_normalized"] = _normalize(
        df=df,
        col="distance_to_optimal_params",
        start_values=x_0_dist,
        target_values=x_opt_dist,
    )
    # create monotone versions of columns
    df["monotone_criterion"] = _make_history_monotone(df, "criterion")
    df["monotone_distance_to_optimal_params"] = _make_history_monotone(
        df, "distance_to_optimal_params"
    )
    df["monotone_criterion_normalized"] = _make_history_monotone(
        df, "criterion_normalized"
    )
    df["monotone_distance_to_optimal_params_normalized"] = _make_history_monotone(
        df, "distance_to_optimal_params_normalized"
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


def _get_history_of_the_distance_to_optimal_params(results, x_opt):
    """Calculate the history of the distances to the optimal parameters.

    Args:
        results (dict): estimagic benchmarking results dictionary. Keys are
            tuples of the form (problem, algorithm), values are dictionaries of the
            collected information on the benchmark run, including 'criterion_history'
            and 'time_history'.
        x_opt (dict): the keys are the problems, the values are pandas.Series with the
            optimal parameters for the respective problem.

    Returns:
        pandas.Series: index levels are "problem", "algorithm", "evaluation". The name
            is "distance_to_optimal_params".

    """
    x_dist_history = {}
    for (prob, algo), res in results.items():
        param_history = res["params_history"]
        x_dist_history[(prob, algo)] = pd.Series(
            np.linalg.norm(param_history - x_opt[prob], axis=1)
        )

    sr = pd.concat(x_dist_history)
    sr.index.names = ["problem", "algorithm", "evaluation"]
    sr.name = "distance_to_optimal_params"
    return sr


def _make_history_monotone(df, target_col, sorting_cols=None, direction="minimize"):
    """Create a monotone Series, i.e. the best so far instead of the current evaluation.

    Args:
        df (pandas.Dataframe): must contain the sorting_cols and the target_col as
            columns.
        target_col (str): column of which to create the monotone version.
        sorting_cols (list): columns on which to make the histories monotone. The
            default is ["problem", "algorithm", "n_evaluations"].
        direction (str): "minimize" or "maximize". "minimize" makes the history
            monotonically decreasing, "maximize" means the history will be monotonically
            increasing.

    Retruns:
        pd.Series: target column where all values that are not weak improvements are
            replaced with the best value so far. Index is the same as that of df.

    """
    if sorting_cols is None:
        sorting_cols = ["problem", "algorithm", "n_evaluations"]
    sorted_df = df.sort_values(sorting_cols)

    is_first_entry = sorted_df["n_evaluations"] == 0
    sr = sorted_df[target_col]

    if direction == "minimize":
        # It is very important not to rewrite the second statement to
        # sr.diff() <= 0 because the treatment of NaNs would change
        keep = is_first_entry | ~(sr.diff() > 0)
    else:
        # It is very important not to rewrite the second statement to
        # sr.diff() >= 0 because the treatment of NaNs would change
        keep = is_first_entry | ~(sr.diff() < 0)
    with_nans = sr.where(keep, np.nan)

    out = with_nans.fillna(method="ffill")

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

    Args: df (pandas.DataFrame): contains the columns **col** and "problem". col (str):
        name of the column to normalize start_values (pandas.Series): index are
        problems, values are start values target_values (pandas.Series): index are
        problems, values are target values.

    Returns: pandas.Series: index is the same as that of sr. The lower the value the
        closer the current value is to the target value. 0 means the target value has
        been reached. 1 means the current value is as far from the target value as the
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
            Columns must include "monotone_criterion".
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
    if "y" in stopping_criterion:
        y_converged = df["monotone_criterion_normalized"] < y_precision

    if "x" in stopping_criterion:
        x_converged = df["monotone_distance_to_optimal_params_normalized"] < x_precision
        if x_converged.isnull().any():
            raise ValueError(
                "You specified x as part of your stopping criterion but the optimal "
                "parameters are not known for every problem in your problem set."
            )

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

    shortened = df[~converged]

    # A prettier solution exists but this does the right thing
    converged.index = pd.MultiIndex.from_frame(df[["problem", "algorithm"]])
    grouped = converged.groupby(["problem", "algorithm"])
    converged_info = grouped.any().unstack("algorithm")

    return shortened, converged_info
