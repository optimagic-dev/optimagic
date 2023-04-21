import numpy as np
import pandas as pd


def process_benchmark_results(
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
    histories = []
    infos = []

    for (problem_name, algorithm_name), result in results.items():
        history, is_converged = _process_one_result(
            problem=problems[problem_name],
            result=result,
            stopping_criterion=stopping_criterion,
            x_precision=x_precision,
            y_precision=y_precision,
        )
        history["problem"] = problem_name
        history["algorithm"] = algorithm_name
        histories.append(history)

        info = {
            "problem": problem_name,
            "algorithm": algorithm_name,
            "is_converged": is_converged,
        }
        infos.append(info)

    histories = pd.concat(histories, ignore_index=True)
    infos = pd.DataFrame(infos).set_index(["problem", "algorithm"]).unstack()
    infos.columns = [tup[1] for tup in infos.columns]

    return histories, infos


def _process_one_result(
    problem,
    result,
    stopping_criterion,
    x_precision,
    y_precision,
):
    # input processing
    assert isinstance(x_precision, float)
    assert isinstance(y_precision, float)

    # needed while we use pandas for results ==========================================================
    result = result.copy()
    for key in ["criterion_history", "time_history", "batches_history"]:
        result[key] = result[key].to_numpy().tolist()
    result["params_history"] = list(result["params_history"].to_numpy())

    # extract information
    _params_hist = result["params_history"]
    _is_noisy = problem["noisy"]
    _solution_crit = problem["solution"]["value"]
    _start_crit = problem["start_criterion"]
    _solution_x = problem["solution"].get("params")
    _start_x = problem["inputs"]["params"]
    _needed_step = np.linalg.norm(_solution_x - _start_x)
    if isinstance(_solution_x, np.ndarray) and not np.isfinite(_solution_x).all():
        _solution_x = None

    # get the noise free criterion value
    if _is_noisy:
        crit_hist = np.array([problem["noise_free_criterion"](p) for p in _params_hist])
        if crit_hist.ndim == 2:
            crit_hist = (crit_hist**2).sum(axis=1)
    else:
        crit_hist = np.array(result["criterion_history"])

    # clip the criterion value at the minimum in case our min was not precise
    crit_hist = np.clip(crit_hist, _solution_crit, np.inf)

    # calculate the different transformations of criterion values
    monotone_crit_hist = np.minimum.accumulate(crit_hist)
    normalized_crit_hist = (crit_hist - _solution_crit) / (_start_crit - _solution_crit)
    normalized_monotone_crit_hist = (monotone_crit_hist - _solution_crit) / (
        _start_crit - _solution_crit
    )

    # calculate the different versions of params distance if we have a solution
    if _solution_x is not None:
        params_dist = np.linalg.norm(np.array(_params_hist - _solution_x), axis=1)
        monotone_params_dist = np.minimum.accumulate(params_dist)
        params_dist_normalized = params_dist / _needed_step
        monotone_params_dist_normalized = monotone_params_dist / _needed_step
    else:
        params_dist = np.full(len(_params_hist), np.nan)
        monotone_params_dist = np.full(len(_params_hist), np.nan)
        params_dist_normalized = np.full(len(_params_hist), np.nan)
        monotone_params_dist_normalized = np.full(len(_params_hist), np.nan)

    # put everything together in a dict
    out_dict = {
        "n_evaluations": np.arange(len(crit_hist)),
        "n_batches": result["batches_history"],
        "walltime": result["time_history"],
        "criterion": crit_hist,
        "criterion_normalized": normalized_crit_hist,
        "monotone_criterion": monotone_crit_hist,
        "monotone_criterion_normalized": normalized_monotone_crit_hist,
        "parameter_distance": params_dist,
        "monotone_parameter_distance": monotone_params_dist,
        "parameter_distance_normalized": params_dist_normalized,
        "monotone_parameter_distance_normalized": monotone_params_dist_normalized,
    }

    # calculate at which iteration the problem has been solved
    if stopping_criterion is not None:
        is_converged_x, x_idx = _check_convergence(params_dist_normalized, x_precision)
        is_converged_y, y_idx = _check_convergence(normalized_crit_hist, y_precision)

        y_idx = np.argmax(normalized_crit_hist <= y_precision)

        flag_aggregators = {
            "x": lambda x, y: x,
            "y": lambda x, y: y,
            "x_and_y": lambda x, y: x and y,
            "x_or_y": lambda x, y: x or y,
        }

        is_converged = flag_aggregators[stopping_criterion](
            x=is_converged_x, y=is_converged_y
        )

        idx_aggregators = {
            "x": lambda x, y: x,
            "y": lambda x, y: y,
            "x_and_y": lambda x, y: max(x, y),
            "x_or_y": lambda x, y: min(x, y),
        }

        solution_idx = idx_aggregators[stopping_criterion](x=x_idx, y=y_idx)

        if is_converged:
            out_dict = {k: v[: solution_idx + 1] for k, v in out_dict.items()}

    # create a DataFrame and add metadata
    out = pd.DataFrame(out_dict)

    return out, is_converged


def _check_convergence(values, threshold):
    boo = values <= threshold
    if boo.any():
        is_converged = True
        idx = np.argmax(boo)
    else:
        is_converged = False
        idx = None
    return is_converged, idx
