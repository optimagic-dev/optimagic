def process_internal_optimizer_result(
    res,
    direction,
    params_from_internal,
):
    """Process results of internal optimizers.

    Args:
        res (dict): Results dictionary of an internal optimizer or multistart optimizer.
        direction (str): One of "maximize" or "minimize". Used to switch sign of
            criterion evaluations.
        params_from_internal (callable): Function that converts internal parameters
            to external ones.

    """
    if isinstance(res, str):
        res = _dummy_result_from_traceback(res)
    else:
        res = _process_one_result(res, direction, params_from_internal)

        if "multistart_info" in res:
            res["multistart_info"] = _process_multistart_info(
                res["multistart_info"],
                direction,
                params_from_internal,
            )
    return res


def _process_one_result(res, direction, params_from_internal):
    res = res.copy()
    p = params_from_internal(res["solution_x"], return_numpy=False)
    res["solution_params"] = p

    if direction == "maximize" and "solution_criterion" in res:
        res["solution_criterion"] = switch_sign(res["solution_criterion"])

    # in the long run we can get some of those from the database if logging was used.
    optional_entries = [
        "solution_criterion",
        "solution_derivative",
        "solution_hessian",
        "n_criterion_evaluations",
        "n_derivative_evaluations",
        "n_iterations",
        "success",
        "reached_convergence_criterion",
        "message",
    ]

    for entry in optional_entries:
        res[entry] = res.get(entry)
    return res


def _process_multistart_info(info, direction, params_from_internal):

    starts = []
    for x in info["start_parameters"]:
        starts.append(params_from_internal(x, return_numpy=False))

    optima = []
    for res in info["local_optima"]:
        processed = _process_one_result(
            res,
            direction=direction,
            params_from_internal=params_from_internal,
        )
        optima.append(processed)

    sample = []
    for x in info["exploration_sample"]:
        sample.append(params_from_internal(x, return_numpy=False))

    if direction == "minimize":
        exploration_res = info["exploration_results"]
    else:
        exploration_res = []
        for res in info["exploration_results"]:
            exploration_res.append(switch_sign(res))

    out = {
        "start_parameters": starts,
        "local_optima": optima,
        "exploration_sample": sample,
        "exploration_results": exploration_res,
    }
    return out


def _dummy_result_from_traceback(candidate):
    if isinstance(candidate, str):
        out = {
            "solution_params": None,
            "solution_criterion": None,
            "solution_derivative": None,
            "solution_hessian": None,
            "n_criterion_evaluations": None,
            "n_derivative_evaluations": None,
            "n_iterations": None,
            "success": False,
            "reached_convergence_criterion": None,
            "message": candidate,
        }
    else:
        out = candidate
    return out


def switch_sign(critval):
    if isinstance(critval, dict):
        out = critval.copy()
        if "value" in critval:
            out["value"] = -critval["value"]
        if "contributions" in critval:
            out["contributions"] = -critval["contributions"]
    else:
        out = -critval
    return out
