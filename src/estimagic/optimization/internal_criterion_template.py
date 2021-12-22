import datetime
import warnings

import numpy as np
from estimagic.differentiation.derivatives import first_derivative
from estimagic.exceptions import get_traceback
from estimagic.logging.database_utilities import append_row
from estimagic.optimization.process_results import switch_sign
from estimagic.utilities import hash_array


DERIVATIVE_ERROR_MESSAGE = (
    "Error during derivative evaluation at parameters at which the criterion has "
    "already been evaluated and used by the optimizer before. Thus it is not possible "
    "to simply replace the criterion function by a penalty function."
)


CRITERION_ERROR_MESSAGE = (
    "Error during criterion evaluation at parameters at which the derivative has "
    "already been evaluated and used by the optimizer before. Thus it is not possible "
    "to simply replace the criterion function by a penalty function."
)


NO_PRIMARY_MESSAGE = (
    "The primary criterion entry of the {} algorithm is {} but the output of your "
    "criterion function only contains the entries:\n{}"
)


def internal_criterion_and_derivative_template(
    x,
    *,
    task,
    direction,
    criterion,
    params,
    reparametrize_from_internal,
    convert_derivative,
    algorithm_info,
    derivative,
    criterion_and_derivative,
    numdiff_options,
    logging,
    db_kwargs,
    error_handling,
    error_penalty,
    first_criterion_evaluation,
    cache,
    cache_size,
    fixed_log_data,
):
    """Template for the internal criterion and derivative function.

    The internal criterion and derivative function only has the arguments x and task
    and algorithm_info. The other arguments will be partialed in by estimagic at some
    point. Algorithm_info and possibly even task will be partialed in by the algorithm.

    That is the reason why this function is called a template.

    Args:
        x (np.ndarray): 1d numpy array with internal parameters.
        task (str): One of "criterion", "derivative" and "criterion_and_derivative".
        direction (str): One of "maximize" or "minimize"
        criterion (callable): (partialed) user provided criterion function that takes a
            parameter dataframe as only argument and returns a scalar, an array like
            object or a dictionary. See :ref:`criterion`.
        params (pd.DataFrame): see :ref:`params`
        reparametrize_from_internal (callable): Function that takes x and returns a
            numpy array with the values of the external parameters.
        convert_derivative (callable): Function that takes the derivative of criterion
            at the external version of x and x and returns the derivative
            of the internal criterion.
        algorithm_info (dict): Dict with the following entries:
            "primary_criterion_entry": One of "value", "contributions" and
                "root_contributions" or "dict".
            "parallelizes": Bool that indicates if the algorithm calls the internal
                criterion function in parallel. If so, caching is disabled.
            "needs_scaling": bool
            "name": string
        derivative (callable, optional): (partialed) user provided function that
            calculates the first derivative of criterion. For most algorithm, this is
            the gradient of the scalar output (or "value" entry of the dict). However
            some algorithms (e.g. bhhh) require the jacobian of the "contributions"
            entry of the dict. You will get an error if you provide the wrong type of
            derivative.
        criterion_and_derivative (callable): Function that returns criterion
            and derivative as a tuple. This can be used to exploit synergies in the
            evaluation of both functions. The fist element of the tuple has to be
            exactly the same as the output of criterion. The second has to be exactly
            the same as the output of derivative.
        numdiff_options (dict): Keyword arguments for the calculation of numerical
            derivatives. See :ref:`first_derivative` for details. Note that the default
            method is changed to "forward" for speed reasons.
        logging (bool): Wether logging is used.
        db_kwargs (dict): Dictionary with entries "database", "path" and "fast_logging".
        error_handling (str): Either "raise" or "continue". Note that "continue" does
            not absolutely guarantee that no error is raised but we try to handle as
            many errors as possible in that case without aborting the optimization.
        error_penalty (dict): Dict with the entries "constant" (float) and "slope"
            (float). If the criterion or derivative raise an error and error_handling is
            "continue", return ``constant + slope * norm(params - start_params)`` where
            ``norm`` is the euclidean distance as criterion value and adjust the
            derivative accordingly. This is meant to guide the optimizer back into a
            valid region of parameter space (in direction of the start parameters).
            Note that the constant has to be high enough to ensure that the penalty is
            actually a bad function value. The default constant is 2 times the criterion
            value at the start parameters. The default slope is 0.1.
        first_criterion_evaluation (dict): Dictionary with entries "internal_params",
            "external_params", "output".
        cache (dict): Dictionary used as cache for criterion and derivative evaluations.
        cache_size (int): Number of evaluations that are kept in cache. Default 10.
        fixed_log_data (dict): Dictionary with fixed data to be saved in the database.
            Has the entries "stage" (str) and "substage" (int).

    Returns:
        float, np.ndarray or tuple: If task=="criterion" it returns the output of
            criterion which can be a float or 1d numpy array. If task=="derivative" it
            returns the first derivative of criterion, which is a numpy array.
            If task=="criterion_and_derivative" it returns both as a tuple.

    """
    if algorithm_info["primary_criterion_entry"] == "root_contributions":
        if direction == "maximize":
            msg = (
                "Optimizers that exploit a least squares structure like {} can only be "
                "used for minimization."
            )
            raise ValueError(msg.format(algorithm_info["name"]))

    x_hash = hash_array(x)
    cache_entry = cache.get(x_hash, {})

    to_dos = _determine_to_dos(task, cache_entry, derivative, criterion_and_derivative)

    caught_exceptions = []
    new_criterion, new_derivative, new_external_derivative = None, None, None
    current_params = params.copy()
    external_x = reparametrize_from_internal(x)
    current_params["value"] = external_x

    if to_dos == []:
        pass
    elif "numerical_criterion_and_derivative" in to_dos:

        def func(x):
            external_x = reparametrize_from_internal(x)
            p = params.copy()
            p["value"] = external_x
            return criterion(p)

        options = numdiff_options.copy()
        options["key"] = algorithm_info["primary_criterion_entry"]
        options["f0"] = cache_entry.get("criterion", None)
        options["return_func_value"] = True

        try:
            derivative_dict = first_derivative(func, x, **options)
            new_derivative = {
                algorithm_info["primary_criterion_entry"]: derivative_dict["derivative"]
            }
            new_criterion = derivative_dict["func_value"]
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            caught_exceptions.append(get_traceback())
            if "criterion" in cache_entry:
                raise Exception(DERIVATIVE_ERROR_MESSAGE) from e

    elif "criterion_and_derivative" in to_dos:
        try:
            new_criterion, new_external_derivative = criterion_and_derivative(
                current_params
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            caught_exceptions.append(get_traceback())
            if "criterion" in cache_entry:
                raise Exception(DERIVATIVE_ERROR_MESSAGE) from e

    else:
        if "criterion" in to_dos:
            try:
                new_criterion = criterion(current_params)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                caught_exceptions.append(get_traceback())
                if "derivative" in cache_entry:
                    raise Exception(CRITERION_ERROR_MESSAGE) from e

        if "derivative" in to_dos:
            try:
                new_external_derivative = derivative(current_params)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                caught_exceptions.append(get_traceback())
                if "criterion" in cache_entry:
                    raise Exception(DERIVATIVE_ERROR_MESSAGE) from e

    if new_derivative is None and new_external_derivative is not None:
        if not isinstance(new_external_derivative, dict):
            new_external_derivative = {
                algorithm_info["primary_criterion_entry"]: new_external_derivative
            }

        new_derivative = {
            k: convert_derivative(v, internal_values=x)
            for k, v in new_external_derivative.items()
        }

    if caught_exceptions:
        if error_handling == "continue":
            new_criterion, new_derivative = _penalty_and_derivative(
                x, first_criterion_evaluation, error_penalty, algorithm_info
            )
            warnings.warn("\n\n".join(caught_exceptions))
        else:
            raise Exception("\n\n".join(caught_exceptions))

    if not algorithm_info["parallelizes"] and cache_size >= 1:
        _cache_new_evaluations(new_criterion, new_derivative, x_hash, cache, cache_size)

    new_criterion = _check_and_harmonize_criterion_output(
        cache_entry.get("criterion", new_criterion), algorithm_info
    )

    new_derivative = _check_and_harmonize_derivative(
        cache_entry.get("derivative", new_derivative), algorithm_info
    )

    if (new_criterion is not None or new_derivative is not None) and logging:
        _log_new_evaluations(
            new_criterion=new_criterion,
            new_derivative=new_derivative,
            external_x=external_x,
            caught_exceptions=caught_exceptions,
            db_kwargs=db_kwargs,
            fixed_log_data=fixed_log_data,
        )

    res = _get_output_for_optimizer(
        new_criterion, new_derivative, task, algorithm_info, direction
    )
    return res


def _determine_to_dos(task, cache_entry, derivative, criterion_and_derivative):
    """Determine which functions have to be evaluated at the new parameters.

    Args:
        task (str): One of "criterion", "derivative", "criterion_and_derivative"
        cache_entry (dict): Possibly empty dict.
        derivative (Callable or None): Only used to determine if a closed form
            derivative is available.
        criterion_and_derivative (callable or None): Only used to determine if this
            function is available.

    Returns:
        list: List of functions that have to be evaluated. Possible values are:
            - [] if nothing has to be done
            - ["criterion_and_derivative"]
            - ["numerical_criterion_and_derivative"]
            - ["criterion", "derivative"]
            - ["criterion"]
            - ["derivative"]

    """
    criterion_needed = "criterion" in task and "criterion" not in cache_entry
    derivative_needed = "derivative" in task and "derivative" not in cache_entry

    to_dos = []
    if criterion_and_derivative is not None and criterion_needed and derivative_needed:
        to_dos.append("criterion_and_derivative")
    elif (
        derivative is None
        and criterion_and_derivative is not None
        and derivative_needed
    ):
        to_dos.append("criterion_and_derivative")
    elif derivative is None and derivative_needed:
        to_dos.append("numerical_criterion_and_derivative")
    else:
        if derivative_needed:
            to_dos.append("derivative")
        if criterion_needed:
            to_dos.append("criterion")
    return to_dos


def _penalty_and_derivative(x, first_eval, error_penalty, algorithm_info):
    constant = error_penalty["constant"]
    slope = error_penalty["slope"]
    x0 = first_eval["internal_params"]

    primary = algorithm_info["primary_criterion_entry"]

    if primary == "value":
        penalty = _penalty_value(x, constant, slope, x0)
        derivative = _penalty_value_derivative(x, constant, slope, x0)
    elif primary == "contributions":
        dim_out = len(first_eval["output"][primary])
        penalty = _penalty_contributions(x, constant, slope, x0, dim_out)
        derivative = _penalty_contributions_derivative(x, constant, slope, x0, dim_out)
    elif primary == "root_contributions":
        dim_out = len(first_eval["output"][primary])
        penalty = _penalty_root_contributions(x, constant, slope, x0, dim_out)
        derivative = _penalty_root_contributions_derivative(
            x, constant, slope, x0, dim_out
        )
    else:
        raise ValueError()

    return penalty, derivative


def _penalty_value(x, constant, slope, x0, dim_out=None):
    return constant + slope * np.linalg.norm(x - x0)


def _penalty_contributions(x, constant, slope, x0, dim_out):
    contrib = (constant + slope * np.linalg.norm(x - x0)) / dim_out
    return np.ones(dim_out) * contrib


def _penalty_root_contributions(x, constant, slope, x0, dim_out):
    contrib = np.sqrt((constant + slope * np.linalg.norm(x - x0)) / dim_out)
    return np.ones(dim_out) * contrib


def _penalty_value_derivative(x, constant, slope, x0, dim_out=None):
    return slope * (x - x0) / np.linalg.norm(x - x0)


def _penalty_contributions_derivative(x, constant, slope, x0, dim_out):
    row = slope * (x - x0) / (dim_out * np.linalg.norm(x - x0))
    return np.full((dim_out, len(x)), row)


def _penalty_root_contributions_derivative(x, constant, slope, x0, dim_out):
    inner_deriv = slope * (x - x0) / np.linalg.norm(x - x0)
    outer_deriv = 0.5 / np.sqrt(_penalty_value(x, constant, slope, x0) * dim_out)
    row = outer_deriv * inner_deriv
    return np.full((dim_out, len(x)), row)


def _cache_new_evaluations(new_criterion, new_derivative, x_hash, cache, cache_size):
    cache_entry = cache.get(x_hash, {}).copy()
    if len(cache) >= cache_size:
        # list(dict) returns keys in insertion order: https://tinyurl.com/o464nrz
        oldest_entry = list(cache)[0]
        del cache[oldest_entry]
    if new_criterion is not None:
        cache_entry["criterion"] = new_criterion
    if new_derivative is not None:
        cache_entry["derivative"] = new_derivative
    cache[x_hash] = cache_entry


def _check_and_harmonize_criterion_output(output, algorithm_info):

    algo_name = algorithm_info.get("name", "your algorithm")
    primary = algorithm_info["primary_criterion_entry"]

    if output is not None:
        if np.isscalar(output):
            output = {"value": float(output)}

        if not isinstance(output, dict):
            raise ValueError("The output of criterion must be a scalar or dict.")

        if "contributions" not in output and "root_contributions" in output:
            output["contributions"] = output["root_contributions"] ** 2

        if "value" not in output and "contributions" in output:
            output["value"] = output["contributions"].sum()

        if primary not in output and primary != "dict":
            raise ValueError(
                NO_PRIMARY_MESSAGE.format(algo_name, primary, list(output))
            )
    return output


def _check_and_harmonize_derivative(derivative, algorithm_info):
    primary = algorithm_info["primary_criterion_entry"]

    if not isinstance(derivative, dict) and derivative is not None:
        derivative = {primary: derivative}

    if derivative is None:
        pass
    else:

        if primary not in derivative:
            raise ValueError(
                "If derivative returns a dict and you use an optimizer that works with "
                f"{primary}, the derivative dictionary also must contain {primary} "
                "as a key."
            )

        if "value" in derivative and np.atleast_2d(derivative["value"]).shape[0] != 1:
            raise ValueError("The derivative of a scalar optimizer must be a 1d array.")
        if "contributions" in derivative:
            if len(derivative["contributions"].shape) != 2:
                raise ValueError(
                    "The derivative of an optimizer that exploits a sum "
                    "structure must be a 2d array."
                )
        if "root_contributions" in derivative:
            if len(derivative["root_contributions"].shape) != 2:
                raise ValueError(
                    "The derivative of an optimizer that exploits a least squares "
                    "structure must be a 2d array."
                )

    return derivative


def _log_new_evaluations(
    new_criterion,
    new_derivative,
    external_x,
    caught_exceptions,
    db_kwargs,
    fixed_log_data,
):
    """Write the new evaluations and additional information into the database.

    Note: There are some seemingly unnecessary type conversions because sqlalchemy
    can fail silently when called with numpy dtypes instead of the equivalent python
    types.

    """
    data = {
        "params": external_x,
        "timestamp": datetime.datetime.now(),
        "valid": True,
        **fixed_log_data,
    }

    if new_derivative is not None:
        data["internal_derivative"] = new_derivative

    if caught_exceptions:
        separator = "\n" + "=" * 80 + "\n"
        data["exceptions"] = separator.join(caught_exceptions)
        data["valid"] = False

    if new_criterion is not None:
        data = {**data, **new_criterion}
        data["value"] = float(data["value"])

    name = "optimization_iterations"

    append_row(data, name, **db_kwargs)


def _get_output_for_optimizer(
    new_criterion, new_derivative, task, algorithm_info, direction
):
    primary = algorithm_info["primary_criterion_entry"]

    if "criterion" in task:
        if primary != "dict":
            crit = new_criterion[primary]
            crit = crit if np.isscalar(crit) else np.array(crit)
            crit = crit if direction == "minimize" else -crit
        else:
            crit = new_criterion
            if direction == "maximize":
                crit = switch_sign(crit)

    if "derivative" in task:
        deriv = np.array(new_derivative[primary])
        deriv = deriv if direction == "minimize" else -deriv

    if task == "criterion_and_derivative":
        res = (crit, deriv)
    elif task == "criterion":
        res = crit
    elif task == "derivative":
        res = deriv
    else:
        raise ValueError()

    return res
