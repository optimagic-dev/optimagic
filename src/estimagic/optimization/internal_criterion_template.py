import datetime
import warnings

import numpy as np
from estimagic.differentiation.derivatives import first_derivative
from estimagic.exceptions import get_traceback
from estimagic.exceptions import UserFunctionRuntimeError
from estimagic.logging.database_utilities import append_row
from estimagic.parameters.conversion import aggregate_func_output_to_value
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
    converter,
    algo_info,
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
    and algo_info. The other arguments will be partialed in by estimagic at some
    point. algo_info and possibly even task will be partialed in by the algorithm.

    That is the reason why this function is called a template.

    Args:
        x (np.ndarray): 1d numpy array with internal parameters.
        task (str): One of "criterion", "derivative" and "criterion_and_derivative".
        direction (str): One of "maximize" or "minimize"
        criterion (callable): (partialed) user provided criterion function that takes a
            parameter dataframe as only argument and returns a scalar, an array like
            object or a dictionary. See :ref:`criterion`.
        params (pd.DataFrame): see :ref:`params`
        converter (Converter): NamedTuple with methods to convert between internal
            and external derivatives, parameters and criterion outputs.
        algo_info (AlgoInfo): NamedTuple with attributes
            - primary_criterion_entry
            - name
            - parallelizes
            - disable_cache
            - needs_scaling
            - is_available
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
    x_hash = hash_array(x)
    cache_entry = cache.get(x_hash, {})

    to_dos = _determine_to_dos(task, cache_entry, derivative, criterion_and_derivative)

    caught_exceptions = []
    new_criterion, new_external_criterion = None, None
    new_derivative, new_external_derivative = None, None
    current_params, external_x = converter.params_from_internal(
        x,
        return_type="tree_and_flat",
    )
    if to_dos == []:
        pass
    elif "numerical_criterion_and_derivative" in to_dos:

        def func(x):
            p = converter.params_from_internal(x, "tree")
            crit_full = criterion(p)
            crit_relevant = converter.func_to_internal(crit_full)
            out = {"full": crit_full, "relevant": crit_relevant}
            return out

        options = numdiff_options.copy()
        f0 = cache_entry.get("criterion", None)
        if f0 is not None:
            options["f0"] = {"relevant": f0, "full": None}
        options["key"] = "relevant"
        options["return_func_value"] = True

        try:
            derivative_dict = first_derivative(func, x, **options)
            new_derivative = derivative_dict["derivative"]
            new_criterion = derivative_dict["func_value"]["relevant"]
            new_external_criterion = derivative_dict["func_value"]["full"]
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            tb = get_traceback()
            caught_exceptions.append(tb)
            if "criterion" in cache_entry or error_handling == "raise":
                msg = (
                    "An error occurred when evaluating criterion to calculate a "
                    "numerical derivative during optimization."
                )
                raise UserFunctionRuntimeError(msg) from e
            else:

                msg = (
                    "The following exception was caught when evaluating criterion to "
                    f"calculate a numerical derivative during optimization:\n\n{tb}"
                )
                warnings.warn(msg)

    elif "criterion_and_derivative" in to_dos:
        try:
            new_external_criterion, new_external_derivative = criterion_and_derivative(
                current_params
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            tb = get_traceback()
            caught_exceptions.append(tb)
            if "criterion" in cache_entry or error_handling == "raise":
                msg = (
                    "An error ocurred when evaluating criterion_and_derivative "
                    "during optimization."
                )
                raise UserFunctionRuntimeError(msg) from e
            else:

                msg = (
                    "The following exception was caught when evaluating "
                    f"criterion_and_derivative during optimization:\n\n{tb}"
                )
                warnings.warn(msg)

    else:
        if "criterion" in to_dos:
            try:
                new_external_criterion = criterion(current_params)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                tb = get_traceback()
                caught_exceptions.append(tb)
                if "derivative" in cache_entry or error_handling == "raise":
                    msg = (
                        "An error ocurred when evaluating criterion during "
                        "optimization."
                    )
                    raise UserFunctionRuntimeError(msg) from e
                else:

                    msg = (
                        "The following exception was caught when evaluating "
                        f"criterion during optimization:\n\n{tb}"
                    )
                    warnings.warn(msg)

        if "derivative" in to_dos:
            try:
                new_external_derivative = derivative(current_params)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                tb = get_traceback()
                caught_exceptions.append(tb)
                if "criterion" in cache_entry or error_handling == "raise":
                    msg = (
                        "An error ocurred when evaluating derivative during "
                        "optimization"
                    )
                    raise UserFunctionRuntimeError(msg) from e
                else:

                    msg = (
                        "The following exception was caught when evaluating "
                        f"derivative during optimization:\n\n{tb}"
                    )
                    warnings.warn(msg)

    if new_external_criterion is not None and new_criterion is None:
        new_criterion = converter.func_to_internal(new_external_criterion)

    if new_external_derivative is not None and new_derivative is None:
        new_derivative = converter.derivative_to_internal(new_external_derivative, x)

    if caught_exceptions:
        new_criterion, new_derivative = _penalty_and_derivative(
            x, first_criterion_evaluation, error_penalty, algo_info
        )

    if not (algo_info.parallelizes or algo_info.disable_cache) and cache_size >= 1:
        _cache_new_evaluations(new_criterion, new_derivative, x_hash, cache, cache_size)

    if (new_criterion is not None or new_derivative is not None) and logging:
        if new_criterion is not None:
            scalar_critval = aggregate_func_output_to_value(
                f_eval=new_criterion,
                primary_key=algo_info.primary_criterion_entry,
            )
        else:
            scalar_critval = None

        _log_new_evaluations(
            new_criterion=new_external_criterion,
            new_derivative=new_derivative,
            external_x=external_x,
            caught_exceptions=caught_exceptions,
            db_kwargs=db_kwargs,
            fixed_log_data=fixed_log_data,
            scalar_value=scalar_critval,
        )

    res = _get_output_for_optimizer(
        new_criterion=new_criterion,
        new_derivative=new_derivative,
        task=task,
        direction=direction,
        cache=cache,
        x_hash=x_hash,
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


def _penalty_and_derivative(x, first_eval, error_penalty, algo_info):
    constant = error_penalty["constant"]
    slope = error_penalty["slope"]
    x0 = first_eval["internal_params"]

    primary = algo_info.primary_criterion_entry

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


def _log_new_evaluations(
    new_criterion,
    new_derivative,
    external_x,
    caught_exceptions,
    db_kwargs,
    fixed_log_data,
    scalar_value,
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
        "criterion_eval": new_criterion,
        "value": scalar_value,
        **fixed_log_data,
    }

    if new_derivative is not None:
        data["internal_derivative"] = new_derivative

    if caught_exceptions:
        separator = "\n" + "=" * 80 + "\n"
        data["exceptions"] = separator.join(caught_exceptions)
        data["valid"] = False

    name = "optimization_iterations"

    append_row(data, name, **db_kwargs)


def _get_output_for_optimizer(
    new_criterion,
    new_derivative,
    task,
    direction,
    cache,
    x_hash,
):
    if "criterion" in task and new_criterion is None:
        new_criterion = cache[x_hash]["criterion"]
    if "criterion" in task and direction == "maximize":
        new_criterion = -new_criterion

    if "derivative" in task and new_derivative is None:
        new_derivative = cache[x_hash]["derivative"]
    if "derivative" in task and direction == "maximize":
        new_derivative = -new_derivative

    if task == "criterion":
        out = new_criterion
    elif task == "derivative":
        out = new_derivative
    elif task == "criterion_and_derivative":
        out = (new_criterion, new_derivative)
    else:
        raise ValueError()  # xxxx
    return out
