import inspect
import warnings
from functools import partial

import numpy as np
from estimagic.logging.database_utilities import update_row
from estimagic.optimization import AVAILABLE_ALGORITHMS
from estimagic.utilities import propose_alternatives


def get_algorithm(
    algorithm,
    lower_bounds,
    upper_bounds,
    algo_options,
    logging,
    db_kwargs,
):
    """Get algorithm-function with partialled optionts

    The resulting function only depends on ``x``,  ``criterion_and_derivative``
    and ``step_id``.

    Args:
        algorithm (str or callable): String with the name of an algorithm or internal
            algorithm function.
        lower_bounds (np.ndarray): 1d numpy array with lower bounds.
        upper_bounds (np.ndarray): 1d numpy array with upper bounds.
        algo_options (dict): Dictionary with additional keyword arguments for the
            algorithm. Entries that are not used by the algorithm are ignored with a
            warning.
        logging (bool): Whether the algorithm should do logging.
        db_kwargs (dict): Dict with the entries "database", "path" and "fast_logging"

    Returns:
        callable: The algorithm.

    """
    internal_algorithm, algo_name = _process_user_algorithm(algorithm)

    internal_options = _adjust_options_to_algorithm(
        algo_options=algo_options,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        algorithm=internal_algorithm,
        algo_name=algo_name,
    )

    partialled_algorithm = partial(internal_algorithm, **internal_options)

    algorithm = partial(
        _algorithm_with_logging_template,
        algorithm=partialled_algorithm,
        logging=logging,
        db_kwargs=db_kwargs,
    )

    return algorithm


def _process_user_algorithm(algorithm):
    """Process the user specfied algorithm.

    Args:
        algorithm (str or callable): The user specified algorithm.

    Returns:
        callable: The internal algorithm.
        str: The name of the algorithm.

    """
    if isinstance(algorithm, str):
        algo_name = algorithm
    else:
        algo_name = getattr(algorithm, "name", "your algorithm")

    if isinstance(algorithm, str):
        try:
            algorithm = AVAILABLE_ALGORITHMS[algorithm]
        except KeyError:
            proposed = propose_alternatives(algorithm, list(AVAILABLE_ALGORITHMS))
            raise ValueError(
                f"Invalid algorithm: {algorithm}. Did you mean {proposed}?"
            ) from None

    return algorithm, algo_name


def _algorithm_with_logging_template(
    criterion_and_derivative,
    x,
    step_id,
    algorithm,
    logging,
    db_kwargs,
):
    """Wrapped algorithm that logs its status.

    Args:
        criterion_and_derivative (callable): Version of the
            ``internal_criterion_and_derivative_template`` all arguments except for
            ``x``, ``task`` and ``fixed_log_data`` have been partialled in.
        x (np.ndarray): Parameter vector.
        step_id (int): Internal id of the optimization step.
        logging (bool): Whether logging is used.
        algorithm (callable): The internal algorithm where all argument except for
            ``x`` and ``criterion_and_derivative`` are already partialled in.
        db_kwargs (dict): Dict with the entries "database", "path" and "fast_logging"


    Returns:
        dict: Same result as internal algorithm.

    """

    if logging:
        step_id = int(step_id)
        update_row(
            data={"status": "running"},
            rowid=step_id,
            table_name="steps",
            **db_kwargs,
        )

    func = partial(criterion_and_derivative, fixed_log_data={"step": step_id})
    res = algorithm(func, x)

    if logging:
        update_row(
            data={"status": "complete"},
            rowid=step_id,
            table_name="steps",
            **db_kwargs,
        )

    return res


def _adjust_options_to_algorithm(
    algo_options, lower_bounds, upper_bounds, algorithm, algo_name
):
    """Reduce the algo_options and check if bounds are compatible with algorithm."""

    # convert algo option keys to valid Python arguments
    algo_options = {key.replace(".", "_"): val for key, val in algo_options.items()}

    valid = set(inspect.signature(algorithm).parameters)

    if isinstance(algorithm, partial):
        partialed_in = set(algorithm.args).union(set(algorithm.keywords))
        valid = valid.difference(partialed_in)

    reduced = {key: val for key, val in algo_options.items() if key in valid}

    ignored = {key: val for key, val in algo_options.items() if key not in valid}

    if ignored:
        warnings.warn(
            "The following algo_options were ignored because they are not compatible "
            f"with {algo_name}:\n\n {ignored}"
        )

    if "lower_bounds" not in valid and not (lower_bounds == -np.inf).all():
        raise ValueError(
            f"{algo_name} does not support lower bounds but your optimization "
            "problem has lower bounds (either because you specified them explicitly "
            "or because they were implied by other constraints)."
        )

    if "upper_bounds" not in valid and not (upper_bounds == np.inf).all():
        raise ValueError(
            f"{algo_name} does not support upper bounds but your optimization "
            "problem has upper bounds (either because you specified them explicitly "
            "or because they were implied by other constraints)."
        )

    if "lower_bounds" in valid:
        reduced["lower_bounds"] = lower_bounds

    if "upper_bounds" in valid:
        reduced["upper_bounds"] = upper_bounds

    return reduced
