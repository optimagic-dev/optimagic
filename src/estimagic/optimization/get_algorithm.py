import functools
import inspect
import warnings
from functools import partial

import numpy as np
from estimagic.logging.database_utilities import update_row
from estimagic.optimization import ALL_ALGORITHMS
from estimagic.utilities import propose_alternatives


def process_user_algorithm(algorithm):
    """Process the user specfied algorithm.

    If the algorithm is a callable, this function just reads out the algorithm_info
    and available options. If algorithm is a string it loads the algorithm function
    from the available algorithms.

    Args:
        algorithm (str or callable): The user specified algorithm.

    Returns:
        callable: the raw internal algorithm
        AlgoInfo: Attributes of the algorithm
        set: The free arguments of the algorithm.

    """
    if isinstance(algorithm, str):
        try:
            # Use ALL_ALGORITHMS and not just AVAILABLE_ALGORITHMS such that the
            # algorithm specific error message with installation instruction will be
            # reached if an optional dependency is not installed.
            algorithm = ALL_ALGORITHMS[algorithm]
        except KeyError:
            proposed = propose_alternatives(algorithm, list(ALL_ALGORITHMS))
            raise ValueError(
                f"Invalid algorithm: {algorithm}. Did you mean {proposed}?"
            ) from None

    algo_info = algorithm._algorithm_info

    arguments = set(inspect.signature(algorithm).parameters)

    if isinstance(algorithm, partial):
        partialed_in = set(algorithm.args).union(set(algorithm.keywords))
        arguments = arguments.difference(partialed_in)

    return algorithm, algo_info, arguments


def get_final_algorithm(
    raw_algorithm,
    algo_info,
    valid_kwargs,
    lower_bounds,
    upper_bounds,
    algo_options,
    logging,
    db_kwargs,
):
    """Get algorithm-function with partialled options.

    The resulting function only depends on ``x``,  the relevant criterion functions
    and derivatives and ``step_id``.

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
    algo_name = algo_info.name

    internal_options = _adjust_options_to_algorithm(
        algo_options=algo_options,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        algo_name=algo_name,
        valid_kwargs=valid_kwargs,
    )

    partialled_algorithm = partial(raw_algorithm, **internal_options)

    raw_algorithm = _add_logging_to_algorithm(
        partialled_algorithm,
        logging=logging,
        db_kwargs=db_kwargs,
    )

    return raw_algorithm


def _add_logging_to_algorithm(algorithm=None, *, logging=None, db_kwargs=None):
    """Add logging of status to the algorithm."""

    def decorator_add_logging_to_algorithm(algorithm):
        @functools.wraps(algorithm)
        def wrapper_add_logging_algorithm(**kwargs):
            step_id = int(kwargs["step_id"])

            _kwargs = {k: v for k, v in kwargs.items() if k != "step_id"}

            if logging:
                update_row(
                    data={"status": "running"},
                    rowid=step_id,
                    table_name="steps",
                    **db_kwargs,
                )

            for task in ["criterion", "derivative", "criterion_and_derivative"]:
                if task in _kwargs:
                    _kwargs[task] = partial(
                        _kwargs[task], fixed_log_data={"step": step_id}
                    )

            res = algorithm(**_kwargs)

            if logging:
                update_row(
                    data={"status": "complete"},
                    rowid=step_id,
                    table_name="steps",
                    **db_kwargs,
                )

            return res

        return wrapper_add_logging_algorithm

    if callable(algorithm):
        return decorator_add_logging_to_algorithm(algorithm)
    else:
        return decorator_add_logging_to_algorithm


def _adjust_options_to_algorithm(
    algo_options,
    lower_bounds,
    upper_bounds,
    algo_name,
    valid_kwargs,
):
    """Reduce the algo_options and check if bounds are compatible with algorithm."""

    # convert algo option keys to valid Python arguments
    algo_options = {key.replace(".", "_"): val for key, val in algo_options.items()}

    reduced = {key: val for key, val in algo_options.items() if key in valid_kwargs}

    ignored = {key: val for key, val in algo_options.items() if key not in valid_kwargs}

    if ignored:
        warnings.warn(
            "The following algo_options were ignored because they are not compatible "
            f"with {algo_name}:\n\n {ignored}"
        )

    if "lower_bounds" not in valid_kwargs and not (lower_bounds == -np.inf).all():
        raise ValueError(
            f"{algo_name} does not support lower bounds but your optimization "
            "problem has lower bounds (either because you specified them explicitly "
            "or because they were implied by other constraints)."
        )

    if "upper_bounds" not in valid_kwargs and not (upper_bounds == np.inf).all():
        raise ValueError(
            f"{algo_name} does not support upper bounds but your optimization "
            "problem has upper bounds (either because you specified them explicitly "
            "or because they were implied by other constraints)."
        )

    if "lower_bounds" in valid_kwargs:
        reduced["lower_bounds"] = lower_bounds

    if "upper_bounds" in valid_kwargs:
        reduced["upper_bounds"] = upper_bounds

    return reduced
