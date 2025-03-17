"""A collection of batch evaluators for process based parallelism.

All batch evaluators have the same interface and any function with the same interface
can be used used as batch evaluator in optimagic.

"""

from joblib import Parallel, delayed

try:
    from pathos.pools import ProcessPool

    pathos_is_available = True
except ImportError:
    pathos_is_available = False

import threading
from typing import Any, Callable, Literal, TypeVar, cast

from optimagic.config import DEFAULT_N_CORES as N_CORES
from optimagic.decorators import catch, unpack
from optimagic.typing import BatchEvaluator, ErrorHandling

T = TypeVar("T")


def pathos_mp_batch_evaluator(
    func: Callable[..., T],
    arguments: list[Any],
    *,
    n_cores: int = N_CORES,
    error_handling: ErrorHandling
    | Literal["raise", "continue"] = ErrorHandling.CONTINUE,
    unpack_symbol: Literal["*", "**"] | None = None,
) -> list[T]:
    """Batch evaluator based on pathos.multiprocess.ProcessPool.

    This uses a patched but older version of python multiprocessing that replaces
    pickling with dill and can thus handle decorated functions.

    Args:
        func (Callable): The function that is evaluated.
        arguments (Iterable): Arguments for the functions. Their interperation
            depends on the unpack argument.
        n_cores (int): Number of cores used to evaluate the function in parallel.
            Value below one are interpreted as one. If only one core is used, the
            batch evaluator disables everything that could cause problems, i.e. in that
            case func and arguments are never pickled and func is executed in the main
            process.
        error_handling (str): Can take the values "raise" (raise the error and stop all
            tasks as soon as one task fails) and "continue" (catch exceptions and set
            the traceback of the raised exception.
            KeyboardInterrupt and SystemExit are always raised.
        unpack_symbol (str or None). Can be "**", "*" or None. If None, func just takes
            one argument. If "*", the elements of arguments are positional arguments for
            func. If "**", the elements of arguments are keyword arguments for func.


    Returns:
        list: The function evaluations.

    """
    if not pathos_is_available:
        raise NotImplementedError(
            "To use the pathos_mp_batch_evaluator, install pathos with "
            "conda install -c conda-forge pathos."
        )

    _check_inputs(func, arguments, n_cores, error_handling, unpack_symbol)
    n_cores = int(n_cores)

    reraise = error_handling in [
        "raise",
        ErrorHandling.RAISE,
        ErrorHandling.RAISE_STRICT,
    ]

    @unpack(symbol=unpack_symbol)
    @catch(default="__traceback__", reraise=reraise)
    def internal_func(*args: Any, **kwargs: Any) -> T:
        return func(*args, **kwargs)

    if n_cores <= 1:
        res = [internal_func(arg) for arg in arguments]
    else:
        p = ProcessPool(nodes=n_cores)
        try:
            res = p.map(internal_func, arguments)
        except Exception as e:
            p.terminate()
            raise e

    return res


def joblib_batch_evaluator(
    func: Callable[..., T],
    arguments: list[Any],
    *,
    n_cores: int = N_CORES,
    error_handling: ErrorHandling
    | Literal["raise", "continue"] = ErrorHandling.CONTINUE,
    unpack_symbol: Literal["*", "**"] | None = None,
) -> list[T]:
    """Batch evaluator based on joblib's Parallel.

    Args:
        func (Callable): The function that is evaluated.
        arguments (Iterable): Arguments for the functions. Their interperation
            depends on the unpack argument.
        n_cores (int): Number of cores used to evaluate the function in parallel.
            Value below one are interpreted as one. If only one core is used, the
            batch evaluator disables everything that could cause problems, i.e. in that
            case func and arguments are never pickled and func is executed in the main
            process.
        error_handling (str): Can take the values "raise" (raise the error and stop all
            tasks as soon as one task fails) and "continue" (catch exceptions and set
            the output of failed tasks to the traceback of the raised exception.
            KeyboardInterrupt and SystemExit are always raised.
        unpack_symbol (str or None). Can be "**", "*" or None. If None, func just takes
            one argument. If "*", the elements of arguments are positional arguments for
            func. If "**", the elements of arguments are keyword arguments for func.


    Returns:
        list: The function evaluations.

    """
    _check_inputs(func, arguments, n_cores, error_handling, unpack_symbol)
    n_cores = int(n_cores) if int(n_cores) >= 2 else 1

    reraise = error_handling in [
        "raise",
        ErrorHandling.RAISE,
        ErrorHandling.RAISE_STRICT,
    ]

    @unpack(symbol=unpack_symbol)
    @catch(default="__traceback__", reraise=reraise)
    def internal_func(*args: Any, **kwargs: Any) -> T:
        return func(*args, **kwargs)

    if n_cores == 1:
        res = [internal_func(arg) for arg in arguments]
    else:
        res = Parallel(n_jobs=n_cores)(delayed(internal_func)(arg) for arg in arguments)

    return res


def threading_batch_evaluator(
    func: Callable[..., T],
    arguments: list[Any],
    *,
    n_cores: int = N_CORES,
    error_handling: ErrorHandling
    | Literal["raise", "continue"] = ErrorHandling.CONTINUE,
    unpack_symbol: Literal["*", "**"] | None = None,
) -> list[T]:
    """Batch evaluator based on Python's threading.

    Args:
        func (Callable): The function that is evaluated.
        arguments (Iterable): Arguments for the functions. Their interperation
            depends on the unpack argument.
        n_cores (int): Number of threads used to evaluate the function in parallel.
            Value below one are interpreted as one.
        error_handling (str): Can take the values "raise" (raise the error and stop all
            tasks as soon as one task fails) and "continue" (catch exceptions and set
            the output of failed tasks to the traceback of the raised exception.
            KeyboardInterrupt and SystemExit are always raised.
        unpack_symbol (str or None). Can be "**", "*" or None. If None, func just takes
            one argument. If "*", the elements of arguments are positional arguments for
            func. If "**", the elements of arguments are keyword arguments for func.

    Returns:
        list: The function evaluations.

    """
    _check_inputs(func, arguments, n_cores, error_handling, unpack_symbol)
    n_cores = int(n_cores) if int(n_cores) >= 2 else 1

    reraise = error_handling in [
        "raise",
        ErrorHandling.RAISE,
        ErrorHandling.RAISE_STRICT,
    ]

    @unpack(symbol=unpack_symbol)
    @catch(default="__traceback__", reraise=reraise)
    def internal_func(*args: Any, **kwargs: Any) -> T:
        return func(*args, **kwargs)

    if n_cores == 1:
        res = [internal_func(arg) for arg in arguments]
    else:
        results = [None] * len(arguments)
        threads = []
        errors = []
        error_lock = threading.Lock()

        def thread_func(index: int, arg: Any) -> None:
            try:
                results[index] = internal_func(arg)
            except Exception as e:
                with error_lock:
                    errors.append(e)

        for i, arg in enumerate(arguments):
            thread = threading.Thread(target=thread_func, args=(i, arg))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        if errors:
            raise errors[0]

        res = cast(list[T], results)
    return res


def _check_inputs(
    func: Callable[..., T],
    arguments: list[Any],
    n_cores: int,
    error_handling: ErrorHandling | Literal["raise", "continue"],
    unpack_symbol: Literal["*", "**"] | None,
) -> None:
    if not callable(func):
        raise TypeError("func must be callable.")

    try:
        arguments = list(arguments)
    except Exception as e:
        raise ValueError("arguments must be list like.") from e

    try:
        int(n_cores)
    except Exception as e:
        raise ValueError("n_cores must be an integer.") from e

    if unpack_symbol not in (None, "*", "**"):
        raise ValueError(
            f"unpack_symbol must be None, '*' or '**', not {unpack_symbol}"
        )

    if error_handling not in [
        "raise",
        "continue",
        ErrorHandling.RAISE,
        ErrorHandling.CONTINUE,
        ErrorHandling.RAISE_STRICT,
    ]:
        raise ValueError(
            "error_handling must be 'raise' or 'continue' or ErrorHandling not "
            f"{error_handling}"
        )


def process_batch_evaluator(
    batch_evaluator: Literal["joblib", "pathos", "threading"]
    | BatchEvaluator = "joblib",
) -> BatchEvaluator:
    batch_evaluator = "joblib" if batch_evaluator is None else batch_evaluator
    if callable(batch_evaluator):
        out = batch_evaluator
    elif isinstance(batch_evaluator, str):
        if batch_evaluator == "joblib":
            out = cast(BatchEvaluator, joblib_batch_evaluator)
        elif batch_evaluator == "pathos":
            out = cast(BatchEvaluator, pathos_mp_batch_evaluator)
        elif batch_evaluator == "threading":
            out = cast(BatchEvaluator, threading_batch_evaluator)
        else:
            raise ValueError(
                "Invalid batch evaluator requested. Currently only 'pathos' and "
                "'joblib' are supported."
            )
    else:
        raise TypeError("batch_evaluator must be a callable or string.")

    return out
