import functools
import itertools
from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Literal, NamedTuple, cast

import numpy as np
from numpy.typing import NDArray

from optimagic import batch_evaluators
from optimagic.config import DEFAULT_N_CORES
from optimagic.differentiation import finite_differences
from optimagic.differentiation.generate_steps import generate_steps
from optimagic.parameters.block_trees import hessian_to_block_tree, matrix_to_block_tree
from optimagic.parameters.bounds import Bounds, get_internal_bounds, pre_process_bounds
from optimagic.pytree import (
    tree_flatten,
    tree_leaves,
    tree_unflatten,
)
from optimagic.typing import BatchEvaluatorLiteral, PyTree, PyTreeNamespace


@dataclass(frozen=True)
class NumdiffResult:
    """Result of a numerical differentiation.

    The following relationship holds for vector-valued functions with vector-valued
    parameters:

    First Derivative:
    -----------------

    - f: R -> R leads to shape (1,), usually called derivative
    - f: R^m -> R leads to shape (m, ), usually called Gradient
    - f: R -> R^n leads to shape (n, 1), usually called Jacobian
    - f: R^m -> R^n leads to shape (n, m), usually called Jacobian

    Second Derivative:
    ------------------

    - f: R -> R leads to shape (1,), usually called second derivative
    - f: R^m -> R leads to shape (m, m), usually called Hessian
    - f: R -> R^n leads to shape (n,), usually called Hessian
    - f: R^m -> R^n leads to shape (n, m, m), usually called Hessian tensor

    Attributes:
        derivative: The estimated derivative at the parameters. The structure of the
            derivative depends on the input parameters and the output of the function.
        func_value: The value of the function at the parameters.

    """

    derivative: PyTree
    func_value: PyTree | None = None


class Evals(NamedTuple):
    pos: NDArray[np.float64]
    neg: NDArray[np.float64]


# The finite difference formulae are written for step arrays with a leading axis for
# multiple steps per direction, but we always take just one step per direction.
N_STEPS = 1
STEP_RATIO = 2.0


def first_derivative(
    func: Callable[[PyTree], PyTree],
    params: PyTree,
    *,
    bounds: Bounds | None = None,
    func_kwargs: dict[str, Any] | None = None,
    method: Literal["central", "forward", "backward"] = "central",
    step_size: float | PyTree | None = None,
    scaling_factor: float | PyTree = 1,
    min_steps: float | PyTree | None = None,
    f0: PyTree | None = None,
    n_cores: int = DEFAULT_N_CORES,
    error_handling: Literal["continue", "raise", "raise_strict"] = "continue",
    batch_evaluator: BatchEvaluatorLiteral | Callable = "joblib",
    unpacker: Callable[[Any], PyTree] | None = None,
) -> NumdiffResult:
    """Evaluate first derivative of func at params according to method and step options.

    Internally, the function is converted such that it maps from a 1d array to a 1d
    array. Then the Jacobian of that function is calculated.

    The parameters and the function output can be optimagic-pytrees; for more details on
    estimagi-pytrees see :ref:`eppytrees`. By default the resulting Jacobian will be
    returned as a block-pytree.

    For a detailed description of all options that influence the step size as well as an
    explanation of how steps are adjusted to bounds in case of a conflict, see
    :func:`~optimagic.differentiation.generate_steps.generate_steps`.

    Args:
        func: Function of which the derivative is calculated.
        params: A pytree. See :ref:`params`.
        bounds: Lower and upper bounds on the parameters. The most general and preferred
            way to specify bounds is an `optimagic.Bounds` object that collects lower,
            upper, soft_lower and soft_upper bounds. The soft bounds are not used during
            numerical differentiation. Each bound type mirrors the structure of params.
            Check our how-to guide on bounds for examples. If params is a flat numpy
            array, you can also provide bounds via any format that is supported by
            scipy.optimize.minimize.
        func_kwargs: Additional keyword arguments for func, optional.
        method: One of ["central", "forward", "backward"], default "central".
        step_size: 1d array of the same length as params.
            step_size * scaling_factor is the absolute value of the step used in the
            finite differences approximation of the derivative.
            If step_size * scaling_factor conflicts with bounds, the actual steps will
            be adjusted. If step_size is not provided, it will be determined according
            to a rule of thumb as long as this does not conflict with min_steps.
        scaling_factor: Scaling factor which is applied to step_size. If it is an
            numpy.ndarray, it needs to be as long as params. scaling_factor is useful if
            you want to increase or decrease the step size relative to the rule-of-thumb
            or user provided step size, for example to benchmark the effect of the step
            size. Default 1.
        min_steps: Minimal possible step sizes that can be chosen to accommodate bounds.
            Must have same length as params. By default min_steps is equal to step_size,
            i.e step size is not decreased beyond what is optimal according to the rule
            of thumb.
        f0: 1d numpy array with func(x), optional.
        n_cores: Number of processes used to parallelize the function evaluations.
            Default 1.
        error_handling: One of "continue" (catch errors and continue to calculate
            derivative estimates. In this case, some derivative estimates can be
            missing but no errors are raised), "raise" (catch errors and continue
            to calculate derivative estimates at first but raise an error if all
            evaluations for one parameter failed) and "raise_strict" (raise an error
            as soon as a function evaluation fails).
        batch_evaluator (str or callable): Name of a pre-implemented batch evaluator
            (currently 'joblib' and 'pathos_mp') or Callable with the same interface
            as the optimagic batch_evaluators.
        unpacker: A callable that takes the output of func and returns the part of the
            output that is needed for the derivative calculation. If None, the output of
            func is used as is. Default None.

    Returns:
        NumdiffResult: A numerical differentiation result.

    """
    bounds = pre_process_bounds(bounds)
    unpacker = _process_unpacker(unpacker)

    # ==================================================================================
    # Convert scalar | pytree arguments to 1d arrays of floats
    # ==================================================================================

    is_fast_path = _is_1d_array(params)

    if not is_fast_path:
        params_leaves, params_treedef = tree_flatten(
            params, namespace=PyTreeNamespace.VALUE
        )
        x = np.array(params_leaves, dtype=np.float64)

        if scaling_factor is not None and not np.isscalar(scaling_factor):
            scaling_factor = np.array(
                tree_leaves(scaling_factor, namespace=PyTreeNamespace.VALUE)
            )

        if min_steps is not None and not np.isscalar(min_steps):
            min_steps = np.array(
                tree_leaves(min_steps, namespace=PyTreeNamespace.VALUE)
            )

        if step_size is not None and not np.isscalar(step_size):
            step_size = np.array(
                tree_leaves(step_size, namespace=PyTreeNamespace.VALUE)
            )
    else:
        x = params.astype(np.float64)

    scaling_factor = _process_scalar_or_array_argument(
        scaling_factor, x, "scaling_factor"
    )
    min_steps = _process_scalar_or_array_argument(min_steps, x, "min_steps")
    step_size = _process_scalar_or_array_argument(step_size, x, "step_size")

    # ==================================================================================

    if np.isnan(x).any():
        raise ValueError("The parameter vector must not contain NaNs.")

    internal_lb, internal_ub = get_internal_bounds(params, bounds=bounds)

    # handle kwargs
    func_kwargs = {} if func_kwargs is None else func_kwargs
    partialed_func = functools.partial(func, **func_kwargs)

    implemented_methods = {"forward", "backward", "central"}
    if method not in implemented_methods:
        raise ValueError(f"Method has to be in {implemented_methods}.")

    # generate the step array
    step_size = generate_steps(
        x=x,
        method=method,
        n_steps=N_STEPS,
        target="first_derivative",
        base_steps=step_size,
        scaling_factor=scaling_factor,
        bounds=Bounds(lower=internal_lb, upper=internal_ub),
        step_ratio=STEP_RATIO,
        min_steps=min_steps,
    )
    step_size = cast(NDArray[np.float64], step_size)

    # generate parameter vectors at which func has to be evaluated as numpy arrays
    evaluation_points: list[float | np.ndarray] = []
    for step_arr in step_size:
        for i, j in product(range(N_STEPS), range(len(x))):
            if np.isnan(step_arr[i, j]):
                evaluation_points.append(np.nan)
            else:
                point = x.copy()
                point[j] += step_arr[i, j]
                evaluation_points.append(point)

    # convert the numpy arrays to whatever is needed by func
    if not is_fast_path:
        evaluation_points = [
            # entries are either a numpy.ndarray or np.nan
            _unflatten_if_not_nan(p, params_treedef)
            for p in evaluation_points
        ]

    # we always evaluate f0, so we can fall back to one-sided derivatives if
    # two-sided derivatives fail. The extra cost is negligible in most cases.
    if f0 is None:
        evaluation_points.append(params)

    # do the function evaluations, including error handling
    batch_error_handling = "raise" if error_handling == "raise_strict" else "continue"
    raw_evals = _nan_skipping_batch_evaluator(
        func=partialed_func,
        arguments=evaluation_points,
        n_cores=n_cores,
        error_handling=batch_error_handling,
        batch_evaluator=batch_evaluator,
    )

    # extract information on exceptions that occurred during function evaluations
    exc_info = "\n\n".join([val for val in raw_evals if isinstance(val, str)])
    raw_evals = [val if not isinstance(val, str) else np.nan for val in raw_evals]

    # store full function value at params as func_value and a processed version of it
    # that we need to calculate derivatives as f0
    if f0 is None:
        f0 = raw_evals[-1]
        raw_evals = raw_evals[:-1]
    func_value = f0

    f0_tree = unpacker(f0)
    scalar_out = np.isscalar(f0_tree)
    vector_out = isinstance(f0_tree, np.ndarray) and f0_tree.ndim == 1

    if scalar_out:
        f0 = np.array([f0_tree], dtype=float)
    elif vector_out:
        f0 = f0_tree.astype(float)
    else:
        f0 = tree_leaves(f0_tree, namespace=PyTreeNamespace.VALUE)
        f0 = np.array(f0, dtype=np.float64)

    # convert the raw evaluations to numpy arrays
    raw_evals_arr = _convert_evals_to_numpy(
        raw_evals=raw_evals,
        unpacker=unpacker,
        namespace=PyTreeNamespace.VALUE,
        is_scalar_out=scalar_out,
        is_vector_out=vector_out,
    )

    # apply finite difference formulae
    evals_data = np.array(raw_evals_arr).reshape(2, N_STEPS, len(x), -1)
    evals_data_transposed = np.transpose(evals_data, axes=(0, 1, 3, 2))
    evals = Evals(pos=evals_data_transposed[0], neg=evals_data_transposed[1])

    jac_candidates = {}
    for m in ["forward", "backward", "central"]:
        jac_candidates[m] = finite_differences.jacobian(evals, step_size, f0, m)

    # get the best derivative estimate out of all derivative estimates that could be
    # calculated, given the function evaluations.
    orders = {
        "central": ["central", "forward", "backward"],
        "forward": ["forward", "backward"],
        "backward": ["backward", "forward"],
    }

    jac = _consolidate_one_step_derivatives(jac_candidates, orders[method])

    # raise error if necessary
    if error_handling in ("raise", "raise_strict") and np.isnan(jac).any():
        raise Exception(exc_info)

    # results processing
    if is_fast_path and vector_out:
        derivative = jac
    elif is_fast_path and scalar_out:
        derivative = jac.flatten()
    else:
        derivative = matrix_to_block_tree(jac, f0_tree, params)

    return NumdiffResult(derivative=derivative, func_value=func_value)


def second_derivative(
    func: Callable[[PyTree], PyTree],
    params: PyTree,
    *,
    bounds: Bounds | None = None,
    func_kwargs: dict[str, Any] | None = None,
    method: Literal[
        "forward", "backward", "central_average", "central_cross"
    ] = "central_cross",
    step_size: float | PyTree | None = None,
    scaling_factor: float | PyTree = 1,
    min_steps: float | PyTree | None = None,
    f0: PyTree | None = None,
    n_cores: int = DEFAULT_N_CORES,
    error_handling: Literal["continue", "raise", "raise_strict"] = "continue",
    batch_evaluator: BatchEvaluatorLiteral | Callable = "joblib",
    unpacker: Callable[[Any], PyTree] | None = None,
) -> NumdiffResult:
    """Evaluate second derivative of func at params according to method and step

    options.

    Internally, the function is converted such that it maps from a 1d array to a 1d
    array. Then the Hessians of that function are calculated. The resulting derivative
    estimate is always a :class:`numpy.ndarray`.

    The parameters and the function output can be pandas objects (Series or DataFrames
    with value column). In that case the output of second_derivative is also a pandas
    object and with appropriate index and columns.

    Detailed description of all options that influence the step size as well as an
    explanation of how steps are adjusted to bounds in case of a conflict,
    see :func:`~optimagic.differentiation.generate_steps.generate_steps`.

    Args:
        func: Function of which the derivative is calculated.
        params: 1d numpy array or
            :class:`pandas.DataFrame` with parameters at which the derivative is
            calculated. If it is a DataFrame, it can contain the columns "lower_bound"
            and "upper_bound" for bounds. See :ref:`params`.
        bounds: Lower and upper bounds on the parameters. The most general and preferred
            way to specify bounds is an `optimagic.Bounds` object that collects lower,
            upper, soft_lower and soft_upper bounds. The soft bounds are not used during
            numerical differentiation. Each bound type mirrors the structure of params.
            Check our how-to guide on bounds for examples. If params is a flat numpy
            array, you can also provide bounds via any format that is supported by
            scipy.optimize.minimize.
        func_kwargs: Additional keyword arguments for func, optional.
        method: One of {"forward", "backward", "central_average", "central_cross"}
            These correspond to the finite difference approximations defined in
            equations [7, x, 8, 9] in Rideout [2009], where ("backward", x) is not found
            in Rideout [2009] but is the natural extension of equation 7 to the backward
            case. Default "central_cross".
        step_size: 1d array of the same length as params.
            step_size * scaling_factor is the absolute value of the step used in the
            finite differences approximation of the derivative.
            If step_size * scaling_factor conflicts with bounds, the actual steps will
            be adjusted. If step_size is not provided, it will be determined according
            to a rule of thumb as long as this does not conflict with min_steps.
        scaling_factor: Scaling factor which is applied to
            step_size. If it is an numpy.ndarray, it needs to be as long as params.
            scaling_factor is useful if you want to increase or decrease the step size
            relative to the rule-of-thumb or user provided step size, for example to
            benchmark the effect of the step size. Default 1.
        min_steps: Minimal possible step sizes that can be chosen to
            accommodate bounds. Must have same length as params. By default min_steps is
            equal to step_size, i.e step size is not decreased beyond what is optimal
            according to the rule of thumb.
        f0: 1d numpy array with func(x), optional.
        n_cores: Number of processes used to parallelize the function
            evaluations. Default 1.
        error_handling: One of "continue" (catch errors and continue to calculate
            derivative estimates. In this case, some derivative estimates can be
            missing but no errors are raised), "raise" (catch errors and continue
            to calculate derivative estimates at first but raise an error if all
            evaluations for one parameter failed) and "raise_strict" (raise an error
            as soon as a function evaluation fails).
        batch_evaluator: Name of a pre-implemented batch evaluator
            (currently 'joblib' and 'pathos_mp') or Callable with the same interface
            as the optimagic batch_evaluators.
        unpacker: A callable that takes the output of func and returns the part of the
            output that is needed for the derivative calculation. If None, the output of
            func is used as is. Default None.


    Returns:
        NumdiffResult: A numerical differentiation result.

    """
    bounds = pre_process_bounds(bounds)
    unpacker = _process_unpacker(unpacker)

    # ==================================================================================
    # Convert scalar | pytree arguments to 1d arrays of floats
    # ==================================================================================

    is_fast_path = _is_1d_array(params)

    if not is_fast_path:
        params_leaves, params_treedef = tree_flatten(
            params, namespace=PyTreeNamespace.VALUE
        )
        x = np.array(params_leaves, dtype=np.float64)

        if scaling_factor is not None and not np.isscalar(scaling_factor):
            scaling_factor = np.array(
                tree_leaves(scaling_factor, namespace=PyTreeNamespace.VALUE)
            )

        if min_steps is not None and not np.isscalar(min_steps):
            min_steps = np.array(
                tree_leaves(min_steps, namespace=PyTreeNamespace.VALUE)
            )

        if step_size is not None and not np.isscalar(step_size):
            step_size = np.array(
                tree_leaves(step_size, namespace=PyTreeNamespace.VALUE)
            )
    else:
        x = params.astype(np.float64)

    scaling_factor = _process_scalar_or_array_argument(
        scaling_factor, x, "scaling_factor"
    )
    min_steps = _process_scalar_or_array_argument(min_steps, x, "min_steps")
    step_size = _process_scalar_or_array_argument(step_size, x, "step_size")

    # ==================================================================================

    unpacker = _process_unpacker(unpacker)

    internal_lb, internal_ub = get_internal_bounds(params, bounds=bounds)

    # handle kwargs
    func_kwargs = {} if func_kwargs is None else func_kwargs
    partialed_func = functools.partial(func, **func_kwargs)

    implemented_methods = {"forward", "backward", "central_average", "central_cross"}
    if method not in implemented_methods:
        raise ValueError(f"Method has to be in {implemented_methods}.")

    # generate the step array
    step_size = generate_steps(
        x=x,
        method=("central" if "central" in method else method),
        n_steps=N_STEPS,
        target="second_derivative",
        base_steps=step_size,
        scaling_factor=scaling_factor,
        bounds=Bounds(lower=internal_lb, upper=internal_ub),
        step_ratio=STEP_RATIO,
        min_steps=min_steps,
    )
    step_size = cast(NDArray[np.float64], step_size)

    # generate parameter vectors at which func has to be evaluated as numpy arrays
    evaluation_points = {
        "one_step": [],
        "two_step": [],
        "cross_step": [],
    }
    for step_arr in step_size:
        # single direction steps
        for i, j in product(range(N_STEPS), range(len(x))):
            if np.isnan(step_arr[i, j]):
                evaluation_points["one_step"].append(np.nan)
            else:
                point = x.copy()
                point[j] += step_arr[i, j]
                evaluation_points["one_step"].append(point)
        # two and cross direction steps
        for i, j, k in product(range(N_STEPS), range(len(x)), range(len(x))):
            if j > k or np.isnan(step_arr[i, j]) or np.isnan(step_arr[i, k]):
                evaluation_points["two_step"].append(np.nan)
                evaluation_points["cross_step"].append(np.nan)
            else:
                point = x.copy()
                point[j] += step_arr[i, j]
                point[k] += step_arr[i, k]
                evaluation_points["two_step"].append(point)
                if j == k:
                    evaluation_points["cross_step"].append(np.nan)
                else:
                    point = x.copy()
                    point[j] += step_arr[i, j]
                    point[k] -= step_arr[i, k]
                    evaluation_points["cross_step"].append(point)

    # convert the numpy arrays to whatever is needed by func
    if not is_fast_path:
        evaluation_points = {
            # entries are either a numpy.ndarray or np.nan, we unflatten only
            step_type: [_unflatten_if_not_nan(p, params_treedef) for p in points]
            for step_type, points in evaluation_points.items()
        }

    # we always evaluate f0, so we can fall back to one-sided derivatives if
    # two-sided derivatives fail. The extra cost is negligible in most cases.
    if f0 is None:
        evaluation_points["one_step"].append(params)

    # do the function evaluations for one and two step, including error handling
    batch_error_handling = "raise" if error_handling == "raise_strict" else "continue"
    raw_evals = _nan_skipping_batch_evaluator(
        func=partialed_func,
        arguments=list(itertools.chain.from_iterable(evaluation_points.values())),
        n_cores=n_cores,
        error_handling=batch_error_handling,
        batch_evaluator=batch_evaluator,
    )

    # extract information on exceptions that occurred during function evaluations
    exc_info = "\n\n".join([val for val in raw_evals if isinstance(val, str)])
    raw_evals = [val if not isinstance(val, str) else np.nan for val in raw_evals]

    n_one_step, n_two_step, n_cross_step = map(len, evaluation_points.values())
    raw_evals = {
        "one_step": raw_evals[:n_one_step],
        "two_step": raw_evals[n_one_step : n_two_step + n_one_step],
        "cross_step": raw_evals[n_two_step + n_one_step :],
    }

    # store full function value at params as func_value and a processed version of it
    # that we need to calculate derivatives as f0
    if f0 is None:
        f0 = raw_evals["one_step"][-1]
        raw_evals["one_step"] = raw_evals["one_step"][:-1]
    func_value = f0

    f0_tree = unpacker(f0)
    f0 = tree_leaves(f0_tree, namespace=PyTreeNamespace.VALUE)
    f0 = np.array(f0, dtype=np.float64)

    # convert the raw evaluations to numpy arrays
    raw_evals = {
        step_type: _convert_evals_to_numpy(
            raw_evals=evals, unpacker=unpacker, namespace=PyTreeNamespace.VALUE
        )
        for step_type, evals in raw_evals.items()
    }

    # reshape arrays into dimension (n_steps, dim_f, dim_x) or (n_steps, dim_f, dim_x,
    # dim_x) for finite differences
    evals = {}
    evals["one_step"] = _reshape_one_step_evals(raw_evals["one_step"], N_STEPS, len(x))
    evals["two_step"] = _reshape_two_step_evals(raw_evals["two_step"], N_STEPS, len(x))
    evals["cross_step"] = _reshape_cross_step_evals(
        raw_evals["cross_step"], N_STEPS, len(x), f0
    )

    # apply finite difference formulae
    hess_candidates = {}
    for m in ["forward", "backward", "central_average", "central_cross"]:
        hess_candidates[m] = finite_differences.hessian(evals, step_size, f0, m)

    # get the best derivative estimate out of all derivative estimates that could be
    # calculated, given the function evaluations.
    orders = {
        "central_cross": ["central_cross", "central_average", "forward", "backward"],
        "central_average": ["central_average", "central_cross", "forward", "backward"],
        "forward": ["forward", "backward", "central_average", "central_cross"],
        "backward": ["backward", "forward", "central_average", "central_cross"],
    }

    hess = _consolidate_one_step_derivatives(hess_candidates, orders[method])

    # raise error if necessary
    if error_handling in ("raise", "raise_strict") and np.isnan(hess).any():
        raise Exception(exc_info)

    # results processing
    derivative = hessian_to_block_tree(hess, f0_tree, params)

    return NumdiffResult(derivative=derivative, func_value=func_value)


def _is_1d_array(candidate: Any) -> bool:
    return isinstance(candidate, np.ndarray) and candidate.ndim == 1


def _reshape_one_step_evals(raw_evals_one_step, n_steps, dim_x):
    """Reshape raw_evals for evaluation points with one step.

    Returned object is a namedtuple with entries 'pos' and 'neg' corresponding to
    positive and negative steps. Each entry will be a numpy array with dimension
    (n_steps, dim_f, dim_x).

    Mathematical:

            evals.pos = (f(x0 + delta_jl e_j))
            evals.neg = (f(x0 - delta_jl e_j))

        for j=1,...,dim_x and l=1,...,n_steps

    """
    evals = np.array(raw_evals_one_step).reshape(2, n_steps, dim_x, -1)
    evals = evals.swapaxes(2, 3)
    evals = Evals(pos=evals[0], neg=evals[1])
    return evals


def _process_unpacker(
    unpacker: None | Callable[[Any], PyTree],
) -> Callable[[Any], PyTree]:
    """Process the user provided unpacker function.

    If the unpacker was None, we set it to the identity.

    """
    if unpacker is None:
        unpacker = lambda x: x
    else:
        raw_unpacker = unpacker

        def unpacker(x):
            if isinstance(x, float) and np.isnan(x):
                return x
            return raw_unpacker(x)

    return unpacker


def _process_scalar_or_array_argument(candidate, x, name):
    if candidate is None:
        return None

    if np.isscalar(candidate):
        return np.full_like(x, candidate, dtype=np.float64)
    else:
        try:
            candidate = np.asarray(candidate, dtype=np.float64)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            msg = f"{name} must be a scalar or have the same structure as params."
            raise ValueError(msg) from e

        if len(candidate) != len(x) or candidate.ndim != 1:
            msg = f"{name} must be a scalar or have the same structure as params."
            raise ValueError(msg)
    return candidate


def _reshape_two_step_evals(raw_evals_two_step, n_steps, dim_x):
    """Reshape raw_evals for evaluation points with two steps.

    Returned object is a namedtuple with entries 'pos' and 'neg' corresponding to
    positive and negative steps. Each entry will be a numpy array with dimension
    (n_steps, dim_f, dim_x, dim_x). Since the array is, by definition, symmetric over
    the last two dimensions, the function is not evaluated on both sides to save
    computation time and the information is simply copied here.

    Mathematical:

            evals.pos = (f(x0 + delta_jl e_j + delta_kl e_k))
            evals.neg = (f(x0 - delta_jl e_j - delta_kl e_k))

        for j,k=1,...,dim_x and l=1,...,n_steps

    """
    tril_idx = np.tril_indices(dim_x, -1)
    evals = np.array(raw_evals_two_step).reshape(2, n_steps, dim_x, dim_x, -1)
    evals = evals.transpose(0, 1, 4, 2, 3)
    evals[..., tril_idx[0], tril_idx[1]] = evals[..., tril_idx[1], tril_idx[0]]
    evals = Evals(pos=evals[0], neg=evals[1])
    return evals


def _reshape_cross_step_evals(raw_evals_cross_step, n_steps, dim_x, f0):
    """Reshape raw_evals for evaluation points with cross steps.

    Returned object is a namedtuple with entries 'pos' and 'neg' corresponding to
    positive and negative steps. Each entry will be a numpy array with dimension
    (n_steps, dim_f, dim_x, dim_x). Since the array is, by definition, symmetric over
    the last two dimensions, the function is not evaluated on both sides to save
    computation time and the information is simply copied here. In comparison to the
    two_step case, however, this symmetry holds only over the dimension 'pos' and 'neg'.
    That is, the lower triangular of the last two dimensions of 'pos' must equal the
    upper triangular of the last two dimensions of 'neg'. Further, the diagonal of the
    last two dimensions must be equal to f0.

    Mathematical:

            evals.pos = (f(x0 + delta_jl e_j - delta_kl e_k))
            evals.neg = (f(x0 - delta_jl e_j + delta_kl e_k))

        for j,k=1,...,dim_x and l=1,...,n_steps

    """
    tril_idx = np.tril_indices(dim_x, -1)
    diag_idx = np.diag_indices(dim_x)
    evals = np.array(raw_evals_cross_step).reshape(2, n_steps, dim_x, dim_x, -1)
    evals = evals.transpose(0, 1, 4, 2, 3)
    evals[0][..., tril_idx[0], tril_idx[1]] = evals[1][..., tril_idx[1], tril_idx[0]]
    evals[0][..., diag_idx[0], diag_idx[1]] = np.atleast_2d(f0).T[np.newaxis, ...]
    evals = Evals(pos=evals[0], neg=evals[0].swapaxes(2, 3))
    return evals


def _convert_evals_to_numpy(
    raw_evals, unpacker, namespace, is_scalar_out=False, is_vector_out=False
):
    """Harmonize the output of the function evaluations.

    The raw_evals might contain dictionaries of which we only need one entry, scalar
    np.nan where we need arrays filled with np.nan or pandas objects. The processed
    evals only contain numpy arrays.

    """
    # get rid of additional output
    evals = [unpacker(val) for val in raw_evals]

    # convert pytrees to arrays
    if is_scalar_out:
        evals = [
            np.array([val], dtype=float) if not _is_scalar_nan(val) else val
            for val in evals
        ]

    elif is_vector_out:
        evals = [val.astype(float) if not _is_scalar_nan(val) else val for val in evals]
    else:
        evals = [
            (
                np.array(tree_leaves(val, namespace=namespace), dtype=np.float64)
                if not _is_scalar_nan(val)
                else val
            )
            for val in evals
        ]

    # find out the correct output shape
    try:
        array = next(x for x in evals if hasattr(x, "shape") or isinstance(x, dict))
        out_shape = array.shape
    except StopIteration:
        out_shape = "scalar"

    # convert to correct output shape
    if out_shape == "scalar":
        evals = [np.atleast_1d(val) for val in evals]
    else:
        for i in range(len(evals)):
            if isinstance(evals[i], float) and np.isnan(evals[i]):
                evals[i] = np.full(out_shape, np.nan)

    return evals


def _consolidate_one_step_derivatives(candidates, preference_order):
    """Replace missing derivative estimates of preferred method with others.

    Args:
        candidates (dict): Dictionary with derivative estimates from different methods.
        preference_order (list): Order on (a subset of) the keys in candidates. Earlier
        entries are preferred.

    Returns:
        consolidated (np.ndarray): Array of same shape as input derivative estimates.

    """
    preferred, others = preference_order[0], preference_order[1:]
    consolidated = candidates[preferred].copy()
    for other in others:
        consolidated = np.where(np.isnan(consolidated), candidates[other], consolidated)

    return consolidated.reshape(consolidated.shape[1:])


def _nan_skipping_batch_evaluator(
    func, arguments, n_cores, error_handling, batch_evaluator
):
    """Evaluate func at each entry in arguments, skipping np.nan entries.

    The function is only evaluated at inputs that are not a scalar np.nan.
    The outputs corresponding to skipped inputs as well as for inputs on which func
    returns np.nan are np.nan.

    Args:
        func (function): Python function that returns a numpy array. The shape
            of the output of func has to be the same for all elements in arguments.
        arguments (list): List with inputs for func.
        n_cores (int): Number of processes.

    Returns:
        evaluations (list): The function evaluations, same length as arguments.

    """
    # extract information
    nan_indices = {
        i for i, arg in enumerate(arguments) if isinstance(arg, float) and np.isnan(arg)
    }
    real_args = [arg for i, arg in enumerate(arguments) if i not in nan_indices]

    # get the batch evaluator if it was provided as string
    if not callable(batch_evaluator):
        batch_evaluator = getattr(
            batch_evaluators, f"{batch_evaluator}_batch_evaluator"
        )

    # evaluate functions
    evaluations = batch_evaluator(
        func=func, arguments=real_args, n_cores=n_cores, error_handling=error_handling
    )

    # combine results
    evaluations = iter(evaluations)
    results = []
    for i in range(len(arguments)):
        if i in nan_indices:
            results.append(np.nan)
        else:
            results.append(next(evaluations))

    return results


def _is_scalar_nan(value):
    return isinstance(value, float) and np.isnan(value)


def _unflatten_if_not_nan(leaves, treedef):
    if isinstance(leaves, np.ndarray):
        out = tree_unflatten(treedef, leaves)
    else:
        out = leaves
    return out
