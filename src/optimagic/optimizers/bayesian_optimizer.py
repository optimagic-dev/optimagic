"""Implement Bayesian optimization using bayes_opt."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Type

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import NonlinearConstraint

from optimagic import mark
from optimagic.config import IS_BAYESOPT_INSTALLED_AND_VERSION_NEWER_THAN_2
from optimagic.exceptions import NotInstalledError
from optimagic.optimization.algo_options import N_RESTARTS
from optimagic.optimization.algorithm import Algorithm, InternalOptimizeResult
from optimagic.optimization.internal_optimization_problem import (
    InternalBounds,
    InternalOptimizationProblem,
)
from optimagic.typing import (
    AggregationLevel,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    UnitIntervalFloat,
)

if TYPE_CHECKING:
    from bayes_opt import BayesianOptimization
    from bayes_opt.acquisition import AcquisitionFunction


@mark.minimizer(
    name="bayes_opt",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_BAYESOPT_INSTALLED_AND_VERSION_NEWER_THAN_2,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=True,
    supports_parallelism=False,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,  # temp
    disable_history=False,
)
@dataclass(frozen=True)
class BayesOpt(Algorithm):
    """Minimize a scalar function using Bayesian Optimization with Gaussian Process.

    This optimizer wraps the BayesianOptimization package :cite:`Nogueira2014`,
    which implements a surrogate model-based global optimization algorithm.
    It works by constructing a posterior distribution over the objective function
    via a Gaussian process that best approximates it. Instead of directly optimizing
    the expensive original function, it uses a proxy optimization problem by finding
    the maximum of an acquisition function, which is computationally cheaper than
    evaluating the original function.

    The algorithm starts by sampling a few initial points (init_points) to gather
    observations of the objective function. These observations are used to fit a
    Gaussian process surrogate model that learns about the function's behavior. The
    optimizer then uses an acquisition function to iteratively select promising new
    points to evaluate, updates its model, and this continues for n_iter iterations.

    This optimizer is well-suited for expensive functions where each evaluation is
    costly (simulations, experiments, model training), black-box optimization where
    gradients are unavailable, and problems with a limited evaluation budget.

    Default parameter values match those of the underlying BayesianOptimization package
    where appropriate. Nonlinear constraints are currently not supported.

    """

    init_points: PositiveInt = 5
    """Number of random points sampled before optimization.

    More points improve initial GP fit but increase evaluation cost. Default = 5.

    """

    n_iter: PositiveInt = 25
    """Number of Bayesian optimization iterations to perform after initial
    exploration."""

    verbose: Literal[0, 1, 2] = 0
    """Verbosity level (0 for silent, 1 for brief, 2 for detailed output)."""

    kappa: NonNegativeFloat = 2.576
    """Exploration-exploitation trade-off parameter for Upper Confidence Bound
    acquisition.

    Controls the balance between exploration and exploitation when using the Upper
    Confidence Bound (UCB) acquisition function. Higher values favor exploration over
    exploitation . This parameter is only used when the acquisition function is "ucb" or
    "upper_confidence_bound". The default value of 2.576 corresponds to a 99% confidence
    interval.

    """

    xi: PositiveFloat = 0.01
    """Exploration-exploitation trade-off parameter for Expected/Probability of
    Improvement.

    Controls the balance between exploration and exploitation for Expected Improvement
    (EI) and Probability of Improvement (POI) acquisition functions. Higher values favor
    exploration over exploitation . This parameter is only used when the acquisition
    function is "ei", "expected_improvement", "poi", or "probability_of_improvement".
    The default value is 0.01.

    """

    exploration_decay: UnitIntervalFloat | None = None
    """Rate at which exploration decays over time during optimization.

    If specified, the exploration parameters (kappa or xi) are multiplied by this factor
    after each iteration, gradually shifting from exploration to exploitation. Must be
    between 0 and 1 (range: (0, 1]) If None, no decay is applied and exploration remains
    constant.

    """

    exploration_decay_delay: NonNegativeInt | None = None
    """Number of iterations to delay before applying exploration decay.

    If specified, exploration decay only begins after this many iterations have
    completed. If None, decay is applied from the first iteration.

    """

    seed: int | None = None
    """Random seed for reproducible results."""

    acquisition_function: (
        str | AcquisitionFunction | Type[AcquisitionFunction] | None
    ) = None
    """Strategy for selecting the next evaluation point during optimization.

    The acquisition function determines how to balance exploration and exploitation when
    selecting the next point to evaluate. Supported options:

    - String: "ucb"/"upper_confidence_bound", "ei"/"expected_improvement",
        "poi"/"probability_of_improvement"
    - AcquisitionFunction instance: Pre-configured acquisition function object
    - AcquisitionFunction class: Class that will be instantiated with default parameters
    - None: Uses package default (UCB for unconstrained, EI for constrained problems)

    """

    allow_duplicate_points: bool = False
    """Whether to allow repeated evaluation of the same point."""

    enable_sdr: bool = False
    """Enable Sequential Domain Reduction (SDR).

    When True, the search domain is iteratively shrunk around promising regions
    using SDR parameters (`sdr_gamma_osc`, `sdr_gamma_pan`, `sdr_eta`,
    `sdr_minimum_window`).

    """

    sdr_gamma_osc: float = 0.7
    """Oscillation shrinkage parameter for SDR.

    Controls how aggressively the search space shrinks in oscillating fashion. Only used
    when enable_sdr is True. Typical range: [0.5, 0.7]. Default = 0.7.

    """

    sdr_gamma_pan: float = 1.0
    """Panning parameter for SDR.

    Controls the panning behavior during domain reduction. Only used when enable_sdr is
    True. Typical value: 1.0. Default = 1.0.

    """

    sdr_eta: float = 0.9
    """Zoom parameter for SDR.

    Only used when enable_sdr is True. Default = 0.9.

    """

    sdr_minimum_window: NonNegativeFloat = 0.0
    """Minimum window size for Sequential Domain Reduction.

    Only used when enable_sdr is True. Default = 0.0.

    """

    alpha: float = 1e-6
    """Noise parameter for the Gaussian Process model.

    Controls the amount of noise assumed in the objective function observations. Default
    is 1e-6.

    """

    n_restarts: int = N_RESTARTS
    """Number of times to restart the optimization."""

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_BAYESOPT_INSTALLED_AND_VERSION_NEWER_THAN_2:
            raise NotInstalledError(
                "To use the 'bayes_opt' optimizer you need to install bayes_opt. "
                "Use 'pip install bayesian-optimization'. "
                "Check the documentation for more details: "
                "https://bayesian-optimization.github.io/BayesianOptimization/index.html"
            )

        from bayes_opt import BayesianOptimization

        pbounds = _process_bounds(problem.bounds)

        acq = _process_acquisition_function(
            acquisition_function=self.acquisition_function,
            kappa=self.kappa,
            xi=self.xi,
            exploration_decay=self.exploration_decay,
            exploration_decay_delay=self.exploration_decay_delay,
            random_seed=self.seed,
        )

        constraint = None
        constraint = self._process_constraints(problem.nonlinear_constraints)

        def objective(**kwargs: dict[str, float]) -> float:
            x = _extract_params_from_kwargs(kwargs)
            return -float(
                problem.fun(x)
            )  # Negate to convert minimization to maximization

        bounds_transformer = None
        if self.enable_sdr:
            from bayes_opt import SequentialDomainReductionTransformer

            bounds_transformer = SequentialDomainReductionTransformer(
                gamma_osc=self.sdr_gamma_osc,
                gamma_pan=self.sdr_gamma_pan,
                eta=self.sdr_eta,
                minimum_window=self.sdr_minimum_window,
            )

        optimizer = BayesianOptimization(
            f=objective,
            pbounds=pbounds,
            acquisition_function=acq,
            constraint=constraint,
            random_state=self.seed,
            verbose=self.verbose,
            bounds_transformer=bounds_transformer,
            allow_duplicate_points=self.allow_duplicate_points,
        )

        # Set Gaussian Process parameters
        optimizer.set_gp_params(alpha=self.alpha, n_restarts_optimizer=self.n_restarts)

        # Use initial point as first probe
        probe_params = {f"param{i}": float(val) for i, val in enumerate(x0)}
        optimizer.probe(
            params=probe_params,
            lazy=True,
        )
        optimizer.maximize(
            init_points=self.init_points,
            n_iter=self.n_iter,
        )

        res = _process_bayes_opt_result(optimizer=optimizer, x0=x0, problem=problem)
        return res

    def _process_constraints(
        self, constraints: list[dict[str, Any]] | None
    ) -> NonlinearConstraint | None:
        """Temporarily skip processing of nonlinear constraints.

        Args:
            constraints: List of constraint dictionaries from the problem

        Returns:
            None. Nonlinear constraint processing is deferred.

        """
        # TODO: Implement proper handling of nonlinear constraints in future.
        return None


def _process_bounds(bounds: InternalBounds) -> dict[str, tuple[float, float]]:
    """Process bounds for bayesian optimization.

    Args:
        bounds: Internal bounds object.

    Returns:
        Dictionary mapping parameter names to (lower, upper) bound tuples.

    Raises:
        ValueError: If bounds are None or infinite.

    """
    if not (
        bounds.lower is not None
        and bounds.upper is not None
        and np.all(np.isfinite(bounds.lower))
        and np.all(np.isfinite(bounds.upper))
    ):
        raise ValueError(
            "Bayesian optimization requires finite bounds for all parameters. "
            "Bounds cannot be None or infinite."
        )

    return {
        f"param{i}": (lower, upper)
        for i, (lower, upper) in enumerate(zip(bounds.lower, bounds.upper, strict=True))
    }


def _extract_params_from_kwargs(params_dict: dict[str, Any]) -> NDArray[np.float64]:
    """Extract parameters from kwargs dictionary.

    Args:
        params_dict: Dictionary with parameter values.

    Returns:
        Array of parameter values.

    """
    return np.array(list(params_dict.values()))


def _process_acquisition_function(
    acquisition_function: (
        str | AcquisitionFunction | Type[AcquisitionFunction] | None
    ),
    kappa: NonNegativeFloat,
    xi: PositiveFloat,
    exploration_decay: float | None,
    exploration_decay_delay: NonNegativeInt | None,
    random_seed: int | None,
) -> AcquisitionFunction | None:
    """Create and return the appropriate acquisition function.

    Args:
        acquisition_function: The acquisition function specification.
            Can be one of the following:
            - A string: "upper_confidence_bound" (or "ucb"), "expected_improvement"
              (or "ei"), "probability_of_improvement" (or "poi")
            - An instance of `AcquisitionFunction`
            - A class inheriting from `AcquisitionFunction`
            - None (uses the default acquisition function from the bayes_opt package)
        kappa: Exploration-exploitation trade-off parameter for Upper Confidence Bound
            acquisition function. Higher values favor exploration over exploitation.
        xi: Exploration-exploitation trade-off parameter for Expected Improvement and
            Probability of Improvement acquisition functions. Higher values favor
            exploration over exploitation.
        exploration_decay: Rate at which exploration parameters (kappa or xi) decay
            over time. None means no decay is applied.
        exploration_decay_delay: Number of iterations before starting the decay.
            None means decay is applied from the start.
        random_seed: Random seed for reproducibility.

    Returns:
        The configured acquisition function instance or None for default.

    Raises:
        ValueError: If acquisition_function is an invalid string.
        TypeError: If acquisition_function is not a string, an AcquisitionFunction
            instance, a class inheriting from AcquisitionFunction, or None.

    """

    from bayes_opt import acquisition

    acquisition_function_aliases = {
        "ucb": "ucb",
        "upper_confidence_bound": "ucb",
        "ei": "ei",
        "expected_improvement": "ei",
        "poi": "poi",
        "probability_of_improvement": "poi",
    }

    if acquisition_function is None:
        return None

    elif isinstance(acquisition_function, str):
        acq_name = acquisition_function.lower()

        if acq_name not in acquisition_function_aliases:
            raise ValueError(
                f"Invalid acquisition_function string: '{acquisition_function}'. "
                f"Must be one of: {', '.join(acquisition_function_aliases.keys())}"
            )

        canonical_name = acquisition_function_aliases[acq_name]

        if canonical_name == "ucb":
            return acquisition.UpperConfidenceBound(
                kappa=kappa,
                exploration_decay=exploration_decay,
                exploration_decay_delay=exploration_decay_delay,
                random_state=random_seed,
            )
        elif canonical_name == "ei":
            return acquisition.ExpectedImprovement(
                xi=xi,
                exploration_decay=exploration_decay,
                exploration_decay_delay=exploration_decay_delay,
                random_state=random_seed,
            )
        elif canonical_name == "poi":
            return acquisition.ProbabilityOfImprovement(
                xi=xi,
                exploration_decay=exploration_decay,
                exploration_decay_delay=exploration_decay_delay,
                random_state=random_seed,
            )
        else:
            raise ValueError(f"Unhandled canonical name: {canonical_name}")

    # If acquisition_function is an instance of AcquisitionFunction class
    elif isinstance(acquisition_function, acquisition.AcquisitionFunction):
        return acquisition_function

    # If acquisition_function is a class inheriting from AcquisitionFunction
    elif isinstance(acquisition_function, type) and issubclass(
        acquisition_function, acquisition.AcquisitionFunction
    ):
        if issubclass(
            acquisition_function, acquisition.ExpectedImprovement
        ) or issubclass(acquisition_function, acquisition.ProbabilityOfImprovement):
            return acquisition_function(
                xi=xi,
                exploration_decay=exploration_decay,
                exploration_decay_delay=exploration_decay_delay,
                random_state=random_seed,
            )
        elif issubclass(acquisition_function, acquisition.UpperConfidenceBound):
            return acquisition_function(
                kappa=kappa,
                exploration_decay=exploration_decay,
                exploration_decay_delay=exploration_decay_delay,
                random_state=random_seed,
            )
        else:
            return acquisition_function()

    else:
        raise TypeError(
            "acquisition_function must be None, a string, "
            "an AcquisitionFunction instance, or a class inheriting from "
            "AcquisitionFunction. "
            f"Got type: {type(acquisition_function).__name__}"
        )


def _process_bayes_opt_result(
    optimizer: BayesianOptimization,
    x0: NDArray[np.float64],
    problem: InternalOptimizationProblem,
) -> InternalOptimizeResult:
    """Convert BayesianOptimization result to InternalOptimizeResult format.

    Args:
        optimizer: The BayesianOptimization instance after optimization
        x0: Initial parameter values
        problem: The internal optimization problem

    Returns:
        InternalOptimizeResult with processed results

    """
    n_evals = len(optimizer.space)

    if optimizer.max is not None:
        best_params = optimizer.max["params"]
        best_x = _extract_params_from_kwargs(best_params)
        best_y = -optimizer.max["target"]  # Un-negate the result
        success = True
        message = "Optimization succeeded"
    else:
        best_x = x0
        best_y = float(problem.fun(x0))
        success = False
        message = (
            "Optimization did not succeed "
            "returning the initial point as the best available result."
        )

    return InternalOptimizeResult(
        x=best_x,
        fun=best_y,
        success=success,
        message=message,
        n_iterations=n_evals,
        n_fun_evals=n_evals,
        n_jac_evals=0,
    )
