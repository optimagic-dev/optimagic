"""Implement Bayesian optimization using bayes_opt."""

from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import NonlinearConstraint

from optimagic import mark
from optimagic.config import IS_BAYESOPT_INSTALLED
from optimagic.exceptions import NotInstalledError
from optimagic.optimization.algo_options import N_RESTARTS
from optimagic.optimization.algorithm import Algorithm, InternalOptimizeResult
from optimagic.optimization.internal_optimization_problem import (
    InternalOptimizationProblem,
)
from optimagic.typing import (
    AggregationLevel,
    PositiveInt,
)

if IS_BAYESOPT_INSTALLED:
    from bayes_opt import BayesianOptimization, acquisition
    from bayes_opt.acquisition import AcquisitionFunction


@mark.minimizer(
    name="bayes_opt",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_BAYESOPT_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    supports_parallelism=False,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,  # temp
    disable_history=False,
)
@dataclass(frozen=True)
class BayesOpt(Algorithm):
    """Bayesian Optimization wrapper using bayes_opt package.

    Args:
        init_points: Number of initial random exploration points
        n_iter: Number of optimization iterations
        verbose: Verbosity level (0-3)
        kappa: Exploration-exploitation trade-off parameter for UCB acquisition
        xi: Exploration-exploitation trade-off parameter for EI and PI acquisitions
        exploration_decay: Rate at which exploration decays over time
            (None for no decay)
        exploration_decay_delay: Delay for decay.
        random_state: Random seed for reproducibility
        acquisition_function: Acquisition function used to guide the Bayesian
            optimization process. Can be one of the following strings:
                - "ucb" or "upper_confidence_bound" for Upper Confidence Bound
                - "ei" or "expected_improvement" for Expected Improvement
                - "poi" or "probability_of_improvement" for Probability of Improvement
            Alternatively, a custom instance of
            `bayes_opt.acquisition.AcquisitionFunction` can be passed.
            If set to None, the default acquisition function defined by the
            `bayes_opt` package is used.
        allow_duplicate_points: Whether to allow duplicate evaluation points
        enable_sdr: Enable Sequential Domain Reduction
        sdr_gamma_osc: Oscillation parameter for SDR
        sdr_gamma_pan: Panning parameter for SDR
        sdr_eta: Zooming parameter for SDR
        sdr_minimum_window: Minimum window size for SDR
        alpha: Noise parameter for Gaussian Process
        n_restarts: Number of times to restart the optimizer

    """

    init_points: PositiveInt = 5
    n_iter: PositiveInt = 25
    verbose: int = 2
    kappa: float = 2.576
    xi: float = 0.01
    exploration_decay: float | None = None
    exploration_decay_delay: int | None = None
    random_state: int | None = None
    acquisition_function: Union[str, AcquisitionFunction, None] = None
    allow_duplicate_points: bool = False
    enable_sdr: bool = False
    sdr_gamma_osc: float = 0.7
    sdr_gamma_pan: float = 1.0
    sdr_eta: float = 0.9
    sdr_minimum_window: float = 0.0
    alpha: float = 1e-6
    n_restarts: int = N_RESTARTS

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_BAYESOPT_INSTALLED:
            raise NotInstalledError(
                "To use the 'bayes_opt' optimizer you need to install bayes_opt. "
                "Use 'pip install bayesian-optimization'. "
                "Check the documentation for more details: "
                "https://bayesian-optimization.github.io/BayesianOptimization/index.html"
            )

        if not (
            problem.bounds.lower is not None
            and problem.bounds.upper is not None
            and np.all(np.isfinite(problem.bounds.lower))
            and np.all(np.isfinite(problem.bounds.upper))
        ):
            raise ValueError(
                "Bayesian optimization requires finite bounds for all parameters. "
                "Bounds cannot be None or infinite."
            )

        pbounds = {
            f"param{i}": (lower, upper)
            for i, (lower, upper) in enumerate(
                zip(problem.bounds.lower, problem.bounds.upper, strict=True)
            )
        }

        acq = self._get_acquisition_function()

        constraint = None
        constraint = self._process_constraints(problem.nonlinear_constraints)

        def objective(**kwargs: dict[str, float]) -> float:
            x = np.array([kwargs[f"param{i}"] for i in range(len(x0))])
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
            random_state=self.random_state,
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

        best_params = optimizer.max["params"]
        best_x = np.array([best_params[f"param{i}"] for i in range(len(x0))])
        best_y = -optimizer.max["target"]  # Un-negate the result

        return InternalOptimizeResult(
            x=best_x,
            fun=best_y,
            success=True,
            n_iterations=self.init_points + self.n_iter,
            n_fun_evals=1 + self.init_points + self.n_iter,
            n_jac_evals=0,
        )

    def _get_acquisition_function(self) -> AcquisitionFunction | None:
        """Create and return the appropriate acquisition function.

        Returns:
            The configured acquisition function or None for default

        Raises:
            ValueError: If acquisition_function is invalid
            TypeError: If acquisition_function is not a string or
            an AcquisitionFunction instance

        """

        if self.acquisition_function is None:
            return None

        # If acquisition_function is  an instance of AcquisitionFunction class
        elif isinstance(self.acquisition_function, AcquisitionFunction):
            return self.acquisition_function

        elif isinstance(self.acquisition_function, str):
            acq_name = self.acquisition_function.lower()
            if acq_name in ["ucb", "upper_confidence_bound"]:
                return acquisition.UpperConfidenceBound(
                    kappa=self.kappa,
                    exploration_decay=self.exploration_decay,
                    exploration_decay_delay=self.exploration_decay_delay,
                    random_state=self.random_state,
                )
            elif acq_name in ["ei", "expected_improvement"]:
                return acquisition.ExpectedImprovement(
                    xi=self.xi,
                    exploration_decay=self.exploration_decay,
                    exploration_decay_delay=self.exploration_decay_delay,
                    random_state=self.random_state,
                )
            elif acq_name in ["poi", "probability_of_improvement"]:
                return acquisition.ProbabilityOfImprovement(
                    xi=self.xi,
                    exploration_decay=self.exploration_decay,
                    exploration_decay_delay=self.exploration_decay_delay,
                    random_state=self.random_state,
                )
            else:
                raise ValueError(
                    f"Invalid acquisition_function: {self.acquisition_function}. "
                    "Must be one of: 'ucb', 'poi', 'ei'"
                )

        else:
            raise TypeError(
                "acquisition_function must be a string, an AcquisitionFunction "
                "instance, or None. "
                f"Got {type(self.acquisition_function)}"
            )

    def _process_constraints(
        self, constraints: Optional[list[dict[str, Any]]]
    ) -> Optional[NonlinearConstraint]:
        """Temporarily skip processing of nonlinear constraints.

        Args:
            constraints: List of constraint dictionaries from the problem

        Returns:
            None. Nonlinear constraint processing is deferred.

        """
        # TODO: Implement proper handling of nonlinear constraints in future.
        return None
