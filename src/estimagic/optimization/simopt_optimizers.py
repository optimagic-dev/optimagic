"""Implement `simopt` optimizers.


.. note::
    `simopt`'s SPSA and STRONG support box constraints, i.e. parameter bounds. However,
    for the moment they are not supported.

"""
import numpy as np
from estimagic.config import IS_SIMOPT_INSTALLED
from estimagic.decorators import mark_minimizer
from estimagic.logging.database_utilities import list_of_dicts_to_dict_of_lists
from estimagic.optimization.algo_options import (
    STOPPING_MAX_CRITERION_EVALUATIONS_GLOBAL,
)


@mark_minimizer(
    name="simopt_adam",
    primary_criterion_entry="value",
    needs_scaling=True,
    is_available=IS_SIMOPT_INSTALLED,
)
def simopt_adam(
    criterion,
    derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    stopping_max_iterations=STOPPING_MAX_CRITERION_EVALUATIONS_GLOBAL,
    crn_across_solns=True,
    r=1,
    beta_1=0.9,
    beta_2=0.999,
    alpha=1.0,
    epsilon=10e-8,
    sensitivity=10e-7,
):
    """Minimize a scalar function using the ADAM algorithm from SimOpt.

    For details see :ref:`list_of_scipy_algorithms`.

    """
    solver_options = {
        "crn_across_solns": crn_across_solns,
        "r": r,
        "beta_1": beta_1,
        "beta_2": beta_2,
        "alpha": alpha,
        "epsilon": epsilon,
        "sensitivity": sensitivity,
    }

    out = _minimize_simopt(
        algorithm="ADAM",
        criterion=criterion,
        derivative=derivative,
        x=x,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        solver_options=solver_options,
        budget=stopping_max_iterations,
    )
    return out


@mark_minimizer(
    name="simopt_astrodf",
    primary_criterion_entry="value",
    needs_scaling=True,
    is_available=IS_SIMOPT_INSTALLED,
)
def simopt_astrodf(
    criterion,
    x,
    lower_bounds,
    upper_bounds,
    *,
    stopping_max_iterations=STOPPING_MAX_CRITERION_EVALUATIONS_GLOBAL,
    bounds_padding=1e-8,
    crn_across_solns=True,
    delta_max=50.0,
    eta_1=0.1,
    eta_2=0.5,
    gamma_1=2.0,
    gamma_2=0.5,
    w=0.85,
    mu=1_000.0,
    beta=10.0,
    lambda_min=None,
    simple_solve=False,
    criticality_select=True,
    criticality_threshold=0.1,
):
    """Minimize a scalar function using the ASTRODF algorithm from SimOpt.

    For details see :ref:`list_of_scipy_algorithms`.

    """
    solver_options = {
        "crn_across_solns": crn_across_solns,
        "delta_max": delta_max,
        "eta_1": eta_1,
        "eta_2": eta_2,
        "gamma_1": gamma_1,
        "gamma_2": gamma_2,
        "w": w,
        "mu": mu,
        "beta": beta,
        "lambda_min": 2 * len(x) + 1 if lambda_min is None else lambda_min,
        "simple_solve": simple_solve,
        "criticality_select": criticality_select,
        "criticality_threshold": criticality_threshold,
    }

    # Revert bounds shifting of ASTRODF to improve accuracy. For details see docstring.
    lower_bounds -= 0.01 - bounds_padding
    upper_bounds += 0.01 + bounds_padding

    out = _minimize_simopt(
        algorithm="ASTRODF",
        criterion=criterion,
        derivative=None,
        x=x,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        solver_options=solver_options,
        budget=stopping_max_iterations,
    )
    return out


@mark_minimizer(
    name="simopt_spsa",
    primary_criterion_entry="value",
    needs_scaling=True,
    is_available=IS_SIMOPT_INSTALLED,
)
def simopt_spsa(
    criterion,
    x,
    *,
    stopping_max_iterations=STOPPING_MAX_CRITERION_EVALUATIONS_GLOBAL,
    crn_across_solns=True,
    alpha=0.602,
    gamma=0.101,
    step=0.5,
    gavg=1,
    n_reps=2,
    n_loss=2,
    eval_pct=2 / 3,
    iter_pct=0.1,
):
    """Minimize a scalar function using the SPSA algorithm from SimOpt.

    For details see :ref:`list_of_scipy_algorithms`.

    """
    solver_options = {
        "crn_across_solns": crn_across_solns,
        "alpha": alpha,
        "gamma": gamma,
        "step": step,
        "gavg": gavg,
        "n_reps": n_reps,
        "n_loss": n_loss,
        "eval_pct": eval_pct,
        "iter_pct": iter_pct,
    }

    out = _minimize_simopt(
        algorithm="SPSA",
        criterion=criterion,
        derivative=None,
        x=x,
        lower_bounds=np.full_like(x, -np.inf),
        upper_bounds=np.full_like(x, np.inf),
        solver_options=solver_options,
        budget=stopping_max_iterations,
    )
    return out


@mark_minimizer(
    name="simopt_strong",
    primary_criterion_entry="value",
    needs_scaling=True,
    is_available=IS_SIMOPT_INSTALLED,
)
def simopt_strong(
    criterion,
    x,
    *,
    stopping_max_iterations=STOPPING_MAX_CRITERION_EVALUATIONS_GLOBAL,
    crn_across_solns=True,
    n0=10,
    n_r=1,
    sensitivity=10e-7,
    delta_threshold=1.2,
    delta_T=2.0,  # noqa: N803
    eta_0=0.01,
    eta_1=0.3,
    gamma_1=0.9,
    gamma_2=1.11,
    _lambda=2,
    lambda_2=1.01,
):
    """Minimize a scalar function using the STRONG algorithm from SimOpt.

    For details see :ref:`list_of_scipy_algorithms`.

    """
    solver_options = {
        "crn_across_solns": crn_across_solns,
        "n0": n0,
        "n_r": n_r,
        "sensitivity": sensitivity,
        "delta_threshold": delta_threshold,
        "delta_T": delta_T,
        "eta_0": eta_0,
        "eta_1": eta_1,
        "gamma_1": gamma_1,
        "gamma_2": gamma_2,
        "lambda": _lambda,
        "lambda_2": lambda_2,
    }

    out = _minimize_simopt(
        algorithm="STRONG",
        criterion=criterion,
        derivative=None,
        x=x,
        lower_bounds=np.full_like(x, -np.inf),
        upper_bounds=np.full_like(x, np.inf),
        solver_options=solver_options,
        budget=stopping_max_iterations,
    )
    return out


# ======================================================================================
# Template
# ======================================================================================


def _minimize_simopt(
    algorithm,
    criterion,
    derivative,
    x,
    lower_bounds,
    upper_bounds,
    solver_options,
    budget,
):
    problem = ProblemSpecification(
        criterion=criterion,
        derivative=derivative,
        x=x,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        budget=budget,
    )
    solver = ProblemSolver(
        solver_name=algorithm, problem=problem, solver_fixed_factors=solver_options
    )
    # overwrite method of simopt ProblemSolver class that pickles temporary results
    solver.record_experiment_results = _do_nothing.__get__(solver, ProblemSolver)
    solver.run(n_macroreps=1)

    criterion_history = criterion.keywords["history_container"]
    criterion_history = list_of_dicts_to_dict_of_lists(criterion_history)
    arg_min = np.argmin(criterion_history["criterion"])

    out = {
        "solution_x": criterion_history["params"][arg_min],
        "solution_criterion": criterion_history["criterion"][arg_min],
        "n_criterion_evaluations": len(criterion_history["criterion"]),
    }
    if derivative is not None:
        derivative_history = derivative.keywords["history_container"]
        derivative_history = list_of_dicts_to_dict_of_lists(derivative_history)
        out["solution_derivative"] = derivative_history["criterion"][arg_min]
        out["n_derivative_evaluations"] = len(derivative_history["criterion"])

    return out


def _do_nothing(self):
    pass


# ======================================================================================
# SimOpt Classes
# ======================================================================================


if IS_SIMOPT_INSTALLED:
    from simopt.base import Model
    from simopt.base import Problem
    from simopt.experiment_base import ProblemSolver

    class ProblemSpecification(Problem):
        def __init__(
            self,
            criterion,
            derivative,
            x,
            lower_bounds,
            upper_bounds,
            budget,
            fixed_factors=None,
        ):
            fixed_factors = {} if fixed_factors is None else fixed_factors
            self.name = "ProblemSpecification"
            self.dim = len(x)
            self.n_objectives = 1
            self.n_stochastic_constraints = 0
            self.minmax = (-1,)  # minimize objective
            self.lower_bounds = lower_bounds
            self.upper_bounds = upper_bounds
            self.gradient_available = derivative is not None
            self.model_default_factors = {}
            self.specifications = {
                "initial_solution": {"default": x},
                "budget": {"default": budget},
            }
            super().__init__(fixed_factors, model_fixed_factors={})
            self.model = CriterionWrapper(
                criterion=criterion,
                gradient=derivative,
                x=x,
            )

        def vector_to_factor_dict(self, x):
            return {"x": x}

        def factor_dict_to_vector(self, d):
            return d["x"]

        def response_dict_to_objectives(self, d):
            return (d["value"],)

    class CriterionWrapper(Model):
        def __init__(self, criterion, gradient, x):
            self.n_rngs = 0
            self.n_responses = 1
            self.specifications = {"x": {"default": x}}
            self.criterion = criterion
            self.gradient_available = gradient is not None
            self.gradient = gradient
            super().__init__(fixed_factors={})

        def replicate(self, rng_list):
            x = np.array(self.factors["x"])
            criterion = {"value": self.criterion(x)}
            if self.gradient_available:
                gradient = {"value": {"x": self.gradient(x)}}
            else:
                gradient = None
            return criterion, gradient
