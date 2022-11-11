"""Implement `simopt` optimizers."""
import numpy as np
from estimagic.config import IS_SIMOPT_INSTALLED
from estimagic.decorators import mark_minimizer
from estimagic.optimization.algo_options import (
    STOPPING_MAX_CRITERION_EVALUATIONS_GLOBAL,
)

try:
    from simopt.base import Model
    from simopt.base import Problem
    from simopt.experiment_base import ProblemSolver
except ImportError:
    pass


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

    Algorithm Options
    -----------------

    - crn_across_solns (bool): Use CRN across solutions? Default True.
    - r (int): Number of replications taken at each solution. Default 1.
    - beta_1 (float): Exponential decay of the rate for the first moment estimates.
    Default 0.9.
    - beta_2 (float): Exponential decay rate for the second-moment estimates. Default
    0.999.
    - alpha (float): Step size. Default 1.0.
    - epsilon (float): A small value to prevent zero-division. Default 10e-8.
    - sensitivity (float): Shrinking scale for variable bounds. Default 10e-7.

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
    name="simopt_aloe",
    primary_criterion_entry="value",
    needs_scaling=True,
    is_available=IS_SIMOPT_INSTALLED,
    is_global=True,
)
def simopt_aloe(
    criterion,
    derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    stopping_max_iterations=STOPPING_MAX_CRITERION_EVALUATIONS_GLOBAL,
    crn_across_solns=True,
    r=30,
    theta=0.2,
    gamma=0.8,
    alpha_max=10,
    alpha_0=1.0,
    epsilon_f=1.0,
    sensitivity=10e-7,
    _lambda=2.0,
):
    """Minimize a scalar function using the ALOE algorithm from SimOpt.

    Algorithm Options
    -----------------
    - crn_across_solns (bool): Use CRN across solutions? Default True.
    - r (int): Number of replications taken at each solution. Default 30.
    - theta (float): Constant in the Armijo condition. Default 0.2.
    - gamma (float): Constant for shrinking the step size. Default 0.8.
    - alpha_max (int): Maximum step size. Default 10.
    - alpha_0 (float): Initial step size. Default 1.0.
    - epsilon_f (float): Additive constant in the Armijo condition. Default 1.0.
    - sensitivity (float): Shrinking scale for variable bounds. Default 1e-7.
    - _lambda (float): Magnifying factor for n_r inside the finite difference function.
    Default 2.0.

    """
    solver_options = {
        "crn_across_solns": crn_across_solns,
        "r": r,
        "theta": theta,
        "gamma": alpha_max,
        "alpha_max": alpha_0,
        "alpha_0": epsilon_f,
        "epsilon_f": gamma,
        "sensitivity": sensitivity,
        "lambda": _lambda,
    }

    out = _minimize_simopt(
        algorithm="ALOE",
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

    Algorithm Options
    -----------------

    - crn_across_solns (bool): Use CRN across solutions? Default True.
    - delta_max (float): Maximum value of the trust-region radius. Default 50.0
    - eta_1 (float): Threshhold for a successful iteration. Default 0.1.
    - eta_2 (float): Threshhold for a very successful iteration. Default 0.5.
    - gamma_1 (float): Very successful step trust-region radius increase. Default 2.0.
    - gamma_2 (float): Unsuccessful step trust-region radius decrease. Default 0.5.
    - w (float): trust-region radius rate of shrinkage in contracation loop. Default
    0.85.
    - mu (int): trust-region radius ratio upper bound in contraction loop. Default 1000.
    - beta (int): trust-region radius ratio lower bound in contraction loop. Default 10.
    - lambda_min (int): minimum sample size value. Default 8.
    - simple_solve (bool): Solve subproblem with Cauchy point (rough approximate)?
    Default False.
    - criticality_select (bool): Skip contraction loop if not near critical
    region? Default True.
    - criticality_threshold (float): Threshold on gradient norm indicating
    near-critical region. Default 0.1.

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
    name="simopt_neldmd",
    primary_criterion_entry="value",
    needs_scaling=True,
    is_available=IS_SIMOPT_INSTALLED,
    is_global=True,
)
def simopt_neldmd(
    criterion,
    x,
    lower_bounds,
    upper_bounds,
    *,
    stopping_max_iterations=STOPPING_MAX_CRITERION_EVALUATIONS_GLOBAL,
    crn_across_solns=True,
    r=30,
    alpha=1.0,
    gammap=0.5,
    betap=0.5,
    delta=0.5,
    sensitivity=10e-7,
    initial_spread=0.1,
):
    """Minimize a scalar function using the Nelder-Mead algorithm from SimOpt.

    Algorithm Options
    -----------------

    - crn_across_solns (bool): Use CRN across solutions? Default True.
    - r (int): Number of replications taken at each solution. Default 30.
    - alpha (float): Reflection coefficient > 0. Default 1.0.
    - gammap (float): Expansion coefficient > 1. Default 2.0.
    - betap (float): Contraction coefficient > 0, < 1. Default 0.5.
    - delta (float): Shrink factor > 0, < 1. Default 0.5.
    - sensitivity (float): Shrinking scale for bounds. Default 10e-7.
    - initial_spread (float): Fraction of the distance between bounds used to select
    initial points. Default 1 / 10.

    """
    solver_options = {
        "crn_across_solns": crn_across_solns,
        "r": r,
        "alpha": alpha,
        "gammap": gammap,
        "betap": betap,
        "delta": delta,
        "sensitivity": sensitivity,
        "initial_spread": initial_spread,
    }

    out = _minimize_simopt(
        algorithm="NELDMD",
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
    lower_bounds,
    upper_bounds,
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

    Algorithm Options
    -----------------

    - crn_across_solns (bool): Use CRN across solutions? Default True.
    - alpha (float): Non-negative coefficient in the SPSA gain sequecence ak. Default
    0.602.
    - gamma (float): Non-negative coefficient in the SPSA gain sequence ck. Default
    0.101.
    - step (float): Initial desired magnitude of change in the theta elements. Default
    0.5.
    - gavg (int): Averaged SP gradients used per iteration. Default 1.
    - n_reps (int): Number of replications takes at each solution. Default 2.
    - n_loss (int): Number of loss function evaluations used in this gain calculation.
    Default 2.
    - eval_pct (float): Percentage of the expected number of loss evaluations per run.
    Default 2/3.
    - iter_pct (float): Percentage of the maximum expected number of iterations. Default
    0.1.

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
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
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
    lower_bounds,
    upper_bounds,
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

    Algorithm Options
    -----------------

    - crn_across_solns (bool): Use CRN across solutions? Default True.
    - n0 (int): Initial sample size Default 10.
    - n_r (int): Number of replications taken at each solution. Default 1.
    - sensitivity (float): Shrinking scale for VarBds. Default 10e-7.
    - delta_threshold (float): Maximum value of the radius. Default 1.2.
    - delta_T (float): Initial size of trust region. Default 2.0.
    - eta_0 (float): Constant for accepting. Default 0.01.
    - eta_1 (float): Constant for more confident accepting. Default 0.3.
    - gamma_1 (float): Constant for shrinking the trust region. Default 0.9.
    - gamma_2 (float): Constant for expanding the trust region. Default 1.11.
    - lambda (int): Magnifying factor for n_r inside the finite difference function.
    Default 2.
    - lambda_2 (float): Magnifying factor for n_r in stage I and stage II. Default 1.01.

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
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
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
    gradient_info = {
        "gradient_available": derivative is not None,
        "gradient": derivative,
    }

    problem = MinimzerClass(
        criterion=criterion,
        gradient_info=gradient_info,
        x=x,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        budget=budget,
    )
    solver = ProblemSolver(
        solver_name=algorithm, problem=problem, solver_fixed_factors=solver_options
    )
    # overwrite method of simopt ProblemSolver class that pickles interim results
    solver.record_experiment_results = _do_nothing.__get__(solver, ProblemSolver)
    solver.run(n_macroreps=1)

    processed = {
        "solution_x": np.array(solver.all_recommended_xs[0][-1]),
        "solution_criterion": 0,
    }
    return processed


def _do_nothing(self):
    pass


# ======================================================================================
# SimOpt Classes
# ======================================================================================


class CriterionClass(Model):
    def __init__(self, criterion, gradient_info, x, fixed_factors=None):
        fixed_factors = {} if fixed_factors is None else fixed_factors
        self.n_rngs = 0
        self.n_responses = 1
        self.specifications = {"x": {"default": x}}
        self.criterion = criterion
        self.gradient_info = gradient_info
        super().__init__(fixed_factors)

    def replicate(self, rng_list):
        x = np.array(self.factors["x"])
        responses = {"value": self.criterion(x)}
        if self.gradient_info["gradient_available"]:
            gradients = {"value": {"x": self.gradient_info["gradient"](x)}}
        else:
            gradients = {"value": {"x": np.nan}}
        return responses, gradients


class MinimzerClass(Problem):
    def __init__(
        self,
        criterion,
        gradient_info,
        x,
        lower_bounds,
        upper_bounds,
        budget,
        fixed_factors=None,
        model_fixed_factors=None,
    ):
        fixed_factors = {} if fixed_factors is None else fixed_factors
        model_fixed_factors = {} if model_fixed_factors is None else model_fixed_factors
        self.name = "MinimizerClass"
        self.dim = len(x)
        self.n_objectives = 1
        self.n_stochastic_constraints = 0
        self.minmax = (-1,)  # minimize objective
        self.constraint_type = "box"
        self.variable_type = "continuous"
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.gradient_available = gradient_info["gradient_available"]
        self.model_default_factors = {}
        self.model_decision_factors = {"x"}
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {"default": x},
            "budget": {"default": budget},
        }
        super().__init__(fixed_factors, model_fixed_factors)
        self.model = CriterionClass(
            criterion=criterion,
            gradient_info=gradient_info,
            x=x,
            fixed_factors=self.model_fixed_factors,
        )

    def vector_to_factor_dict(self, vector):
        factor_dict = {"x": vector}
        return factor_dict

    def factor_dict_to_vector(self, factor_dict):
        vector = factor_dict["x"]
        return vector

    def response_dict_to_objectives(self, response_dict):
        objectives = (response_dict["value"],)
        return objectives
