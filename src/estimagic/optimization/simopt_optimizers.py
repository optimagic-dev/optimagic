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
    is_global=True,
)
def simopt_adam(
    criterion,
    derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    stopping_max_iterations=STOPPING_MAX_CRITERION_EVALUATIONS_GLOBAL,
):
    out = _minimize_simopt(
        name="ADAM",
        criterion=criterion,
        derivative=derivative,
        x=x,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
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
):
    out = _minimize_simopt(
        name="ALOE",
        criterion=criterion,
        derivative=derivative,
        x=x,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        budget=stopping_max_iterations,
    )
    return out


@mark_minimizer(
    name="simopt_astrodf",
    primary_criterion_entry="value",
    needs_scaling=True,
    is_available=IS_SIMOPT_INSTALLED,
    is_global=True,
)
def simopt_astrodf(
    criterion,
    derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    stopping_max_iterations=STOPPING_MAX_CRITERION_EVALUATIONS_GLOBAL,
):
    out = _minimize_simopt(
        name="ASTRODF",
        criterion=criterion,
        derivative=derivative,
        x=x,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
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
    derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    stopping_max_iterations=STOPPING_MAX_CRITERION_EVALUATIONS_GLOBAL,
):
    out = _minimize_simopt(
        name="NELDMD",
        criterion=criterion,
        derivative=derivative,
        x=x,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        budget=stopping_max_iterations,
    )
    return out


@mark_minimizer(
    name="simopt_spsa",
    primary_criterion_entry="value",
    needs_scaling=True,
    is_available=IS_SIMOPT_INSTALLED,
    is_global=True,
)
def simopt_spsa(
    criterion,
    derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    stopping_max_iterations=STOPPING_MAX_CRITERION_EVALUATIONS_GLOBAL,
):
    out = _minimize_simopt(
        name="SPSA",
        criterion=criterion,
        derivative=derivative,
        x=x,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        budget=stopping_max_iterations,
    )
    return out


@mark_minimizer(
    name="simopt_strong",
    primary_criterion_entry="value",
    needs_scaling=True,
    is_available=IS_SIMOPT_INSTALLED,
    is_global=True,
)
def simopt_strong(
    criterion,
    derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    budget=None,
):
    out = _minimize_simopt(
        name="STRONG",
        criterion=criterion,
        derivative=derivative,
        x=x,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        budget=budget,
    )
    return out


# ======================================================================================
# Template
# ======================================================================================


def _minimize_simopt(
    name,
    criterion,
    derivative,
    x,
    lower_bounds,
    upper_bounds,
    *,
    budget=None,
):
    budget = 1_000 if budget is None else budget

    problem = MinimzerClass(
        criterion=criterion,
        gradient=derivative,
        x=x,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        budget=budget,
    )
    solver = ProblemSolver(solver_name=name, problem=problem)
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
    def __init__(self, criterion, gradient, x, fixed_factors=None):
        fixed_factors = {} if fixed_factors is None else fixed_factors
        self.n_rngs = 0
        self.n_responses = 1
        self.specifications = {"x": {"default": x}}
        self.criterion = criterion
        self.gradient = gradient
        super().__init__(fixed_factors)

    def replicate(self, rng_list):
        x = np.array(self.factors["x"])
        responses = {"value": self.criterion(x)}
        gradients = {"value": {"x": self.gradient(x)}}
        return responses, gradients


class MinimzerClass(Problem):
    def __init__(
        self,
        criterion,
        gradient,
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
        self.minmax = (-1,)
        self.constraint_type = "box"
        self.variable_type = "continuous"
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.gradient_available = True
        self.model_default_factors = {}
        self.model_decision_factors = {"x"}
        self.factors = fixed_factors
        self.specifications = {
            "initial_solution": {"default": x},
            "budget": {"default": budget},
        }
        self.check_factor_list = {
            "initial_solution": self.check_initial_solution,
            "budget": self.check_budget,
        }
        super().__init__(fixed_factors, model_fixed_factors)
        self.model = CriterionClass(
            criterion=criterion,
            gradient=gradient,
            x=x,
            fixed_factors=self.model_fixed_factors,
        )

    def vector_to_factor_dict(self, vector):
        factor_dict = {"x": vector[:]}
        return factor_dict

    def factor_dict_to_vector(self, factor_dict):
        vector = factor_dict["x"]
        return vector

    def response_dict_to_objectives(self, response_dict):
        objectives = (response_dict["value"],)
        return objectives
