import numpy as np
from estimagic.optimization.tranquilo.adjust_radius import adjust_radius
from estimagic.optimization.tranquilo.aggregate_models import get_aggregator
from estimagic.optimization.tranquilo.fit_models import get_fitter
from estimagic.optimization.tranquilo.models import ModelInfo
from estimagic.optimization.tranquilo.options import Bounds
from estimagic.optimization.tranquilo.options import RadiusOptions
from estimagic.optimization.tranquilo.options import TrustRegion
from estimagic.optimization.tranquilo.sample_points import get_sampler
from estimagic.optimization.tranquilo.solve_subproblem import get_subsolver
from estimagic.optimization.tranquilo.tranquilo_history import History


def _tranquilo(
    criterion,
    x,
    functype,
    lower_bounds=None,
    upper_bounds=None,
):
    # ==================================================================================
    # hardcoded stuff that needs to be made flexible
    # ==================================================================================
    maxiter = 15
    functype = "scalar"

    sampler = "naive"
    sampler_options = {}
    target_sample_size = 10 * len(x) ** 2

    radius_options = RadiusOptions()

    fitter = "ols"
    fit_options = {}

    model_info = ModelInfo()
    subsolver = "bntr"
    solver_options = {}

    aggregator = "identity"

    # ==================================================================================

    history = History(functype=functype)
    bounds = Bounds(lower=lower_bounds, upper=upper_bounds)
    trustregion = TrustRegion(center=x, radius=radius_options.initial_radius)

    sample_points = get_sampler(sampler, bounds=bounds, user_options=sampler_options)

    aggregate_vector_model = get_aggregator(
        aggregator=aggregator,
        functype=functype,
        model_info=model_info,
    )

    fit_model = get_fitter(
        fitter=fitter,
        user_options=fit_options,
        model_info=model_info,
    )

    solve_subproblem = get_subsolver(
        solver=subsolver,
        user_options=solver_options,
        bounds=bounds,
    )

    first_eval = criterion(x)
    history.add_entries(x, first_eval)
    accepted_x, accepted_fvec, accepted_fval = history.get_entries(0)

    for _ in range(maxiter):
        # update some quantities after acceptance
        fvec_center = accepted_fvec

        old_indices = history.get_indices_in_trustregion(trustregion)
        old_xs = history.get_xs(old_indices)
        old_fvals = history.get_fvals(old_indices)

        new_xs, _ = sample_points(
            trustregion=trustregion,
            target_size=target_sample_size,
            existing_xs=old_xs,
            existing_fvals=old_fvals,
        )

        new_fvecs = [criterion(_x) for _x in new_xs]

        history.add_entries(new_xs, new_fvecs)

        # these could be calculated more quickly without searching entire history!
        model_indices = history.get_indices_in_trustregion(trustregion)

        model_xs = history.get_xs(model_indices)
        model_fvecs = history.get_fvecs(model_indices)

        centered_xs = (model_xs - trustregion.center) / trustregion.radius

        vector_model = fit_model(centered_xs, model_fvecs)

        scalar_model = aggregate_vector_model(
            vector_model=vector_model,
            fvec_center=fvec_center,
        )

        sub_sol = solve_subproblem(model=scalar_model, trustregion=trustregion)

        candidate_x = sub_sol["x"]

        candidate_fvec = criterion(candidate_x)
        history.add_entries(candidate_x, candidate_fvec)
        candidate_fval = history.get_fvals(-1)
        actual_improvement = -(candidate_fval - accepted_fval)

        rho = _calculate_rho(
            actual_improvement=actual_improvement,
            expected_improvement=sub_sol["expected_improvement"],
        )

        if actual_improvement > 0:
            new_radius = adjust_radius(
                radius=trustregion.radius,
                rho=rho,
                step=candidate_x - accepted_x,
                options=radius_options,
            )
            accepted_x = candidate_x
            trustregion = trustregion._replace(center=candidate_x, radius=new_radius)

    res = {
        "solution_x": accepted_x,
    }

    return res


def _calculate_rho(actual_improvement, expected_improvement):
    if expected_improvement == 0:
        rho = np.inf * np.sign(actual_improvement)
    else:
        rho = actual_improvement / expected_improvement

    return rho
