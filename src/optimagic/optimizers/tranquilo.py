from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal

import numpy as np
from numpy.typing import NDArray
from packaging import version

from optimagic import mark
from optimagic.config import IS_TRANQUILO_INSTALLED
from optimagic.exceptions import NotInstalledError
from optimagic.optimization.algorithm import Algorithm, InternalOptimizeResult
from optimagic.optimization.internal_optimization_problem import (
    InternalOptimizationProblem,
)
from optimagic.typing import (
    AggregationLevel,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)

if TYPE_CHECKING:
    from tranquilo.options import (
        AcceptanceOptions,
        FilterOptions,
        FitterOptions,
        NoiseAdaptationOptions,
        RadiusOptions,
        SamplerOptions,
        StagnationOptions,
        SubsolverOptions,
        VarianceEstimatorOptions,
    )

if IS_TRANQUILO_INSTALLED:
    import tranquilo

    IS_TRANQUILO_VERSION_NEWER_OR_EQUAL_TO_0_1_0 = version.parse(
        tranquilo.__version__
    ) >= version.parse("0.1.0")
else:
    IS_TRANQUILO_VERSION_NEWER_OR_EQUAL_TO_0_1_0 = False

TRANQUILO_INSTALLATION_INSTRUCTIONS = (
    "The 'tranquilo' algorithm requires the tranquilo package version 0.1.0 or newer "
    "to be installed. Install it with 'conda -c conda-forge install tranquilo>=0.1.0'."
)


@mark.minimizer(
    name="tranquilo",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_TRANQUILO_INSTALLED,
    is_global=False,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=False,
    supports_parallelism=True,
    supports_bounds=True,
    supports_infinite_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class Tranquilo(Algorithm):
    """Minimize a scalar function using the tranquilo algorithm.

    Tranquilo, short for "TrustRegion Adaptive Noise robust QuadratIc or Linear
    approximation Optimizer", is a trust-region optimizer for derivative-free
    optimization that was developed by the optimagic developers
    (:cite:`Gabler2024`). It is designed for black-box problems whose objective
    function is computationally expensive, noisy, or both, as they arise, for
    example, when structural econometric models are estimated via the method of
    simulated moments.

    This is the scalar version of tranquilo. If your objective function has a
    least-squares structure, use ``tranquilo_ls`` instead, which exploits that
    structure and typically needs far fewer function evaluations.

    tranquilo is a local optimizer that does not require derivatives. It supports
    bound constraints, but no linear or nonlinear constraints. In each iteration
    it fits a quadratic surrogate model to a sample of evaluated points, minimizes
    the surrogate model within the current trust region to obtain a candidate
    point, and accepts or rejects the candidate by comparing the actual with the
    predicted improvement. For background on derivative-free trust-region methods,
    see :cite:`Conn2009`.

    Two features distinguish tranquilo from most other derivative-free
    trust-region optimizers:

    - Parallelization: If ``n_cores`` is larger than one, the objective function
      is evaluated in parallel batches. A parallel line search and speculative
      sampling exploit all available cores even in steps of the algorithm that are
      inherently sequential. tranquilo thereby minimizes the number of evaluation
      batches rather than the number of function evaluations.
    - Noise handling: If ``noisy=True``, tranquilo adaptively determines how often
      the objective function must be evaluated (and averaged) at each point in
      order to keep making progress. The user does not have to provide any
      information about the amount or type of noise in the objective function.

    All components of the algorithm (sampling of new points, filtering of existing
    points, fitting of surrogate models, solving of the trust-region subproblem,
    and the acceptance decision) are exchangeable and configurable via options.
    The default values were selected based on extensive benchmarking and rarely
    need to be adjusted. In benchmarks on noise-free problems without
    parallelization, the scalar version of tranquilo is competitive with, but
    slightly slower than, established derivative-free optimizers such as
    ``nlopt_bobyqa``. Its strengths come into play for noisy objective functions
    and parallel function evaluation.

    .. note::
       This algorithm requires the tranquilo package (version 0.1.0 or newer) to
       be installed. You can install it with ``pip install tranquilo`` or
       ``conda install -c conda-forge tranquilo``.

    """

    # function type
    functype: Literal["scalar"] = "scalar"
    """The type of the objective function.

    For this algorithm it is always "scalar". Use ``tranquilo_ls`` for problems with
    least-squares structure.

    """

    # basic options
    noisy: bool = False
    """Whether the objective function is noisy, i.e. whether repeated evaluations at
    the same parameter vector return different values.

    If True, tranquilo activates its adaptive noise handling: the objective function
    is evaluated multiple times at each sample point, and the number of evaluations
    per point is adjusted over the course of the optimization such that just enough
    noise is averaged out to keep making progress. Moreover, the acceptance decision
    for candidate points is then based on a statistical power analysis, and the
    number of evaluations at the start parameters defaults to 5.

    """

    # convergence options
    disable_convergence: bool = False
    """If True, the optimization is terminated only by the stopping criteria
    (``stopping_maxfun``, ``stopping_maxiter``, ``stopping_maxtime``), never because
    a convergence criterion is satisfied.

    This is mostly useful for benchmarking.

    """

    convergence_ftol_abs: NonNegativeFloat = 0.0
    r"""Converge if the absolute change in the objective function value between two
    accepted iterations is less than this value. More formally, this is expressed as

    .. math::

        |f^k - f^{k+1}| \leq \textsf{convergence_ftol_abs}.

    This criterion is disabled by default (0.0).

    """

    convergence_gtol_abs: NonNegativeFloat = 0.0
    """Converge if the Euclidean norm of the gradient of the surrogate model at the
    current trust-region center is less than this value.

    Since tranquilo never evaluates derivatives of the objective function, the
    gradient of the fitted surrogate model is used instead of the true gradient.
    This criterion is disabled by default (0.0).

    """

    convergence_xtol_abs: NonNegativeFloat = 0.0
    r"""Converge if the Euclidean distance between the accepted parameter vectors of
    two consecutive iterations is less than this value. More formally, this is
    expressed as

    .. math::

        \lVert x^k - x^{k+1} \rVert \leq \textsf{convergence_xtol_abs}.

    This criterion is disabled by default (0.0).

    """

    convergence_ftol_rel: NonNegativeFloat = 2e-9
    r"""Converge if the relative change in the objective function value between two
    accepted iterations is less than this value. More formally, this is expressed as

    .. math::

        \frac{|f^k - f^{k+1}|}{\max\{|f^k|, 1\}} \leq \textsf{convergence_ftol_rel}.

    """

    convergence_gtol_rel: NonNegativeFloat = 1e-8
    r"""Converge if the Euclidean norm of the gradient of the surrogate model at the
    current trust-region center, divided by its maximum with 1, is less than this
    value. Denoting the gradient of the surrogate model by :math:`g^k`, this is
    expressed as

    .. math::

        \frac{\lVert g^k \rVert}{\max\{\lVert g^k \rVert, 1\}} \leq
        \textsf{convergence_gtol_rel}.

    """

    convergence_xtol_rel: NonNegativeFloat = 1e-8
    r"""Converge if the relative change of the accepted parameter vectors between two
    consecutive iterations is less than this value. More formally, this is expressed
    as

    .. math::

        \left\lVert \frac{x^k - x^{k+1}}{\max\{|x^k|, 1\}} \right\rVert \leq
        \textsf{convergence_xtol_rel},

    where the division and the maximum are taken element-wise.

    """

    convergence_min_trust_region_radius: NonNegativeFloat = 0.0
    """Consider the optimization converged if the trust-region radius shrinks below
    this value.

    This criterion is disabled by default (0.0).

    """

    # stopping options
    stopping_maxfun: PositiveInt = 2_000
    """Maximum number of objective function evaluations before termination.

    Repeated evaluations at the same point (as used for noisy problems) count
    towards this limit.

    """

    stopping_maxiter: PositiveInt = 200
    """Maximum number of iterations (trust-region steps) before termination."""

    stopping_maxtime: NonNegativeFloat = np.inf
    """Maximum running time in seconds before termination."""

    # single advanced options
    batch_evaluator: Literal[
        "joblib",
        "pathos",
    ] = "joblib"
    """Batch evaluator that is used to parallelize evaluations of the objective
    function.

    See :ref:`batch_evaluators` for details.

    """

    n_cores: PositiveInt = 1
    """Number of cores used to evaluate the objective function in parallel.

    If larger than one, tranquilo evaluates batches of points in parallel and uses a
    parallel line search as well as speculative sampling to keep all cores busy.

    """

    batch_size: PositiveInt | None = None
    """Number of points that are evaluated in each batch of parallel function
    evaluations.

    Must be at least as large as ``n_cores``, to which it defaults.

    """

    sample_size: PositiveInt | None = None
    """Target number of evaluated points in the trust region on which the surrogate
    model is fitted.

    By default, the sample size is determined from the model type and the dimension
    :math:`n` of the problem: :math:`2 n + 1` for quadratic models, and :math:`n +
    1` (in the noisy or parallel case) or :math:`n + 2` (otherwise) for linear
    models.

    """

    model_type: (
        Literal[
            "quadratic",
            "linear",
        ]
        | None
    ) = None
    """Type of surrogate model that is fitted in each iteration.

    Defaults to "quadratic" for the scalar version of tranquilo. Linear models
    require fewer function evaluations per iteration but contain no curvature
    information.

    """

    search_radius_factor: PositiveFloat | None = None
    """Factor by which the trust-region radius is multiplied to obtain the region in
    which existing evaluation points are searched for reuse in the surrogate model.

    Reusing existing points saves function evaluations. The default is 4.25 for the
    scalar version and 5.0 for the least-squares version of tranquilo.

    """

    n_evals_per_point: NonNegativeInt | None = None
    """Initial number of times the objective function is evaluated (and averaged) at
    each sample point.

    This is only relevant for noisy problems, where the default is derived from
    ``noise_adaptation_options`` and the number of evaluations per point is
    adaptively adjusted over the course of the optimization. For non-noisy problems,
    the default is 1.

    """

    n_evals_at_start: NonNegativeInt | None = None
    """Number of objective function evaluations at the start parameters.

    The default is 5 for noisy problems and 1 otherwise. Multiple evaluations at the
    start parameters guarantee that enough evaluations are available to estimate the
    variance of the noise in the objective function.

    """

    seed: int | None = 925408
    """Seed for the random number generators used to sample new points and to
    simulate noise during the noise adaptation.

    Pass None for non-reproducible randomness.

    """

    # bundled advanced options
    radius_options: RadiusOptions | None = None
    """Advanced options for the management of the trust-region radius, e.g. the
    initial, minimal and maximal radius as well as expansion and shrinking factors.

    Pass an instance of ``tranquilo.options.RadiusOptions``; see the tranquilo
    package for details.

    """

    stagnation_options: StagnationOptions | None = None
    """Advanced options that determine how the algorithm reacts if it stagnates,
    i.e. if a proposed step is very small relative to the trust-region radius.

    Pass an instance of ``tranquilo.options.StagnationOptions``; see the tranquilo
    package for details.

    """

    noise_adaptation_options: NoiseAdaptationOptions | None = None
    """Advanced options that govern how the number of evaluations per point is
    adapted for noisy problems, e.g. the minimal and maximal number of evaluations
    per point.

    Pass an instance of ``tranquilo.options.NoiseAdaptationOptions``; see the
    tranquilo package for details.

    """

    # component names and related options
    sampler: (
        Literal[
            "optimal_hull",
            "random_hull",
            "random_interior",
        ]
        | Callable
    ) = "optimal_hull"
    """Method used to sample new points within the trust region.

    "optimal_hull" (the default) places new points on the hull of the trust region
    such that the minimal distance between all points (including existing ones) is
    approximately maximized. "random_hull" and "random_interior" sample random
    points on the hull or in the interior of the trust region. A custom sampling
    function can be passed as a callable.

    """

    sampler_options: SamplerOptions | None = None
    """Advanced options for the sampler.

    Pass an instance of ``tranquilo.options.SamplerOptions``; see the tranquilo
    package for details.

    """

    sample_filter: (
        Literal[
            "discard_all",
            "keep_all",
            "clustering",
            "drop_excess",
        ]
        | Callable
        | None
    ) = None
    """Method used to select which of the existing evaluation points inside the
    search region are used to fit the surrogate model.

    "keep_all" keeps all points, "discard_all" keeps only the trust-region center,
    "drop_excess" drops points if there are more than needed to fit the surrogate
    model, preferring points that are spread out within the trust region, and
    "clustering" keeps only one point per cluster of nearby points. The default is
    "drop_excess" if ``batch_size`` is larger than one and "keep_all" otherwise. A
    custom filter function can be passed as a callable.

    """

    sample_filter_options: FilterOptions | None = None
    """Advanced options for the sample filter.

    Pass an instance of ``tranquilo.options.FilterOptions``; see the tranquilo
    package for details.

    """

    model_fitter: (
        Literal[
            "ols",
            "ridge",
            "powell",
            "tranquilo",
        ]
        | Callable
        | None
    ) = None
    """Method used to fit the surrogate model to the sample of evaluated points.

    "ols" uses ordinary least-squares regression, "ridge" uses ridge regression,
    "powell" switches between ordinary least squares and a minimal-Frobenius-norm-
    of-Hessian fit depending on the sample size, following ideas from
    :cite:`Powell2004`, and "tranquilo" is a variant of "ols" that penalizes the
    linear terms of the model less strongly when the fitting problem is
    underdetermined. By default, "ols" is used if the sample size is large enough to
    identify all model parameters and "tranquilo" otherwise. A custom fitting
    function can be passed as a callable.

    """

    model_fitter_options: FitterOptions | None = None
    """Advanced options for the model fitter, e.g. penalty terms for ridge
    regression.

    Pass an instance of ``tranquilo.options.FitterOptions``; see the tranquilo
    package for details.

    """

    cube_subsolver: (
        Literal[
            "bntr",
            "bntr_fast",
            "fallback_cube",
            "fallback_multistart",
        ]
        | Callable
    ) = "bntr_fast"
    """Solver for the trust-region subproblem if the trust region is a cube, which
    is the case whenever bounds are binding.

    "bntr" and "bntr_fast" implement a bounded Newton trust-region algorithm, where
    "bntr_fast" (the default) is a numba-accelerated version of "bntr".
    "fallback_cube" and "fallback_multistart" are robust fallback solvers based on
    SciPy's L-BFGS-B, where the multistart version restarts the solver from multiple
    starting points. A custom subsolver can be passed as a callable.

    """

    sphere_subsolver: (
        Literal[
            "gqtpar",
            "gqtpar_fast",
            "fallback_reparametrized",
            "fallback_inscribed_cube",
            "fallback_norm_constraint",
        ]
        | Callable
    ) = "gqtpar_fast"
    """Solver for the trust-region subproblem if the trust region is a sphere, which
    is the case when no bounds are binding.

    "gqtpar" and "gqtpar_fast" solve the subproblem almost exactly with the method
    of :cite:`More1983`, where "gqtpar_fast" (the default) is a numba-accelerated
    version of "gqtpar". The "fallback_*" solvers are robust but less precise
    alternatives based on general-purpose optimizers. A custom subsolver can be
    passed as a callable.

    """

    retry_subproblem_with_fallback: bool = True
    """Whether to retry solving the trust-region subproblem with a fallback solver
    if the main subsolver raises an exception."""

    subsolver_options: SubsolverOptions | None = None
    """Advanced options for the trust-region subproblem solvers, e.g. the maximum
    number of iterations and gradient tolerances.

    Pass an instance of ``tranquilo.options.SubsolverOptions``; see the tranquilo
    package for details.

    """

    acceptance_decider: (
        Literal[
            "classic",
            "naive_noisy",
            "classic_line_search",
            "noisy",
        ]
        | Callable
        | None
    ) = None
    """Method used to decide whether a candidate point is accepted as the new
    trust-region center.

    "classic" performs the standard acceptance step of trust-region algorithms,
    which compares the actual with the predicted improvement.
    "classic_line_search" additionally evaluates speculative points along the
    search direction in parallel. "naive_noisy" averages a fixed number of
    evaluations at the candidate point. "noisy" uses a statistical power analysis
    to determine how many evaluations at the candidate and the current point are
    needed to decide which one is better. The default is "noisy" for noisy problems
    and "classic" otherwise. A custom acceptance function can be passed as a
    callable.

    """

    acceptance_decider_options: AcceptanceOptions | None = None
    """Advanced options for the acceptance decider, e.g. the confidence and power
    levels of the power analysis for noisy problems.

    Pass an instance of ``tranquilo.options.AcceptanceOptions``; see the tranquilo
    package for details.

    """

    variance_estimator: Literal["classic"] | Callable = "classic"
    """Method used to estimate the variance of the noise in the objective function
    for noisy problems.

    The "classic" estimator uses existing repeated function evaluations at points in
    a neighborhood of the current trust region and treats the noise variance as
    locally constant. A custom estimation function can be passed as a callable.

    """

    variance_estimator_options: VarianceEstimatorOptions | None = None
    """Advanced options for the variance estimator, e.g. the minimal number of
    evaluations per point used in the estimation.

    Pass an instance of ``tranquilo.options.VarianceEstimatorOptions``; see the
    tranquilo package for details.

    """

    infinity_handler: Literal["relative"] | Callable = "relative"
    """Method used to clip infinite objective function values before fitting the
    surrogate model.

    The "relative" method clips infinite values at a penalty value that is derived
    from the range of the finite values in the sample. A custom function can be
    passed as a callable.

    """

    residualize: bool | None = None
    """Whether the surrogate model is fitted to the deviations of the function
    values from the predictions of the previous surrogate model instead of the
    function values themselves.

    Defaults to True if the "tranquilo" model fitter is used and False otherwise.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_TRANQUILO_VERSION_NEWER_OR_EQUAL_TO_0_1_0:
            raise NotInstalledError(TRANQUILO_INSTALLATION_INSTRUCTIONS)
        from tranquilo.tranquilo import _tranquilo

        raw_res = _tranquilo(
            functype="scalar",
            batch_fun=problem.batch_fun,
            x=x0,
            lower_bounds=problem.bounds.lower,
            upper_bounds=problem.bounds.upper,
            noisy=self.noisy,
            disable_convergence=self.disable_convergence,
            convergence_absolute_criterion_tolerance=self.convergence_ftol_abs,
            convergence_absolute_gradient_tolerance=self.convergence_gtol_abs,
            convergence_absolute_params_tolerance=self.convergence_xtol_abs,
            convergence_relative_criterion_tolerance=self.convergence_ftol_rel,
            convergence_relative_gradient_tolerance=self.convergence_gtol_rel,
            convergence_relative_params_tolerance=self.convergence_xtol_rel,
            convergence_min_trust_region_radius=self.convergence_min_trust_region_radius,
            stopping_max_criterion_evaluations=self.stopping_maxfun,
            stopping_max_iterations=self.stopping_maxiter,
            stopping_max_time=self.stopping_maxtime,
            n_cores=self.n_cores,
            batch_size=self.batch_size,
            sample_size=self.sample_size,
            model_type=self.model_type,
            search_radius_factor=self.search_radius_factor,
            n_evals_per_point=self.n_evals_per_point,
            n_evals_at_start=self.n_evals_at_start,
            seed=self.seed,
            radius_options=self.radius_options,
            stagnation_options=self.stagnation_options,
            noise_adaptation_options=self.noise_adaptation_options,
            sampler=self.sampler,
            sampler_options=self.sampler_options,
            sample_filter=self.sample_filter,
            sample_filter_options=self.sample_filter_options,
            model_fitter=self.model_fitter,
            model_fitter_options=self.model_fitter_options,
            cube_subsolver=self.cube_subsolver,
            sphere_subsolver=self.sphere_subsolver,
            retry_subproblem_with_fallback=self.retry_subproblem_with_fallback,
            subsolver_options=self.subsolver_options,
            acceptance_decider=self.acceptance_decider,
            acceptance_decider_options=self.acceptance_decider_options,
            variance_estimator=self.variance_estimator,
            variance_estimator_options=self.variance_estimator_options,
            infinity_handler=self.infinity_handler,
            residualize=self.residualize,
        )

        res = InternalOptimizeResult(
            x=raw_res["solution_x"],
            fun=raw_res["solution_criterion"],
            message=raw_res["message"],
            info={"states": raw_res["states"]},
        )
        return res


@mark.minimizer(
    name="tranquilo_ls",
    solver_type=AggregationLevel.LEAST_SQUARES,
    is_available=IS_TRANQUILO_INSTALLED,
    is_global=False,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=False,
    supports_parallelism=True,
    supports_bounds=True,
    supports_infinite_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class TranquiloLS(Algorithm):
    r"""Minimize a least-squares problem using the tranquilo algorithm.

    Tranquilo, short for "TrustRegion Adaptive Noise robust QuadratIc or Linear
    approximation Optimizer", is a trust-region optimizer for derivative-free
    optimization that was developed by the optimagic developers
    (:cite:`Gabler2024`). It is designed for black-box problems whose objective
    function is computationally expensive, noisy, or both, as they arise, for
    example, when structural econometric models are estimated via the method of
    simulated moments.

    This is the least-squares version of tranquilo. It minimizes objective
    functions of the form :math:`f(x) = \sum_j r_j(x)^2`, where the user provides
    the vector of residuals :math:`r(x)`. Instead of fitting one surrogate model
    for the objective function, it fits a surrogate model for each residual and
    aggregates them into a model of the objective function. Exploiting the
    least-squares structure in this way is very efficient: as few as :math:`n + 1`
    function evaluations (where :math:`n` is the number of parameters) can be
    enough to build a useful surrogate model. Use this version whenever your
    objective function can be expressed as a sum of squares, e.g. in nonlinear
    regression or method of simulated moments estimation; use the scalar version
    (``tranquilo``) otherwise.

    tranquilo_ls is a local optimizer that does not require derivatives. It
    supports bound constraints, but no linear or nonlinear constraints. In each
    iteration it fits surrogate models to a sample of evaluated points, minimizes
    the aggregated surrogate model within the current trust region to obtain a
    candidate point, and accepts or rejects the candidate by comparing the actual
    with the predicted improvement. For background on derivative-free trust-region
    methods, see :cite:`Conn2009`.

    Two features distinguish tranquilo from most other derivative-free
    trust-region optimizers:

    - Parallelization: If ``n_cores`` is larger than one, the objective function
      is evaluated in parallel batches. A parallel line search and speculative
      sampling exploit all available cores even in steps of the algorithm that are
      inherently sequential. tranquilo thereby minimizes the number of evaluation
      batches rather than the number of function evaluations.
    - Noise handling: If ``noisy=True``, tranquilo adaptively determines how often
      the objective function must be evaluated (and averaged) at each point in
      order to keep making progress. The user does not have to provide any
      information about the amount or type of noise in the objective function.

    All components of the algorithm (sampling of new points, filtering of existing
    points, fitting of surrogate models, solving of the trust-region subproblem,
    and the acceptance decision) are exchangeable and configurable via options.
    The default values were selected based on extensive benchmarking and rarely
    need to be adjusted. In benchmarks on noise-free problems without
    parallelization, tranquilo_ls is slightly slower than DFO-LS
    (:cite:`Cartis2018`) and faster than POUNDERS (:cite:`Wild2015`). With two or
    more cores, it is considerably faster than DFO-LS, and in noisy settings it
    outperforms all benchmarked configurations of DFO-LS.

    .. note::
       This algorithm requires the tranquilo package (version 0.1.0 or newer) to
       be installed. You can install it with ``pip install tranquilo`` or
       ``conda install -c conda-forge tranquilo``.

    """

    # basic options
    noisy: bool = False
    """Whether the residual functions are noisy, i.e. whether repeated evaluations
    at the same parameter vector return different values.

    If True, tranquilo activates its adaptive noise handling: the objective function
    is evaluated multiple times at each sample point, and the number of evaluations
    per point is adjusted over the course of the optimization such that just enough
    noise is averaged out to keep making progress. Moreover, the acceptance decision
    for candidate points is then based on a statistical power analysis, and the
    number of evaluations at the start parameters defaults to 5.

    """

    # convergence options
    disable_convergence: bool = False
    """If True, the optimization is terminated only by the stopping criteria
    (``stopping_maxfun``, ``stopping_maxiter``, ``stopping_maxtime``), never because
    a convergence criterion is satisfied.

    This is mostly useful for benchmarking.

    """

    convergence_ftol_abs: NonNegativeFloat = 0.0
    r"""Converge if the absolute change in the objective function value between two
    accepted iterations is less than this value. More formally, this is expressed as

    .. math::

        |f^k - f^{k+1}| \leq \textsf{convergence_ftol_abs}.

    This criterion is disabled by default (0.0).

    """

    convergence_gtol_abs: NonNegativeFloat = 0.0
    """Converge if the Euclidean norm of the gradient of the surrogate model at the
    current trust-region center is less than this value.

    Since tranquilo never evaluates derivatives of the objective function, the
    gradient of the fitted surrogate model is used instead of the true gradient.
    This criterion is disabled by default (0.0).

    """

    convergence_xtol_abs: NonNegativeFloat = 0.0
    r"""Converge if the Euclidean distance between the accepted parameter vectors of
    two consecutive iterations is less than this value. More formally, this is
    expressed as

    .. math::

        \lVert x^k - x^{k+1} \rVert \leq \textsf{convergence_xtol_abs}.

    This criterion is disabled by default (0.0).

    """

    convergence_ftol_rel: NonNegativeFloat = 2e-9
    r"""Converge if the relative change in the objective function value between two
    accepted iterations is less than this value. More formally, this is expressed as

    .. math::

        \frac{|f^k - f^{k+1}|}{\max\{|f^k|, 1\}} \leq \textsf{convergence_ftol_rel}.

    """

    convergence_gtol_rel: NonNegativeFloat = 1e-8
    r"""Converge if the Euclidean norm of the gradient of the surrogate model at the
    current trust-region center, divided by its maximum with 1, is less than this
    value. Denoting the gradient of the surrogate model by :math:`g^k`, this is
    expressed as

    .. math::

        \frac{\lVert g^k \rVert}{\max\{\lVert g^k \rVert, 1\}} \leq
        \textsf{convergence_gtol_rel}.

    """

    convergence_xtol_rel: NonNegativeFloat = 1e-8
    r"""Converge if the relative change of the accepted parameter vectors between two
    consecutive iterations is less than this value. More formally, this is expressed
    as

    .. math::

        \left\lVert \frac{x^k - x^{k+1}}{\max\{|x^k|, 1\}} \right\rVert \leq
        \textsf{convergence_xtol_rel},

    where the division and the maximum are taken element-wise.

    """

    convergence_min_trust_region_radius: NonNegativeFloat = 0.0
    """Consider the optimization converged if the trust-region radius shrinks below
    this value.

    This criterion is disabled by default (0.0).

    """

    # stopping options
    stopping_maxfun: PositiveInt = 2_000
    """Maximum number of objective function evaluations before termination.

    Repeated evaluations at the same point (as used for noisy problems) count
    towards this limit.

    """

    stopping_maxiter: PositiveInt = 200
    """Maximum number of iterations (trust-region steps) before termination."""

    stopping_maxtime: NonNegativeFloat = np.inf
    """Maximum running time in seconds before termination."""

    # single advanced options
    batch_evaluator: Literal[
        "joblib",
        "pathos",
    ] = "joblib"
    """Batch evaluator that is used to parallelize evaluations of the objective
    function.

    See :ref:`batch_evaluators` for details.

    """

    n_cores: PositiveInt = 1
    """Number of cores used to evaluate the objective function in parallel.

    If larger than one, tranquilo evaluates batches of points in parallel and uses a
    parallel line search as well as speculative sampling to keep all cores busy.

    """

    batch_size: PositiveInt | None = None
    """Number of points that are evaluated in each batch of parallel function
    evaluations.

    Must be at least as large as ``n_cores``, to which it defaults.

    """

    sample_size: PositiveInt | None = None
    """Target number of evaluated points in the trust region on which the surrogate
    models are fitted.

    By default, the sample size is determined from the model type and the dimension
    :math:`n` of the problem: :math:`n + 1` (in the noisy or parallel case) or
    :math:`n + 2` (otherwise) for linear models, and :math:`2 n + 1` for quadratic
    models.

    """

    model_type: (
        Literal[
            "quadratic",
            "linear",
        ]
        | None
    ) = None
    """Type of surrogate model that is fitted to each residual in every iteration.

    Defaults to "linear" for the least-squares version of tranquilo. The residual
    models are aggregated into a quadratic model of the objective function, so even
    with linear residual models the algorithm has curvature information.

    """

    search_radius_factor: PositiveFloat | None = None
    """Factor by which the trust-region radius is multiplied to obtain the region in
    which existing evaluation points are searched for reuse in the surrogate model.

    Reusing existing points saves function evaluations. The default is 5.0 for the
    least-squares version and 4.25 for the scalar version of tranquilo.

    """

    n_evals_per_point: NonNegativeInt | None = None
    """Initial number of times the objective function is evaluated (and averaged) at
    each sample point.

    This is only relevant for noisy problems, where the default is derived from
    ``noise_adaptation_options`` and the number of evaluations per point is
    adaptively adjusted over the course of the optimization. For non-noisy problems,
    the default is 1.

    """

    n_evals_at_start: NonNegativeInt | None = None
    """Number of objective function evaluations at the start parameters.

    The default is 5 for noisy problems and 1 otherwise. Multiple evaluations at the
    start parameters guarantee that enough evaluations are available to estimate the
    variance of the noise in the objective function.

    """

    seed: int | None = 925408
    """Seed for the random number generators used to sample new points and to
    simulate noise during the noise adaptation.

    Pass None for non-reproducible randomness.

    """

    # bundled advanced options
    radius_options: RadiusOptions | None = None
    """Advanced options for the management of the trust-region radius, e.g. the
    initial, minimal and maximal radius as well as expansion and shrinking factors.

    Pass an instance of ``tranquilo.options.RadiusOptions``; see the tranquilo
    package for details.

    """

    stagnation_options: StagnationOptions | None = None
    """Advanced options that determine how the algorithm reacts if it stagnates,
    i.e. if a proposed step is very small relative to the trust-region radius.

    Pass an instance of ``tranquilo.options.StagnationOptions``; see the tranquilo
    package for details.

    """

    noise_adaptation_options: NoiseAdaptationOptions | None = None
    """Advanced options that govern how the number of evaluations per point is
    adapted for noisy problems, e.g. the minimal and maximal number of evaluations
    per point.

    Pass an instance of ``tranquilo.options.NoiseAdaptationOptions``; see the
    tranquilo package for details.

    """

    # component names and related options
    sampler: (
        Literal[
            "optimal_hull",
            "random_hull",
            "random_interior",
        ]
        | Callable
    ) = "optimal_hull"
    """Method used to sample new points within the trust region.

    "optimal_hull" (the default) places new points on the hull of the trust region
    such that the minimal distance between all points (including existing ones) is
    approximately maximized. "random_hull" and "random_interior" sample random
    points on the hull or in the interior of the trust region. A custom sampling
    function can be passed as a callable.

    """

    sampler_options: SamplerOptions | None = None
    """Advanced options for the sampler.

    Pass an instance of ``tranquilo.options.SamplerOptions``; see the tranquilo
    package for details.

    """

    sample_filter: (
        Literal[
            "discard_all",
            "keep_all",
            "clustering",
            "drop_excess",
        ]
        | Callable
        | None
    ) = None
    """Method used to select which of the existing evaluation points inside the
    search region are used to fit the surrogate models.

    "keep_all" keeps all points, "discard_all" keeps only the trust-region center,
    "drop_excess" drops points if there are more than needed to fit the surrogate
    model, preferring points that are spread out within the trust region, and
    "clustering" keeps only one point per cluster of nearby points. The default is
    "drop_excess" if ``batch_size`` is larger than one and "keep_all" otherwise. A
    custom filter function can be passed as a callable.

    """

    sample_filter_options: FilterOptions | None = None
    """Advanced options for the sample filter.

    Pass an instance of ``tranquilo.options.FilterOptions``; see the tranquilo
    package for details.

    """

    model_fitter: (
        Literal[
            "ols",
            "ridge",
            "powell",
            "tranquilo",
        ]
        | Callable
        | None
    ) = None
    """Method used to fit the surrogate models to the sample of evaluated points.

    "ols" uses ordinary least-squares regression, "ridge" uses ridge regression,
    "powell" switches between ordinary least squares and a minimal-Frobenius-norm-
    of-Hessian fit depending on the sample size, following ideas from
    :cite:`Powell2004`, and "tranquilo" is a variant of "ols" that penalizes the
    linear terms of the model less strongly when the fitting problem is
    underdetermined. By default, "ols" is used if the sample size is large enough to
    identify all model parameters and "tranquilo" otherwise. A custom fitting
    function can be passed as a callable.

    """

    model_fitter_options: FitterOptions | None = None
    """Advanced options for the model fitter, e.g. penalty terms for ridge
    regression.

    Pass an instance of ``tranquilo.options.FitterOptions``; see the tranquilo
    package for details.

    """

    cube_subsolver: (
        Literal[
            "bntr",
            "bntr_fast",
            "fallback_cube",
            "fallback_multistart",
        ]
        | Callable
    ) = "bntr_fast"
    """Solver for the trust-region subproblem if the trust region is a cube, which
    is the case whenever bounds are binding.

    "bntr" and "bntr_fast" implement a bounded Newton trust-region algorithm, where
    "bntr_fast" (the default) is a numba-accelerated version of "bntr".
    "fallback_cube" and "fallback_multistart" are robust fallback solvers based on
    SciPy's L-BFGS-B, where the multistart version restarts the solver from multiple
    starting points. A custom subsolver can be passed as a callable.

    """

    sphere_subsolver: (
        Literal[
            "gqtpar",
            "gqtpar_fast",
            "fallback_reparametrized",
            "fallback_inscribed_cube",
            "fallback_norm_constraint",
        ]
        | Callable
    ) = "gqtpar_fast"
    """Solver for the trust-region subproblem if the trust region is a sphere, which
    is the case when no bounds are binding.

    "gqtpar" and "gqtpar_fast" solve the subproblem almost exactly with the method
    of :cite:`More1983`, where "gqtpar_fast" (the default) is a numba-accelerated
    version of "gqtpar". The "fallback_*" solvers are robust but less precise
    alternatives based on general-purpose optimizers. A custom subsolver can be
    passed as a callable.

    """

    retry_subproblem_with_fallback: bool = True
    """Whether to retry solving the trust-region subproblem with a fallback solver
    if the main subsolver raises an exception."""

    subsolver_options: SubsolverOptions | None = None
    """Advanced options for the trust-region subproblem solvers, e.g. the maximum
    number of iterations and gradient tolerances.

    Pass an instance of ``tranquilo.options.SubsolverOptions``; see the tranquilo
    package for details.

    """

    acceptance_decider: (
        Literal[
            "classic",
            "naive_noisy",
            "classic_line_search",
            "noisy",
        ]
        | Callable
        | None
    ) = None
    """Method used to decide whether a candidate point is accepted as the new
    trust-region center.

    "classic" performs the standard acceptance step of trust-region algorithms,
    which compares the actual with the predicted improvement.
    "classic_line_search" additionally evaluates speculative points along the
    search direction in parallel. "naive_noisy" averages a fixed number of
    evaluations at the candidate point. "noisy" uses a statistical power analysis
    to determine how many evaluations at the candidate and the current point are
    needed to decide which one is better. The default is "noisy" for noisy problems
    and "classic" otherwise. A custom acceptance function can be passed as a
    callable.

    """

    acceptance_decider_options: AcceptanceOptions | None = None
    """Advanced options for the acceptance decider, e.g. the confidence and power
    levels of the power analysis for noisy problems.

    Pass an instance of ``tranquilo.options.AcceptanceOptions``; see the tranquilo
    package for details.

    """

    variance_estimator: Literal["classic"] | Callable = "classic"
    """Method used to estimate the variance of the noise in the objective function
    for noisy problems.

    The "classic" estimator uses existing repeated function evaluations at points in
    a neighborhood of the current trust region and treats the noise variance as
    locally constant. A custom estimation function can be passed as a callable.

    """

    variance_estimator_options: VarianceEstimatorOptions | None = None
    """Advanced options for the variance estimator, e.g. the minimal number of
    evaluations per point used in the estimation.

    Pass an instance of ``tranquilo.options.VarianceEstimatorOptions``; see the
    tranquilo package for details.

    """

    infinity_handler: Literal["relative"] | Callable = "relative"
    """Method used to clip infinite objective function values before fitting the
    surrogate models.

    The "relative" method clips infinite values at a penalty value that is derived
    from the range of the finite values in the sample. A custom function can be
    passed as a callable.

    """

    residualize: bool | None = None
    """Whether the surrogate models are fitted to the deviations of the function
    values from the predictions of the previous surrogate model instead of the
    function values themselves.

    Defaults to True if the "tranquilo" model fitter is used and False otherwise.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        if not IS_TRANQUILO_VERSION_NEWER_OR_EQUAL_TO_0_1_0:
            raise NotInstalledError(TRANQUILO_INSTALLATION_INSTRUCTIONS)
        from tranquilo.tranquilo import _tranquilo

        raw_res = _tranquilo(
            functype="least_squares",
            batch_fun=problem.batch_fun,
            x=x0,
            lower_bounds=problem.bounds.lower,
            upper_bounds=problem.bounds.upper,
            noisy=self.noisy,
            disable_convergence=self.disable_convergence,
            convergence_absolute_criterion_tolerance=self.convergence_ftol_abs,
            convergence_absolute_gradient_tolerance=self.convergence_gtol_abs,
            convergence_absolute_params_tolerance=self.convergence_xtol_abs,
            convergence_relative_criterion_tolerance=self.convergence_ftol_rel,
            convergence_relative_gradient_tolerance=self.convergence_gtol_rel,
            convergence_relative_params_tolerance=self.convergence_xtol_rel,
            convergence_min_trust_region_radius=self.convergence_min_trust_region_radius,
            stopping_max_criterion_evaluations=self.stopping_maxfun,
            stopping_max_iterations=self.stopping_maxiter,
            stopping_max_time=self.stopping_maxtime,
            n_cores=self.n_cores,
            batch_size=self.batch_size,
            sample_size=self.sample_size,
            model_type=self.model_type,
            search_radius_factor=self.search_radius_factor,
            n_evals_per_point=self.n_evals_per_point,
            n_evals_at_start=self.n_evals_at_start,
            seed=self.seed,
            radius_options=self.radius_options,
            stagnation_options=self.stagnation_options,
            noise_adaptation_options=self.noise_adaptation_options,
            sampler=self.sampler,
            sampler_options=self.sampler_options,
            sample_filter=self.sample_filter,
            sample_filter_options=self.sample_filter_options,
            model_fitter=self.model_fitter,
            model_fitter_options=self.model_fitter_options,
            cube_subsolver=self.cube_subsolver,
            sphere_subsolver=self.sphere_subsolver,
            retry_subproblem_with_fallback=self.retry_subproblem_with_fallback,
            subsolver_options=self.subsolver_options,
            acceptance_decider=self.acceptance_decider,
            acceptance_decider_options=self.acceptance_decider_options,
            variance_estimator=self.variance_estimator,
            variance_estimator_options=self.variance_estimator_options,
            infinity_handler=self.infinity_handler,
            residualize=self.residualize,
        )
        res = InternalOptimizeResult(
            x=raw_res["solution_x"],
            fun=raw_res["solution_criterion"],
            message=raw_res["message"],
            info={"states": raw_res["states"]},
        )
        return res
