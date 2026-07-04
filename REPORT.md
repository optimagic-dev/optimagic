# Report: Algorithm documentation standardization

Branch `standardize-algo-docs`, 2026-07-04. All ~60 remaining algorithms were migrated
to the documentation system described in
`docs/source/how_to/how_to_document_optimizers.md`. During the migration, every claim in
the old `algorithms.md` text was checked against the optimagic code and the wrapped
libraries' documentation/source. This file records (1) errors in the old documentation
that were corrected, (2) code bugs discovered along the way that were deliberately NOT
fixed (documentation-focused branch), and (3) the one behavioral fix that was applied.

## 1. Errors in the old documentation (corrected)

### scipy

- **Stale constraint-support notes.** The old text said "SLSQP's / COBYLA's /
  trust_constr's general nonlinear constraints are not supported yet by optimagic".
  All three have `supports_nonlinear_constraints=True` today. The new docstrings
  instead explain how optimagic converts equality constraints to inequality pairs
  for COBYLA (which natively only handles inequalities).
- **Copy-paste error in `scipy_newton_cg`.** The warning advised refining "an optimum
  found with Powell" with another optimizer — it meant an optimum found with
  newton_cg.
- **Wrong default for `scipy_direct`.** The old text claimed the volume tolerance
  `eps` defaults to 1e-6, "differing from scipy's default 1e-4". The actual optimagic
  default is `CONVERGENCE_FTOL_REL = 2e-9`.
- **Nonexistent options documented.** `scipy_truncated_newton` was documented with
  `stopping.maxiter`, but the class only has `stopping_maxfun`; `scipy_ls_lm` was
  documented with `tr_solver`/`tr_solver_options`, which only exist on the trf and
  dogbox least-squares solvers. Dropped.
- Typo `finitie_difference_precision`; several places where optimagic defaults
  silently differ from SciPy are now stated explicitly (Nelder-Mead ftol 1e-8 vs
  SciPy 1e-4, SLSQP 1e-8 vs 1e-6, BFGS xrtol 1e-5 vs 0, trust_constr xtol 1e-5 vs
  1e-8, dual_annealing maxfun 1e6 vs 1e7, differential_evolution atol 1e-8 vs 0).

### nlopt

- **MMA/CCSAQ constraint support was backwards.** The old text said these support
  "nonlinear equality (but not inequality) constraints". NLopt supports the
  opposite: nonlinear *inequality* (not equality) constraints. In optimagic, MMA
  gets equalities via conversion to inequality pairs, and CCSAQ currently exposes no
  nonlinear constraints at all (`supports_nonlinear_constraints=False`).
- **Wrong dropdown title.** The section was titled `nlopt_lbfgs`, but the registered
  algorithm name is `nlopt_lbfgsb`.
- **`nlopt_tnewton` is not preconditioned.** The old text described it as
  "preconditioned truncated Newton"; the wrapped variant is `LD_TNEWTON`, the plain
  variant without preconditioning or restarting (those are separate NLopt variants).
- **Wrong DIRECT variant list.** The old text offered a nonexistent "DIRECT_RAND"
  and omitted "DIRECT_NOSCAL". The code maps six combinations, and
  `random_search=True` only takes effect together with `locally_biased=True` (now
  documented on the field).
- **Outdated PRAXIS bound-handling claim.** "Returns infinity when constraints are
  violated" no longer matches NLopt's documentation ("bounds emulated by variable
  transformation"). The warning that optimagic disables bounds for PRAXIS is kept.
- "Globally convergent" for MMA/CCSAQ clarified per NLopt's own note (convergence to
  a *local* optimum from any feasible starting point); removed an unsourced "<10
  past updates" claim for LBFGS storage; numerous typos (Nedler-Mead, alggorithm,
  faunction, a `:cite:` role missing its colon).

### pygmo

- **Wrong default population size formula.** The old text said "twice the number of
  parameters but at least X". The code (`get_population_size`) uses
  `10 * (n_params + 1)`, clipped below at the algorithm-specific minimum.
- **`ftol`/`xtol` descriptions swapped** for sade, cmaes and xnes — a known pagmo
  documentation bug that had been copied into algorithms.md. Verified against
  `cmaes.cpp`/`de.cpp`: ftol is flatness of the population's fitness values, xtol is
  flatness in parameter space.
- **Wrong option ranges/defaults** (all verified against the pagmo C++ source):
  `pygmo_de.weight_coefficient` documented as range [0, 2], pagmo enforces [0, 1];
  `pygmo_sga.mutation_polynomial_distribution_index` documented "[0, 1], default 1",
  pagmo enforces [1, 100]; `pygmo_sga.selection_tournament_size` default is 2, not 1.
- **Undocumented deviations from pygmo defaults** now stated: bee_colony
  `max_n_trials` (optimagic 1 vs pygmo 20), mbh
  `stopping_max_inner_runs_without_improvement` (30 vs 5), simulated_annealing
  end temperature (0.01 vs 0.1, must be below the start temperature).
- Broken `.. note:` directive (single colon); GWO section typos ("usinng", "pased",
  "shokingly") and its criticism of the algorithm is now attributed to the pagmo
  developers rather than "our opinion"; `batch_evaluator` bullets dropped for gaco
  and pso_gen (the classes only have `n_cores`).

### ipopt

- **Contradictory `linear_solver` default.** The old text said the default is
  "ma27" while its own value list said "mumps (default)". The code default is
  `"mumps"`.
- **`acceptable_dual_inf_tol` mismatch.** The old text documented Ipopt's default
  1e+10, but the code passes 1e-10 (see code bugs below). The docstring now
  documents the actual behavior and flags the deviation.
- **Mangled bullets un-fused.** `theta_max_fact` had been fused into the
  `watchdog_trial_iter_max` bullet; `recalc_y_feas_tol` was mislabeled.
- **Wrong default for `resto_failure_feasibility_threshold`.** Old text said 0; the
  wrapper maps `None` to `100 * convergence_ftol_rel`.
- Value typos: "10-05" → 1e-5 (`filter_margin_fact`), "10-06" → 1e-6 (`sigma_min`),
  `"bound_mult"` → `"bound-mult"` (the code's Literal),
  `adaptive_mu_kkterror_red_iters` labeled float but is an int; word typos
  ("lwer", "bunded", "brrier"). Deviations from Ipopt defaults now flagged
  (`convergence_ftol_rel` 2e-9 vs Ipopt `tol` 1e-8; `stopping_maxiter` 1,000,000 vs
  Ipopt 3000).

### NAG (DFO-LS, Py-BOBYQA)

- **`seek_global_optimum` restriction was wrong.** Old text: "Only applies for noisy
  criterion functions." Py-BOBYQA's documentation demonstrates it on noise-free
  functions; the actual requirement is finite lower and upper bounds.
- **Fast-start default condition inverted.** Old text said "jacobian" is the default
  fast-start method "if `len(x) >= number of root contributions`". DFO-LS documents
  the opposite (`use_full_rank_interp` defaults to True iff m >= n), i.e. "jacobian"
  is default when the number of residuals is at least the number of parameters.
- **Wrong option value in prose.** "trustregion_step" — the value accepted by the
  code is `"trustregion"`.
- Stale `:ref:` pointers for options whose default constants no longer live in
  `algo_options.py`; descriptions are now inline on the fields.

### fides

- **Wrong documented defaults.** `trustregion.subspace_dimension` was documented as
  defaulting to "2D"; the optimagic default is `"full"` (the fides package itself
  defaults to "2D"). Similarly, optimagic defaults `stepback_strategy` to
  `"truncate"` while fides defaults to `"reflect"`. Both deviations are now stated.
- **Nonexistent options dropped.** `trustregion.refine_stepback` and
  `trustregion.scaled_gradient_as_possible_stepback` were documented but are not
  fields of the class.
- Convergence criteria formulas verified against fides 0.7.4
  (`Optimizer.check_convergence`) and written as math blocks.

### TAO pounders

- **Outdated link.** The old text linked github.com/erdc/petsc4py, an unofficial and
  outdated mirror; petsc4py is developed as part of PETSc.
- **Wrong disabling semantics.** "Set to False to disable" a tolerance — the fields
  are floats and the code disables on any falsy value; correct advice is "set to
  zero to disable".
- LaTeX typos (missing backslash on `\epsilon`, `X0` → `X_0`).

### Own optimizers (bhhh, neldermead_parallel, pounders)

- **bhhh:** outdated interface description (fun returning a dict with a
  "contributions" entry) replaced by the current `@om.mark.likelihood` interface;
  "only vaid" typo.
- **neldermead_parallel:** the class docstring was a broken numpydoc stub
  documenting nonexistent `criterion`/`x` parameters; `adaptive` adapts the simplex
  parameters to the problem *dimension* (Gao–Han), not to "simplex size"; the code's
  capping of `n_cores` at `n_params - 1` is now documented.
- **pounders:** `convergence.gtol_rel` default written as "1-8" instead of 1e-8;
  `trustregion_expansion_factor_successful` was described as a "Shrinking factor" —
  it is an expansion factor; the documented literal "steihaug-toint" is actually
  `"steihaug_toint"`; old option names `trustregion_threshold_successful` /
  `..._very_successful` are now `trustregion_threshold_acceptance` /
  `..._successful` and are documented under the current names; the documented
  `batch_evaluator` option no longer exists (only `n_cores`).

### Other documentation fixes

- **`nevergrad_wizard` / `nevergrad_portfolio` dropdowns were broken.** They
  referenced classes that no longer exist (`NevergradWizard`, `Wizard`,
  `NevergradPortfolio`, `Portfolio`, including `from ... import Wizard` usage
  examples). The algorithms are registered as `nevergrad_ngopt` and
  `nevergrad_meta`; dropdowns renamed and examples rewritten using the string-based
  `optimizer` option.
- **`how_to_add_optimizers.ipynb` crashed the docs build.** Its example
  `@mark.minimizer` call was missing the now-required `needs_bounds` and
  `supports_infinite_bounds` arguments. Also fixed its dead cross-reference
  `algo_options_docs` → `algo_options`.
- **tranquilo was not documented anywhere** (no docstrings, no algorithms.md
  section). Documented from scratch based on the tranquilo paper (Gabler, Gsell,
  Mensinger, Petrosyan 2024) and the tranquilo 0.1.1 source.

## 2. Code bugs discovered but NOT fixed here

These surfaced while verifying documentation against the code. They are behavioral
issues and deserve their own PRs.

- **`BHHH.converence_gtol_abs`** (bhhh.py): the field name is misspelled
  ("converence"). Renaming is user-facing (algo_options key), so it needs a
  deprecation path.
- **`Ipopt.acceptable_dual_inf_tol = 1e-10`** (ipopt.py): Ipopt's default is 1e+10.
  A "desired" threshold of 1e-10 makes the acceptable-point heuristic essentially
  impossible to trigger via dual infeasibility — this looks like a sign/typo bug
  when the default was transcribed.
- **NAG `noise_n_evals_per_point`** (nag_optimizers.py): annotated
  `NonNegativeInt | None`, but `_change_evals_per_point_interface` only works with a
  *callable*; passing an int would crash inside the solver.
- **NAG fast-start dict key `"min_inital_points"`** (nag_optimizers.py): the key
  validated by `_build_options_dict` is literally misspelled; the correctly spelled
  key raises. Additionally, the old constant docstring mentioned
  `scale_of_jacobian_singular_value_floor`, which is not an accepted key.
- **`PygmoSga.mutation_strategy`** (pygmo_optimizers.py): the Literal type lacks
  `"gaussian"` although the implementation handles it — as typed, the
  `mutation_gaussian_width` option is unreachable.
- **`PygmoXnes.population_size`** (pygmo_optimizers.py): typed `float | None`;
  should presumably be an int type.
- **`TAOPounders` convergence criteria may be inert** (tao_optimizers.py): the
  wrapper always installs a user-defined maxiter convergence test (since
  `stopping_maxiter` is never None), which per TAO semantics replaces the default
  gradient-tolerance tests — so the documented gatol/grtol/gttol criteria appear to
  have no effect through this interface.
- **Tranquilo dead options** (tranquilo.py, verified against tranquilo 0.1.1):
  `batch_evaluator` is never read by the wrapper (the internal evaluator is
  hardcoded to "joblib"); `stopping_maxtime` is passed but never checked in
  tranquilo's main loop; `convergence_min_trust_region_radius` maps to
  `ConvOptions.min_radius`, which `_is_converged` never reads; `functype` is
  hardcoded to "scalar" by the wrapper.
- **scipy `relative_step_size_diff_approx` is effectively dead** (scipy_optimizers.py,
  trf/dogbox): optimagic always passes a Jacobian callable to
  `scipy.optimize.least_squares`, so SciPy's internal finite differencing (which
  this option controls) is never used.
- **refs.bib duplicate**: `Cartis2018` and `Cartis2018b` are two entries for the same
  arXiv paper (1804.00154). Both keys are cited; consolidating requires touching the
  citations.

## 3. Behavioral fix applied in this branch

**`from __future__ import annotations` silently disabled algo-option coercion.**
`Algorithm.__post_init__` converts/validates option values by looking up each
dataclass field's type in `TYPE_CONVERTERS` (keyed by type objects such as
`PositiveInt`). With the future import — which the documentation how-to guide
*requires* so that autodoc renders type aliases readably — `field.type` is the
annotation *string* (`"PositiveInt"`), so no lookup ever matched and coercion was
silently skipped. This already affected every module that had the import before this
branch (scipy, gfo, nevergrad, pyswarms, pygad, bayesian, pygmo, tranquilo, ...) and
surfaced here as a test failure when ipopt gained the import
(`test_ipopt_algo_options` passes 5.5 for an integer-typed Ipopt option and relies on
coercion to int).

Fix: `TYPE_CONVERTERS_BY_NAME` in `src/optimagic/type_conversion.py` (string-keyed
companion table) and a two-branch lookup in `Algorithm.__post_init__`. Coercion now
works identically whether or not a module uses the future import. Full fast test
suite passes (3055 passed). If this branch should stay documentation-only, the fix
extracts cleanly into a separate commit/PR — but note the ipopt options test fails
without it.
