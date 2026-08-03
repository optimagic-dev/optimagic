# Review guide: constraints refactoring stack

Context document for Claude sessions that review one PR of the constraints
refactoring. Read this first, then check out the branch under review. This file is
deliberately untracked; delete it when the whole stack is merged.

## Status (2026-07-03, evening)

- PRs 1-3 are **merged into `main`** (squash: #686, #687, #688). `main` is the base
  of the remaining stack.
- Branches 4-6 have been rebased onto the new `main` (each squash drops the original
  commits, so this used `git rebase --onto origin/main <old-branch-tip> <branch>`
  per branch). All remaining branches contain their predecessor and are green on
  the targeted test loop, mypy, and pre-commit; the full fast suite passed on the
  last branch.
- PR 3 review landed major design changes (now in `main`): selector resolution is a
  `Constraint._resolve(context)` method on each user constraint class; the
  `Resolved*Constraint` dataclasses live in `optimagic.constraints` directly after
  their user-facing counterparts (`parameters/constraints/types.py` no longer
  exists); resolved class names exactly match their user counterparts
  (e.g. `ResolvedFlatCovConstraint`); resolving a `NonlinearConstraint` raises
  `NotImplementedError`. Constraint-type-specific behavior is unit tested in
  `tests/optimagic/test_constraints.py`; `test_resolution.py` covers only general
  end-to-end resolution behavior.

## What this refactoring is

The code that implements constraints via reparametrization (formerly
`process_constraints.py`, `consolidate_constraints.py`, `check_constraints.py`,
`process_selectors.py`, `kernel_transformations.py`, `space_conversion.py` in
`src/optimagic/parameters/`) was rewritten in 5 stacked PRs. Goals: typed internals
(frozen dataclasses instead of the mutable `constr_info` dict and constraint dicts),
less entanglement, provenance-based error messages that cite the user's original
constraints, and extension slots for kernel second derivatives, jax derivatives,
probability-with-fixed-params, and native linear-constraint pass-through (none of
these extensions are implemented; do not add them during review).

The full design and rationale are in the plan file
`~/.claude/plans/i-want-to-refactor-moonlit-nygaard.md`. The math is unchanged and
documented in `docs/source/explanation/implementation_of_constraints.md`.

## The branch stack (review in this order)

Each branch is one PR, sits on top of the previous one, and is green on the full
fast test suite, mypy, and pre-commit.

1. `constraints-refactor-1-characterization-tests` — **MERGED (squash, #686)**; now
   the base of the stack.
   Test-only safety net: `tests/optimagic/parameters/test_constraint_pipeline_invariants.py`
   pins golden internal params for 26 constraint sets, round trips, the feasibility
   invariant, derivatives, and first-error types at the `get_converter` seam. During
   review the corpus grew (shifted covariance blocks, sdcorr simplified to bounds,
   uncorrelated sdcorr) and the fixtures were made typed/frozen (a `Case` dataclass
   and an `ExpectedInternal` dataclass instead of dicts).
   Also carries two behavior fixes: `check_fixes_and_bounds` crashed with `TypeError`
   instead of raising `InvalidConstraintError`; and the uncorrelated-covariance
   simplification now works for covariance blocks that do not start at position 0 of
   the flat vector (the off-diagonal indices are selected by block-local position, not
   by matching global index values). The second fix was moved up from PR 6.

2. `constraints-refactor-2-typed-boundary`
   `deprecations.pre_process_constraints` inverted: legacy dict constraints
   (incl. `loc/locs/query/queries`, which become selector closures) are converted
   INTO `Constraint` objects; `OptimizationProblem.constraints` is
   `list[Constraint]`; nonlinear split via `isinstance`. Temporary `_to_dict` seams
   keep the old dict pipeline running (removed in PR 5).
   New tests were added to `tests/optimagic/test_deprecations.py` (not a separate
   file: they cover `deprecations.pre_process_constraints`, so they can be deleted
   together with the rest of the deprecation-era tests once dict constraints are
   removed in 0.6.0).

3. `constraints-refactor-3-typed-resolution` — **MERGED (squash, #688)**.
   Selector resolution as typed `Constraint._resolve(context)` methods; the
   `Resolved*Constraint` dataclasses and `ConstraintSource` provenance live in
   `optimagic.constraints`; `resolution.py` holds the loop and `ResolutionContext`.
   `process_selectors.py` deleted. A temporary `to_legacy_dicts` seam feeds the
   still-dict-based consolidation (removed in PR 5).

4. `constraints-refactor-4-numpy-linear-consolidation`
   Shape-preserving pandas -> numpy rewrite of the linear consolidation, verified
   by a 200-scenario differential test against a verbatim copy of the pandas code
   (`tests/optimagic/parameters/test_linear_consolidation_differential.py`,
   deleted again in PR 5). Preserves first-occurrence dedup order and lb/ub swaps
   under negative rescaling; linear constraints no longer get their index rewritten
   during equality plugging (the rewritten index was unused before but would have
   corrupted the numpy weight scatter).

5. `constraints-refactor-5-typed-core`
   The big one: `validation.py`, `consolidation.py`, `consolidate_linear.py`,
   `transforms.py`, `kernels.py`; `SpaceConversionSpec` replaces `constr_info`;
   `SpaceConverter` gets real methods; `get_space_converter` orchestrates
   validate -> normalize -> consolidate. Old modules, all seams, and
   `Constraint._to_dict` are deleted. Test retargeting changed only constraint
   construction lines and imports — no expected value changed.

6. `constraints-refactor-6-review-fixes`
   Collects genuine behavior bugs found during review, one commit + regression test
   each. The uncorrelated-covariance-on-shifted-blocks fix now lands in PR 1 (`main`)
   and is carried through the typed core in PR 5, so this PR now only adds the
   dedicated consolidation-stage regression test (`test_consolidation.py`).

## Ground rules for review sessions

- **Golden values are the contract.** Never edit expected numbers in
  `test_constraint_pipeline_invariants.py` or `test_space_conversion.py`. If a
  change would require that, it changes the reparametrization and needs explicit
  discussion, not a test update.
- **Deliberate behavior changes — do not revert them:**
  - fixes/bounds colliding with cov/sdcorr/probability raise
    `InvalidConstraintError` (was `TypeError`, PR 1),
  - conflicting fixes on equality-constrained params raise
    `InvalidConstraintError` (was `AssertionError`, PR 5),
  - conflicting linear fixed values raise `InvalidConstraintError`
    (was `ValueError`, PR 5),
  - misaligned linear weights raise `InvalidConstraintError` at resolution time
    (was `ValueError` later, PR 3),
  - resolving a `NonlinearConstraint` raises `NotImplementedError` (was
    `InvalidConstraintError`, PR 3): nonlinear constraints are passed directly to
    optimizers and are split off before resolution, so this is an internal error,
    not invalid user input,
  - post-consolidation error messages cite the originating user constraints.
- **Where changes go:**
  - Review edits to PR N (naming, docstrings, structure, behavior-preserving
    refactors) -> new commit on branch N.
  - Genuine behavior bugs (pre-existing or introduced) -> branch
    `constraints-refactor-6-review-fixes`, one commit per fix with a regression
    test. Not on the PR branches, so those stay behavior-preserving.

## Propagation rule (important)

Every change committed to a branch during review MUST be merged into all later
branches of the stack by rebasing the rest of the stack. It is enough to do this
ONCE at the end of the session (not after every commit):

```bash
# after all review changes on constraints-refactor-<N>-... are committed:
git switch constraints-refactor-<N+1>-...
git rebase constraints-refactor-<N>-...
# repeat for each later branch in order, up to and including
# constraints-refactor-6-review-fixes
```

After rebasing, rerun the verification (below) on the LAST branch of the stack, not
just the branch under review. Never leave the session with the stack in a state
where a later branch does not contain an earlier branch.

## Verification commands

```bash
# fast targeted loop (seconds)
pixi run pytest tests/optimagic/parameters tests/optimagic/test_constraints.py \
  tests/optimagic/test_deprecations.py \
  tests/optimagic/optimization/test_with_constraints.py \
  tests/optimagic/optimization/test_with_advanced_constraints.py \
  tests/estimagic -m "not slow"

# full fast suite (~4-5 min), strict typing, hooks
pixi run tests-fast
pixi run mypy
pre-commit run --all-files
```

All new modules under `src/optimagic/parameters/constraints/` and the rewritten
`space_conversion.py` are strictly type checked (not in the mypy override exempt
list in `pyproject.toml`) — keep it that way.

## Known context that saves time

- Legacy dict constraints are deprecated (FutureWarning, removal in 0.6.0). The
  adapter in `deprecations.py` (incl. `FixedValueConstraint` and the loc/query
  selector closures) exists only to support them and dies with them.
- `InternalParams` and `SpaceConverter` still live in
  `parameters/space_conversion.py` on purpose: import paths for estimagic, the
  slice plots and `constraint_tools` are unchanged.
- Consolidation order in `consolidation.py::consolidate_constraints` is load
  bearing (equality merging -> fix propagation -> cov/sdcorr simplification ->
  bound tightening -> replacements -> dedup -> linear bundling -> checks -> spec
  assembly). Do not reorder while reviewing for style.
- The tests directory has no `__init__.py` files: test module basenames must be
  unique across the whole tests tree.
