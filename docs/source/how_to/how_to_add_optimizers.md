(how-to-add-optimizers)=

# How to add optimizers to optimagic

This guide shows you how to wrap an optimizer for optimagic and which standards a
wrapper has to meet. The same standards apply when you improve an existing wrapper.

You can use a wrapper in your own project without contributing it. Then only the
[walkthrough](wrapper-walkthrough) is relevant for you. Wrappers that become part of
optimagic also have to follow the [standards](wrapper-standards).

## The steps

1. **Learn to use the optimizer.** Solve a few simple problems with the library you want
   to wrap, without optimagic. Find out how to pass the objective function, options,
   bounds, constraints and derivatives, and how to get the results.
1. **Learn how the optimizer works.** Read the documentation and the papers. Find out
   which problems the optimizer was made for, what each option does and how to choose
   its value.
1. **Write a minimal wrapper** and test it.
1. **Make the option table.** List every option of the library and decide whether it is
   exposed, under which name, with which default and how it will be tested. For
   contributions, discuss the table with a maintainer before you continue. The
   [standards](wrapper-standards) tell you how to fill it.
1. **Complete the wrapper**: all options, tests, documentation and, for highly
   configurable optimizers, a tutorial.
1. **Integrate the wrapper into optimagic.**

If you use Claude Code, the `wrap-optimizer` skill (`.claude/skills/wrap-optimizer`)
guides an agent through these steps. It stops for your decision on the option table.

## Using AI tools

Follow the [AI policy of the contributor guide](../development/how_to_contribute.md).
For wrappers, two things are especially important:

1. Only use AI for drafts that you double-check. Option names, defaults and
   documentation are user-facing design decisions. Never rely on AI producing correct
   results, and give it the papers and the documentation of the library as context.
1. Be transparent about your use of AI.

(wrapper-walkthrough)=

## Walkthrough: wrapping `pygmo_gaco`

The walkthrough tells the story of how the `pygmo_gaco` optimizer was added by someone
who knew neither pygmo nor the gaco algorithm.

### 1. Learn to use the optimizer

The pygmo [tutorials](https://esa.github.io/pygmo2/tutorials/tutorials.html) show that
objective functions are passed as a problem class:

```python
import numpy as np
import pygmo as pg


class sphere_function:
    def fitness(self, x):
        return [sum(x * x)]

    def get_bounds(self):
        return ([-1, -1], [1, 1])


prob = pg.problem(sphere_function())
```

`fitness` returns the function value as a list with one entry and `get_bounds` returns
the bounds. An optimization at default values looks like this:

```python
# The initial population
pop = pg.population(prob, size=20)
# The algorithm; ker needs to be at most the population size to avoid errors
algo = pg.algorithm(pg.gaco(ker=20))
# The actual optimization process
pop = algo.evolve(pop)
# The best individual in the population
best_fitness = pop.get_f()[pop.best_idx()]
best_x = pop.get_x()[pop.best_idx()]
```

From this exploration we learn:

- Options such as `ker` are passed when the algorithm is created. They are listed
  [here](https://esa.github.io/pygmo2/algorithms.html#pygmo.gaco).
- Bounds are part of the problem class. gaco supports no other constraints and uses no
  derivatives.
- gaco can evaluate the function in parallel. For this the problem class needs a
  [`batch_fitness`](https://esa.github.io/pygmo2/problem.html#pygmo.problem.batch_fitness)
  method and the algorithm needs `set_bfe(pg.bfe())`.
- The results are stored in the evolved population. The number of function evaluations
  is `pop.problem.get_fevals()`.

For a real contribution you have to explore much more, and document it, to convince us
that you understand the library in detail.

### 2. Learn how the optimizer works

Answer these questions before you write the wrapper:

- For which problems and situations was the optimizer designed?
- How does it work, intuitively?
- Which papers and other sources describe it?
- What does each option do and how should a user choose its value?
- Which features of the library can optimagic not support, and why?
- Are there known limitations?

For gaco the answers are: It is a global optimizer that uses no derivatives and can
evaluate the function in parallel, so it is meant for expensive functions with several
local optima. It should not be used if a local optimum is enough or if derivatives are
available. It simulates ants that explore the parameter space and leave more
"pheromones" at good solutions, so that later generations search there more often. It
was proposed by
[Schlueter et al. (2009)](https://digital.csic.es/bitstream/10261/54957/3/Extended_ant_colony_2009.pdf).

The descriptions of the options in the pygmo documentation are short. Reading the paper
was necessary to understand them. For example, for `q` the documentation only says
"convergence speed parameter". The paper shows that smaller values make the search
concentrate faster on the best solutions found so far.

### 3. Write a minimal wrapper

A wrapper is a frozen dataclass that inherits from `Algorithm`, is decorated with
`mark.minimizer` and implements `_solve_internal_problem`. This method takes an
`InternalOptimizationProblem` and returns an `InternalOptimizeResult`. These
[classes and functions](functions_and_classes_for_internal_optimizers) are explained in
detail elsewhere.

```{note}
Users never create an `InternalOptimizationProblem`. They call `minimize` or
`maximize`, which create it from their inputs. The public interface is made for
convenience, the internal problem is made for easy wrapping: its functions take a
flat numpy array and already handle constraints, scaling, errors and the history.
```

```python
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

import optimagic as om
from optimagic.optimization.algorithm import Algorithm, InternalOptimizeResult
from optimagic.optimization.internal_optimization_problem import (
    InternalOptimizationProblem,
)
from optimagic.typing import AggregationLevel, PositiveInt

try:
    import pygmo as pg

    IS_PYGMO_INSTALLED = True
except ImportError:
    IS_PYGMO_INSTALLED = False


@om.mark.minimizer(
    # lowercase with underscores, starting with the name of the library
    name="pygmo_gaco",
    # scalar, least-squares or likelihood problems
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_PYGMO_INSTALLED,
    is_global=True,
    # gaco is derivative free
    needs_jac=False,
    needs_hess=False,
    # gaco needs finite bounds
    needs_bounds=True,
    supports_parallelism=True,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class PygmoGaco(Algorithm):
    # Only two options to get things running. The rest comes later.
    stopping_maxiter: PositiveInt = 1000
    n_cores: int = 1

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        n_cores = self.n_cores

        class PygmoProblem:
            def fitness(self, x):
                return [problem.fun(x)]

            def get_bounds(self):
                return (problem.bounds.lower, problem.bounds.upper)

            def batch_fitness(self, dvs):
                dim = len(self.get_bounds()[0])
                x_list = list(dvs.reshape(-1, dim))
                # problem.batch_fun is a parallelized version of problem.fun
                return np.array(problem.batch_fun(x_list, n_cores))

        prob = pg.problem(PygmoProblem())
        pop = pg.population(prob, size=20)
        pygmo_uda = pg.gaco(ker=20, gen=self.stopping_maxiter)
        pygmo_uda.set_bfe(pg.bfe())
        pop = pg.algorithm(pygmo_uda).evolve(pop)

        return InternalOptimizeResult(
            x=pop.get_x()[pop.best_idx()],
            fun=pop.get_f()[pop.best_idx()][0],
            n_fun_evals=pop.problem.get_fevals(),
        )
```

Test the wrapper in the way it will be used:

```python
res = om.minimize(
    fun=lambda x: x @ x,
    params=np.arange(5),
    algorithm=PygmoGaco,
    bounds=om.Bounds(lower=-np.ones(5), upper=np.ones(5)),
)
om.criterion_plot(res, monotone=True)
```

For debugging you can also call the wrapper directly with an example problem:

```python
from optimagic.optimization.internal_optimization_problem import (
    SphereExampleInternalOptimizationProblem,
)

problem = SphereExampleInternalOptimizationProblem()
result = PygmoGaco()._solve_internal_problem(problem, x0=np.array([1.0, 1.0]))
```

The arguments of `mark.minimizer` have to be correct even in a minimal wrapper.
optimagic uses them to decide which problems an algorithm can solve and what it passes
to the algorithm.

### 4. to 6. Complete and integrate the wrapper

The final version of the wrapper is in
[`pygmo_optimizers.py`](https://github.com/optimagic-dev/optimagic/blob/main/src/optimagic/optimizers/pygmo_optimizers.py).
The remaining steps are described by the standards below and by the
[integration checklist](wrapper-integration).

(wrapper-standards)=

## Standards for wrappers

The wrappers of
[fides](https://github.com/optimagic-dev/optimagic/blob/main/src/optimagic/optimizers/fides.py)
and of the
[NAG optimizers](https://github.com/optimagic-dev/optimagic/blob/main/src/optimagic/optimizers/nag_optimizers.py)
follow these standards and serve as examples.

(wrapper-standards-option-table)=

### The option table

Before you complete a wrapper, make a table with one row for every option of the
library. Options are the arguments of the function or class you call and everything that
can be set through option dictionaries or configuration objects. The columns are:

| Column                   | Content                                                                              |
| ------------------------ | ------------------------------------------------------------------------------------ |
| Library name and default | As in the newest release of the library.                                             |
| optimagic name           | Following the [naming rules](wrapper-standards-names), with the group it belongs to. |
| optimagic default        | Following the [rules for defaults](wrapper-standards-defaults).                      |
| Replaces                 | The old optimagic name, if you improve an existing wrapper.                          |
| Test                     | Behavior or forwarding, see [tests](wrapper-standards-tests).                        |

Options that are not exposed are listed below the table with the reason. The table is
the basis of the discussion with the maintainers and goes into the description of the
pull request.

(wrapper-standards-coverage)=

### Which options to expose

Expose every option that optimagic can serve. An option is not exposed if

- optimagic already handles it, for example logging, printing, the history, scaling or
  additional arguments of the objective function. Point to the optimagic feature in the
  docstring of the class.
- it does not do what it promises in the release you wrap. For example, a parallel mode
  that still evaluates the function one point at a time.
- it cannot have an effect inside optimagic. For example, a check that catches errors of
  the objective function which optimagic already handles before the library sees them.
- it needs information that optimagic cannot pass to an optimizer yet, for example exact
  Hessians, projections onto constraint sets or proximal operators.

List the features that are not supported in the docstring of the class and say why. Open
an issue if a feature could be supported with a reasonable extension of optimagic.

(wrapper-standards-structure)=

### Keep the library's grouping

- Options the library takes directly are direct options of the algorithm.
- If the library bundles options into a group, the group becomes one option called
  `<group>_options`. Examples are the option dictionary of DFO-LS with keys such as
  `tr_radius.gamma_dec`, which becomes `trustregion_options`, or a configuration object
  that is passed to the optimizer.
- If a group belongs to a feature that can be switched on and off, the switch and the
  group become one option (see [below](wrapper-standards-switch)).
- A group that is only a miscellaneous bucket is dissolved. DFO-LS has a group
  `general.*` with three unrelated options. A `general_options` argument would tell
  users nothing, so the three are direct options.
- Do not invent groups the library does not have and do not flatten groups it has.

A group is a frozen dataclass with one attribute per option. Decorate it with
`option_group`, annotate the field of the algorithm with the class and use the class as
`default_factory`:

```python
from optimagic.optimization.algo_option_tools import option_group


@option_group
@dataclass(frozen=True)
class TrustregionOptions:
    """Options that control how the trust region changes its size."""

    shrinking_factor: ProbabilityFloat = 0.5
    """Factor by which the radius shrinks after an unsuccessful step. ..."""

    expansion_factor: PositiveFloat = 2.0
    """..."""


@mark.minimizer(...)
@dataclass(frozen=True)
class MyAlgorithm(Algorithm):
    trustregion_options: TrustregionOptions = field(default_factory=TrustregionOptions)
    """Options that control how the trust region changes its size.

    An instance of :class:`TrustregionOptions`.

    """
```

Users pass an instance of the class. The entries are validated when the instance is
created:

```python
om.algos.my_algorithm(trustregion_options=TrustregionOptions(shrinking_factor=0.9))
om.minimize(
    ...,
    algo_options={"trustregion_options": TrustregionOptions(shrinking_factor=0.9)},
)
```

A dictionary with the same entries is rejected. This keeps one way of setting a group
that editors and type checkers understand. Write each group as its own class without
base classes: inherited attributes lose their type in the rendered documentation.

(wrapper-standards-switch)=

**Features that can be switched on and off:** Some libraries have a flag that switches a
feature on and options that configure it. DFO-LS has `restarts.use_restarts` and a dozen
more `restarts.*` options, which do nothing while the flag is off. Do not expose the
flag and the options separately, because options that are set without the flag would
silently do nothing. Combine them in one option that takes a boolean or the group. The
option is named after the feature and has no `_options` suffix, and the group has no
entry for the flag:

```python
restarts: bool | RestartOptions = False
"""Whether the optimizer restarts when it stagnates, and how. ..."""
```

```python
om.algos.my_algorithm(restarts=True)  # on, with all defaults, nothing to import
om.algos.my_algorithm(restarts=RestartOptions(max_restarts=5))  # on, configured
om.algos.my_algorithm(restarts=False)  # off
```

Passing the group always switches the feature on. This is the same pattern as `scaling`
and `multistart` in `minimize`. Declaring the function as noisy works the same way if
the library has options that describe the noise: `noisy=True` or
`noisy=NoiseOptions(additive_noise_level=0.1)`. If the library switches the feature on
by itself in some situations, add `None` as the default and state the rule in the
docstring.

If the library is configured with its own objects, for example a Hessian approximation
or an acquisition function, accept these objects without hiding them. In addition, offer
names for the common choices and a default, so that nobody has to import anything from
the library to get started. An example is `hessian_update` of `fides`, which accepts
`"bfgs"`, `"sr1"` or an instance of a class from `fides.hessian_approximation`.

(wrapper-standards-names)=

### Names of options

Keep the name the library uses if it is readable. Replace only

- symbols from the paper: `eta1`, `rho`, `gamma_dec`,
- cut-off words: `thresh` becomes `threshold`, `geom` becomes `geometry`,
- words for which optimagic has its own term (see the table below),
- names that describe the direction of the optimization. Every algorithm can also be
  used with `maximize`, so `seek_global_minimum` becomes `seek_global_optimum`.

Symbols that are used everywhere in the optimization literature are fine. For example,
counts start with `n_` as in `n_restarts`.

Longer is not better. A name should be as short as it can be while still telling a user
who has not read the paper what the option is about.

Options that occur in many algorithms have fixed names:

| Meaning                                               | Name                                                           |
| ----------------------------------------------------- | -------------------------------------------------------------- |
| The objective function                                | `fun` in all names, not `objfun` or `criterion`                |
| Maximum number of function evaluations                | `stopping_maxfun`                                              |
| Maximum number of iterations                          | `stopping_maxiter`                                             |
| Maximum runtime in seconds                            | `stopping_maxtime`                                             |
| Tolerances on the function value                      | `convergence_ftol_abs`, `convergence_ftol_rel`                 |
| Tolerances on the gradient                            | `convergence_gtol_abs`, `convergence_gtol_rel`                 |
| Tolerances on the parameters                          | `convergence_xtol_abs`, `convergence_xtol_rel`                 |
| Stop when this function value is reached              | `convergence_target_value`                                     |
| Stop when the trust-region radius is this small       | `convergence_min_trustregion_radius`                           |
| Radius of the trust region at the start               | `trustregion_initial_radius`                                   |
| Step quality below which the trust region shrinks     | `trustregion_shrinking_threshold`                              |
| Step quality above which the trust region grows       | `trustregion_expansion_threshold`                              |
| Factors for shrinking and growing the trust region    | `trustregion_shrinking_factor`, `trustregion_expansion_factor` |
| Solver of the trust-region subproblem and its options | `trustregion_subsolver`, `trustregion_subsolver_options`       |
| Whether the function is noisy, and how                | `noisy`                                                        |
| Evaluations that are averaged at each point           | `n_evals_per_point`                                            |
| Number of cores and random seed                       | `n_cores`, `seed`                                              |

All other criteria that end an optimization start with `convergence_` if they indicate
success and with `stopping_` if they limit resources. Inside a group the prefix of the
group is dropped: the entry of `trustregion_options` is called `shrinking_factor`. The
default values of the harmonized options are listed {ref}`here <algo_options>`.

If the same option exists in other wrappers of the same kind of algorithm, use the same
name.

**Optimizers from scipy:** optimagic aims to be a drop-in replacement for
`scipy.optimize`. An option of a scipy optimizer gets a readable optimagic name like any
other option. The scipy name has to keep working in addition, as a permanent alias
without a warning. The mechanism for these aliases is tracked in
[issue 745](https://github.com/optimagic-dev/optimagic/issues/745).

(wrapper-standards-defaults)=

### Defaults

- Every option has a default. Users can run the algorithm without setting anything and
  without importing anything from the library.
- Use the default of the library, with two exceptions: the harmonized stopping and
  convergence options, and cases where a benchmark shows that another value is better.
  State every deviation and its reason in the docstring of the option.
- If the default of the library depends on the problem, for example on the number of
  parameters or on whether the function is noisy, use `None`, do not pass the option to
  the library and state the rule in the docstring. Leaving the option out is not the
  same as passing `None`: some libraries only choose a default themselves if the option
  is absent.
- Annotate each option with the range the library accepts, for example `PositiveInt` or
  `ProbabilityFloat`. Invalid values then raise when the algorithm is created.

**Convergence criteria:** At the defaults, an optimizer should stop because it has
reached an optimum, not because progress became slow. Criteria that measure closeness to
an optimum are the main criteria. These are the gradient norm or, for derivative-free
trust-region methods, the trust-region radius. Criteria that only measure progress, such
as the change in the function value or in the parameters or a count of slow iterations,
stay switched on as a safety net. Their defaults have to be strict enough that they do
not cost accuracy. Check this by running a benchmark set at the defaults and looking at
which criterion ends the runs and how close they get to the best known value.

(wrapper-standards-result)=

### The result

A wrapper reports as many fields of `InternalOptimizeResult` as it can: the parameters
and the function value, `success`, `message` and `status`, the numbers of function
evaluations, derivative evaluations and iterations, and the derivatives at the solution
if the library provides them. Everything else the library reports goes into `info`.

- If the library does not count something, count it in the wrapper where this is
  possible, for example by counting the calls of the function.
- If collecting a value costs time, collect it anyway unless the cost is large compared
  to a cheap function evaluation, and switch off the expensive parts that are not
  needed.
- Say in the docstring of the class if a reported derivative is an approximation.
- A value that an existing wrapper reported is still reported after a change.

(wrapper-standards-deprecations)=

### Changing an existing wrapper

Every change that users can notice needs a deprecation cycle. Old code keeps working for
one minor release and emits a `FutureWarning` that says what to use instead and in which
version the old behavior is removed. Changes are made in place; we do not add a second
wrapper for the same algorithm under a new name.

**Renamed, moved or converted options** are declared next to the algorithm and passed to
the `mark.deprecated_options` decorator, which is placed above `mark.minimizer`:

```python
from optimagic.optimization.algo_option_tools import DeprecatedOption

DEPRECATED_OPTIONS = (
    # a new name
    DeprecatedOption("stopping_max_seconds", "stopping_maxtime", "0.7.0"),
    # moved into a group
    DeprecatedOption("n_restarts", "restarts.max_restarts", "0.7.0"),
    # a value that has to be converted, or an option that is split up
    DeprecatedOption(
        "reset_type",
        lambda value, options: {"restarts.use_soft_restarts": value == "soft"},
        "0.7.0",
    ),
)


@mark.deprecated_options(DEPRECATED_OPTIONS)
@mark.minimizer(...)
@dataclass(frozen=True)
class MyAlgorithm(Algorithm): ...
```

The old names then work wherever options can be set, i.e. when the algorithm is created,
in `with_option` and in `algo_options`. They do not appear in the signature or in the
documentation. Combining an old name with the option or the group that replaces it
raises an error.

A function that converts a value is called with the old value and with all options that
were passed in the same call. Use the second argument if the old option only had an
effect together with another one. An old option that moves into an option of the form
`bool | <group>` switches the feature on. To switch it off, let the function return the
option itself, for example `{"restarts": False}`.

**Changed defaults:** Give the option the default `None` for one release and resolve it
with `resolve_changed_default` when the optimization starts. Users who did not set the
option keep the old default and get a warning that announces the new one. Do not warn in
situations in which the option has no effect.

The version in the warnings is the minor release after the one that ships the change.

(wrapper-standards-tests)=

### Tests

Every wrapper has two sets of tests.

**A few fast tests that run by default.** All algorithms are tested on simple problems
in `test_many_algorithms.py`. Add a small file with tests for everything this does not
cover, for example that deprecated names work.

**Wrapper tests** check every option. They are in
`tests/optimagic/optimizers/test_<library>_wrappers.py` and carry the marker
`<library>_wrappers`. Tests with such a marker only run if they are selected:

```console
$ pixi run tests -m fides_wrappers    # one library
$ pixi run tests -m wrappers          # all libraries
```

For every pull request, a separate job runs all wrapper tests on Linux with the newest
supported Python version. In addition, the `wrapper-tests` workflow runs them once a
month against the newest releases of the wrapped libraries and can be started by hand.
The wrapper tests consist of:

- **Contract tests** that detect changes in the library. They compare the options the
  library offers (found by introspection) with tables in the wrapper module: one that
  maps library names to optimagic names, one with the options that are not exposed and
  one with the defaults that deviate on purpose. The wrapper uses the same tables to
  build its call, so that tests and wrapper cannot diverge. Also compare signatures and
  sets of valid choices.
- **Behavior tests** that run `om.minimize` on a tiny problem and show that an option
  has its intended effect: a looser tolerance needs fewer iterations, the first step has
  the length of the initial radius, a limit ends the run with the right message.
- **Forwarding tests** for options whose effect cannot be observed on a tiny problem.
  They replace the library function by a spy and check that the value arrives under the
  right name, and that options with the default `None` are not passed.
- **Deprecation tests** for every deprecated name and every changed default.

Choose problems with a handful of parameters and fixed seeds. The wrapper tests of one
library should run in well under a minute.

(wrapper-standards-docs)=

### Documentation

All documentation of an algorithm is in the docstrings of its class and its options, and
of its option groups. [How to document optimizers](how_to_document_optimizers.md)
explains what to write and how the docstrings are rendered.

(wrapper-standards-tutorials)=

### Tutorials

A wrapper that accepts objects from the library needs a tutorial notebook in
`docs/source/tutorials`. For other highly configurable optimizers, for example those
with several option groups, a tutorial is recommended.

A tutorial explains why and how to use the advanced options. Each section is either a
situation in which the defaults struggle and an option solves the problem, or an
illustration of what is possible and how it is done. Only keep examples in which you can
reproduce the effect. The [fides tutorial](fides_tutorial) is an example.

(wrapper-integration)=

## Integrate the wrapper into optimagic

1. Add the library to `[tool.pixi.pypi-dependencies]` or `[tool.pixi.dependencies]` in
   `pyproject.toml` and run `pixi install` to update the lock file. Use the newest
   release as the minimal version. If the library does not work on all platforms, reach
   out to a maintainer.
1. Put the wrapper into a module in `src/optimagic/optimizers`. Use the existing module
   if we already wrap other optimizers from the library.
1. Run `prek run --all-files`. This also generates the code that makes the algorithm
   available as `om.algos.<name>`.
1. Add the tests. Register the marker `<library>_wrappers` in `pyproject.toml` and add
   the library to `WRAPPED_PACKAGES` in `.github/workflows/wrapper-tests.yml`.
1. Add the algorithm to `docs/source/algorithms.md` as described in
   [how to document optimizers](how_to_document_optimizers.md).
1. Add an entry to `CHANGES.md` and create a pull request. Put the final option table
   into its description.
