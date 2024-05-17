(eeppytrees)=

# EEP-02: Static typing

```{eval-rst}
+------------+------------------------------------------------------------------+
| Author     | `Janos Gabler <https://github.com/janosg>`_                      |
+------------+------------------------------------------------------------------+
| Status     | Draft                                                            |
+------------+------------------------------------------------------------------+
| Type       | Standards Track                                                  |
+------------+------------------------------------------------------------------+
| Created    | 2024-05-02                                                       |
+------------+------------------------------------------------------------------+
| Resolution |                                                                  |
+------------+------------------------------------------------------------------+
```

## Abstract

This enhancement proposal explains how we want to adopt static typing in estimagic. The
overarching goals of the proposal are the folloing:

- More robust code due to static type checking and use of stricter types in internal
  functions.
- Better readability of code due to type hints
- Better discoverability and autocomplete for users of estimagic

Achieving these goals requires more than adding type hints. Estimagic is currently
mostly [stringly typed](https://wiki.c2.com/?StringlyTyped) and full of dictionaries
with a fixed set of required keys (e.g.
[constraints](https://estimagic.readthedocs.io/en/latest/how_to_guides/optimization/how_to_specify_constraints.html),
[option dictionaries](https://estimagic.readthedocs.io/en/latest/how_to_guides/optimization/how_to_specify_algorithm_and_algo_options.html),
etc.).

This enhancement proposal outlines how we can accomodate the changes needed to reap the
benefits of static typing without breaking users' code in too many places.

## Motivation and ressources

- [Writing Python like it's Rust](https://kobzol.github.io/rust/python/2023/05/20/writing-python-like-its-rust.html).
  A very good blogpost that summarizes the drawbacks of "stringly-typed" Python code and
  shows how to incorporate typing philosophies from Rust into Python projects. Read this
  if you don't have time to read the other ressources.
- [Robust Python](https://www.oreilly.com/library/view/robust-python/9781098100650/), an
  excellent book that discusses how to design code around types and provides an
  introduction to static type checkers in Python.
- A
  [jax enhancement proposal](https://jax.readthedocs.io/en/latest/jep/12049-type-annotations.html)
  for adopting static typing. It has a very good discussion on benefits of static
  typing.
- [Subclassing in Python Redux](https://hynek.me/articles/python-subclassing-redux/)
  explains which types of subclassing are considered harmful and was very helpful for
  designing this proposal.

## Changes for optimization

The following changes apply to all functions that are directly related to optimization,
i.e. `maximize`, `minimize`, `slice_plot`, `criterion_plot`, `params_plot`,
`count_free_params`, `check_constraints` and `OptimizeResult`.

### The `criterion` function

#### Current situation

A function that takes params (a pytree) as first argument and returns a scalar (if only
scalar algorithms will be used) or a dictionary that contains the entries "value" (a
scalar float), "contributions" (a pytree containing the summands that make up the
criterion value of a likelihood problem) or "root_contributions" (a pytree containing
the residuals of a least-squares problem). Moreover, the dict can have any number of
additional entries. The additional dict entries will be stored in a database if logging
is used.

A few simple examples of valid criterion functions are:

```python
def sphere(params: np.ndarray) -> float:
    return params @ params


def dict_sphere(params: dict) -> float:
    return params["a"] ** 2 + params["b"] ** 2


def least_squares_sphere(params: np.ndarray) -> dict[str, Any]:
    out = {"root_contributions": params, "p_mean": params.mean(), "p_std": params.std()}
    return out
```

**Things we want to keep**

- The fact that `params` can be arbitrary pytrees makes estimagic flexible and popular.
  We do not need to restrict this type in any way because flattening the pytree gives us
  a very precise type no matter how complicated the tree was.
- We do not need to restrict the type of additional arguments of the criterion function.
- The simplest form of our criterion functions is also compatible with scipy.optimize

**Problems**

- Newcomers find it hard to specify least-squares problems
- Internally we can make almost no assumptions about the output of a criterion function,
  making the code very complex and brittle
- The best typing information we could get for the output of the criterion function is
  `float | dict[str, Any]` which is not very useful.

#### Proposal

`params` and additional keyword arguments of the criterion function stay unchanged. The
output of the criterion function becomes `float | CriterionValue`. There are decorators
that help the user to write valid criterion functions without making an explicit
instance of `CriterionValue`.

The first two previous examples remain valid. The third one will be deprecated and
should be replaced by:

```python
def least_squares_sphere(params: np.ndarray) -> em.CriterionValue:
    out = CriterionValue(
        residuals=params, info={"p_mean": params.mean, "p_std": params.std()}
    )
    return out
```

We can exploit this deprecation cycle to rename `root_contributions` to `residuals`
which is more in line with the literature.

If a user only wants to express the least-squares structure of the problem without
logging any additional information, they can use a decorator to simplify things:

```python
@em.mark.least_squares
def decorated_sphere(params: np.ndarray) -> np.ndarray:
    return params
```

In this last syntax, the criterion function is implemented the same way as in existing
least-squares optimizers (e.g. DFO-LS), which will make it very easy for new users of
estimagic. Similarly, `em.mark.likelihood` will be available for creating criterion
functions that are compatible with the BHHH optimizer.

Since there is no need to modify instances of CriterionValue, it should be immutable.

### Bundling bounds

#### Current situation

Currently we have four arguments of `maximize`, `minimize` and related functions that
let the user specify bounds:

```python
em.minimize(
    # ...
    lower_bounds=params - 1,
    upper_bounds=params + 1,
    soft_lower_bounds=params - 2,
    soft_lower_bounds=params + 2,
    # ...
)
```

Each of them is a pytree that mirrors the structure of params or None

**Problems**

- Usually all of these arguments are used together and passing them around individually
  is annoying
- The names are very long because the word `bounds` is repeated

#### Proposal

We bundle the bounds together in a `Bound` type:

```python
bounds = em.Bounds(
    lower=params - 1,
    upper=params + 1,
    soft_lower=params - 2,
    soft_lower=params + 2,
)
em.minimize(
    # ...
    bounds=bounds,
    # ...
)
```

As a bonus feature, the `Bounds` type can do some checks on the bounds at instance
creation time such that users get errors before running an optimization.

Using the old arguments will be deprecated.

Since there is no need to modify instances of Bounds, it should be immutable.

### Constraints

#### Current situation

Currently, constraints are dictionaries with a set of required keys. The exact
requirements depend on the type of constraint and even on the structure of params.

Each constraint needs a way to select the parameters to which the constraint applies.
There are three dictionary keys for this:

- "loc", which works if params are numpy arrays, pandas.Series or pandas.DataFrame
- "query", which works only if params are pandas.DataFrame
- "Selector", which works for all valid formats of params

Moreover, each constraint needs to specify its type using the "type" key.

Some constraints have additional required keys:

- linear constraints have "weights", "lower_bound", "upper_bound", and "value"
- nonlinear constraints have "func", "lower_bounds", "upper_bound", and "value"

Details and examples can be found
[here](https://estimagic.readthedocs.io/en/latest/how_to_guides/optimization/how_to_specify_constraints.html)

**Things we want to keep**

- The constraints interface is very declarative; Constraints purely collect information
  and are completely separate from the implementation.
- All three ways of selecting parameters have their strength and can be very concise and
  readable in specific applications.

**Problems**

- Constraints are hard to document and generally not understood by most users
- Having multiple ways of selecting parameters (not all compatible with all params
  formats) is confusing for users and annoying when processing constraints. We have to
  handle the case where no selection or multiple selections are specified.
- Dicts with required keys are brittle and do not provide autocomplete. This is made
  worse by the fact that each type of constraint requires different sets of keys.

#### Proposal

1. We implement simple dataclasses for each type of constraint
1. We get rid of `loc` and `query` as parameter selection methods. Instead we show in
   the documentation how both selection methods can be used inside a `selector`
   function.

Examples of the new syntax are:

```python
constraints = [
    em.constraints.FixedConstraint(selector=lambda x: x[0, 5]),
    em.constraints.IncreasingConstraint(selector=lambda x: x[1:4]),
]

res = em.minimize(
    criterion=criterion,
    params=np.array([2.5, 1, 1, 1, 1, -2.5]),
    algorithm="scipy_lbfgsb",
    constraints=constraints,
)
```

Since there is no need to modify instances of constraints, they should be immutable.

All constraints can subclass `Constraint` which will only have the `selector` attribute.
During the deprecation phase, `Constraint` will also have `loc` and `query` attributes.

The current `cov` and `sdcorr` constraints apply to flattened covariance matrices as
well as standard deviations and flattened correlation matrices. This comes from a time
where estimagic only supported an essentially flat parameter format (DataFrames with
"value" column). We can exploit the fact that we already have breaking changes to rename
the current `cov` and `sdcorr` constraints to `FlatCovConstraint` and
`FlatSdcorrConstraint`. This prepares the introduction of a more natural `CovConstraint`
and `SdcorrConstraint` later.

(algorithm-selection)=

### Algorithm selection

#### Current situation

`algorithm` is a string or a callable that satisfies the internal algorithm interface.
If the user passes a string, we look up the algorithm implementation in a dictionary
containing all installed algorithms. We implement suggestions for typical typos based on
fuzzy matching of strings.

**Problems**

- There is no autocomplete
- It is very easy to make typos and they only get caught at runtime
- Users cannot select algorithms without reading the documentation

#### Difficulties

The usual solution to selecting algorithms in an autocomplete friendly way is an Enum.
However, there are two difficulties that make this solution suboptimal:

1. The set of available algorithms depends on the packages a user has installed. Almost
   all algorithms come from optional dependencies and very few users install all
   optional dependencies.

1. We already have more than 50 algorithms and plan to add many more. A simple
   autocomplete is not very helpful. Instead the user would have to be able to filter
   the autocomplete results according to the problem properties (e.g. least-squares,
   gradient-based, local, ...). However, it is not clear which filters are relevant and
   in which order a user wants to apply them.

#### Proposal

We continue to support passing algorithms as strings. This is important because
estimagic promises to work "just like scipy" for simple things. On top, we offer a new
way of specifying algorithms that is less prone to typos, supports autocomplete and will
be useful for advanced algortihm configuration.

To exemplify the new approach, assume a simplified situation with 5 algorithms. We only
consider whether an algorithm is gradient free or gradient based. One algorithm is not
installed, so should never show up anywhere. Here is the ficticious list:

- neldermead: installed, gradient_free
- bobyqa: installed, gradient_free
- lbfgs: installed, gradient_based
- slsqp: installed, gradient_based
- ipopt: not installed, gradient_based

We want the following behavior:

The user types `em.algorithms.` and autocomplete shows

- GradientBased
- GradientFree
- neldermead
- bobyqa
- lbfgs
- slsqp

A user can either select one of the algorithms (lowercase) directly or filter the
further by selecting a category (CamelCase). This would look as follows:

The user types `em.algorithms.GradientFree.` and autocomplete shows

- neldermead
- bobyqa

Once the user arrives at an algorithm, a subclass of `Algorithm` is returned. This class
will be passed to `minimize` or `maximize`. Passing configured instances of an
`Algorithm`s will be discussed in [Algorithm Options](algorithm-options).

In practice we would have a lot more algorithms and a lot more categories. Some
categories might be mutually exclusive, in that case the second category is omitted
after the first one is selected.

We have the following categories:

- GradientBased vs. GradientFree
- Local vs. Global
- Bounded vs. Unbounded
- Scalar vs. LeastSquares vs. Likelihood
- LinearConstrained vs. NonlinearConstrained vs. Unconstrained

Potentially, we could also offer a `.All` attribute that returns a list of all currently
selected algorithms. That way a user could for example loop over all Bounded and
GradientBased LeastSquares algorithms and compare them in a criterion plot.

These categories match nicely with our
[algorithm selection tutorials](https://effective-programming-practices.vercel.app/scientific_computing/optimization_algorithms/objectives_materials.html)

We can use
[`dataclasses.make_dataclass`](https://docs.python.org/3/library/dataclasses.html#dataclasses.make_dataclass)
to programatically build up a data structure with the autocomplete behavior described
above. `make_dataclass` also supports type hints.

```{note}
The first solution I found when
playing with this is eager, i.e. the complete data structure is created created at
import time, no matter what the user does. A lazy solution where only the branches of
the data structure we need are created. Maybe, this can be achieved with properties
but I don't know yet how easy that is to add properties via `make_dataclass`
```

(algorithm-options)=

### Algorithm options

Algorithm options refer to options that are not handled by estimagic but directly by the
algorithms. Examples are convergence criteria, stopping criteria and advanced
configuration of algorithms. Some of them are supported by many algorithms (e.g.
stopping after a maximum number of function evaluations is reached), some are supported
by certain classes of algorithms (e.g. most genetic algorithms have a population size,
most trustregion algorithms allow to set an initial trustregion radius) and some of them
are completely specific to one algorithm (e.g. ipopt has more than 100 very specific
options, nag_dfols supports very specific restarting strategies, ...).

While nothing can be changed about the fact that every algorithm supports different
options (e.g. there is simply no trustregion radius in a genetic algorithm), we go very
far in harmonizing algo options across optimizers:

1. Options that are the same in spirit (e.g. stop after a specific number of iterations)
   get the same name across all optimizers wrapped in estimagic. Most of them even get
   the same default value.
1. Options that have undescriptive (and often heavily abbreviated) names in their
   original implementation get more readable names, even if they appear only in a single
   algorithm.
1. Options that are specific to a well known optimizer (e.g. ipopt) are not renamed

#### Current situation

The user passes `algo_options` as a dictionary of keyword arguments. All options that
are not supported by the selected algorithm are discarded with a warning. The names of
most options is very descriptive (even though a bit too long at times).

We implement basic namespaces by introducing a dot notation. Example:

```python
options = {
    "stopping.max_iterations": 1000,
    "stopping.max_criterion_evaluations": 1500,
    "convergence.relative_criterion_tolerance": 1e-6,
    "convergence.scaled_gradient_tolerance": 1e-6,
    "initial_radius": 0.1,
    "population_size": 100,
}
```

The option dictionary is then used as follows:

```python
minimize(
    # ...
    algorithm="scipy_lbfgsb",
    algo_options=options,
    # ...
)
```

In the example, only the options `stopping.max_criterion_evaluations`,
`stopping.max_iterations` and `convergence.relative_criterion_tolerance` are supported
by `scipy_lbfgsb`. All other options would be ignored.

**Things we want to keep**

- Mixing the options for all optimizers in a single dictionary and discarding options
  that do not apply to the selected optimizer allows to loop very efficiently over very
  different algorithms (without if conditions in the user's code). This is very good for
  quick experimentation, e.g. solving the same problem with three different optimizers
  and limiting each optimizer to 100 function evaluations.
- The basic namespaces help to quickly see what is influenced by a specific option. This
  works especially well to distinguish stopping options and convergence criteria from
  other tuning parameters of the algorithms.
- All options are documented in the estimagic documentation, i.e. we do not link to the
  docs of original packages.

**Problems**

- There is no autocomplete and the only way to find out which options are supported is
  the documentation.
- A small typo in an option name can easily lead to the option being discarded
- Option dictionaries can grow very big
- Only the namespaces for stopping and convergence work really well, everything else is
  too different across optimizers.
- The fact that option dictionaries are mutable can lead to errors, for example when a
  user wants to try out a grid of values for one tuning parameter while keeping all
  other options constant.

**Secondary problems**

The following problems are not related to the specific goals of this enhancement
proposal but it might be a good idea to address them in the same deprecation cycle.

- In an effort to make everything very descriptive, some names got too long. For example
  `"convergence.absolute_gradient_tolerance"` is very long but most people are so
  familiar with reading `"gtol_abs"` (from scipy and nlopt) that
  `"convergence.gtol_abs"` would be a better name.
- It might have been a bad idea to harmonize default values for similar options that
  appear in multiple optimizers. Sometimes the options, while similar in spirit, are
  defined slightly differently and usually algorithm developers will set all tuning
  parameters to maximize performance on a benchmark set they care about. If we change
  how options are handled in estimagic, should consider to just harmonize names and not
  default values.

#### Proposal

In total we want to offer three entry points for the configuration of optimizers:

1. Instead of passing an `Algorithm` class (as described in
   [Algorithm Slection](algorithm-selection)) the user can create an instance of their
   selected algorithm. When creating the instance, they have autocompletion for all
   options supported by the selected algorithm. `Algorithm`s are immutable.
1. Given an instance of an `Algorithm`, a user can easily create a modified copy of that
   instance by using the following methods:

- `with_stopping`: To modify the stopping criteria
- `with_convergence`: To modify the convergence criteria
- `with_option`: To modify other algorithm options.

3. As before, the user can pass a global set of options to `maximize` or `minimize`. We
   continue to support option dictionaries but also allow `AlgorithmOption` objects that
   enable better autocomplete and immutability. Global options override the options that
   were directly passed to an optimizer. For consistency, `AlgorithmOptions` can offer
   the `with_stopping`, `with_convergence` and `with_option` methods.

The previous example continues to work. Examples of the new possibilities are:

```python
# configured algorithm
algo = em.algorithms.scipy_lbfgsb(
    stopping_max_iterations=1000,
    stopping_max_criterion_evaluations=1500,
    convergence_relative_criterion_tolerance=1e-6,
)
minimize(
    # ...
    algorithm=algo,
    # ...
)
```

```python
# using copy constructors for better namespaces
algo = (
    em.algorithms.scipy_lbfgsb()
    .with_stopping(
        max_iterations=1000,
        max_criterion_evaluations=1500,
    )
    .with_convergence(
        relative_criterion_tolerance=1e-6,
    )
)

minimize(
    # ...
    algorithm=algo,
    # ...
)
```

```python
# using copy constructors to create variants
base_algo = em.algorithms.fides(stopping_max_iterations=1000)
algorithms = [base_algo.with_option(initial_radius=r) for r in [0.1, 0.2, 0.5]]

for algo in algorithms:
    minimize(
        # ...
        algorithm=algo,
        # ...
    )
```

```python
# option object
options = em.AlgorithmOptions(
    stopping_max_iterations=1000,
    stopping_max_criterion_evaluations=1500,
    convergence_relative_criterion_tolerance=1e-6,
    convergence_scaled_gradient_tolerance=1e-6,
    initial_radius=0.1,
    population_size=100,
)


minimize(
    # ...
    algorithm=em.algorithms.scipy_lbfgsb,
    algo_options=options,
    # ...
)
```

The implementation of this behavior is not trivial since the signatures across several
methods need to be aligned. This introduces code duplication unless helpers like
`functools.wraps` or dynamic signature creation as in
[dags](https://github.com/OpenSourceEconomics/dags/blob/main/src/dags/signature.py) are
used. Unfortunately, those helpers break autocompletion in Vscode and Pycharm. A
possible way out is dynamic creation of algorithm classes. This is also necessary to
preserve backwards compatibility with the current internal algorithm interface. The
discussion is continued in
[The internal algorithm interface and \`Algorithm objects](algorithm-interface)

### Custom derivatives

#### Current situation

**Things we want to keep** **Problems**

#### Proposal

### Other option dictionaries

#### Current situation

**Things we want to keep** **Problems**

#### Proposal

(algorithm-interface)=

### The internal algorithm interface and `Algorithm` objects

#### Current situation

Currently, algorithms are defined as `minimize` functions that are decorated with
`em.mark_minimizer`. The `minimize` function returns a dictionary with a few mandatory
and several optional keys. Algorithms can provide information to estimagic in two ways:

1. The signature of the minimize function signals whether the algorithm needs
   derivatives and whether it supports bounds and nonlinear constraints. Moreover, it
   signals which algorithm specific options are supported. Default values for algorithm
   specific options are also defined in the signature of the minimize function.
1. `@mark_minimizer` collectn the following information via keyword arguments

- Is the algorithm a scalar, least-squares or likelihood optimizer?
- The algorithm name
- Does the algorithm require well scaled problems
- Is the algortihm currently installed
- Is the algorithm global or local
- Should the history tracking be disabled (e.g. because the algorithm tracks its own
  history)
- Does the algorithm parallelize criterion evaluations

A slightly simplified example of the current internal algorithm interface is:

```python
@mark_minimizer(name="scipy_neldermead", needs_scaling=True)
def scipy_neldermead(
    criterion,
    x,
    lower_bounds,
    upper_bounds,
    *,
    stopping_max_iterations=1_000_000,
    stopping_max_criterion_evaluations=1_000_000,
    convergence_absolute_criterion_tolerance=1e-8,
    convergence_absolute_params_tolerance=1e-8,
    adaptive=False,
):
    pass
```

The first two arguments (`criterion` and `x`) are mandatory. The lack of any arguments
related to derivatives signifies that `scipy_neldermead` is a gradient free algorithm.
The bounds show that it supports box constraints. The remaining arguments define the
supported stoppin criteria and algorithm options as well as their default values.

The decorator simply attaches information to the function as `_algorithm_info`
attribute. This originated as a hack but was never changed afterwards. The
`AlgorithmInfo` looks as follows:

```python
class AlgoInfo(NamedTuple):
    primary_criterion_entry: str
    name: str
    parallelizes: bool
    needs_scaling: bool
    is_available: bool
    arguments: list  # this is read from the signature
    is_global: bool = False
    disable_history: bool = False
```

**Things we want to keep**

- Writing `minimize` functions is very simple in many cases we only need minimal
  wrappers around optimizer libraries.
- The internal interface has proven flexible enough for many optimizers we had not
  wrapped when we designed it. It is easy to add more optional arguments to the
  decorator without breaking any existing code.
- The decorator approach completely hides how we represent algorithms internally
- Since we read a lot of information from function signatures (as opposed to registering
  options somewhere) there is no duplicated information.

**Problems**

- Type checkers complain about the `._algorithm_info` hack
- A function with attached `._algorithm_info` is not a good internal representation
- All computations and signature checking are done eagerly for all algorithms at import
  time. This is one of the reasons why imports are slow.
- The first few arguments to the minimize functions follow a naming scheme and any typo
  in those names would lead to situations that are hard to debug (e.g. if `lower_bound`
  was misstyped as `lower_buond` we would assume that the algorithm does not support
  lower bounds but has a tuning parameter called `lower_buond`)

#### Proposal

The primary changes to the internal algorithm interface are:

1. The minimize function takes an instance of `em.InternalProblem` as first argument.
   `em.InternalProblem` bundles all problem specific arguments like `criterion`,
   `lower_bounds`, `upper_bounds`, `derivative`, `criterion_and_derivative`, and
   `nonlinear_constraints`. This reduces the potential for typos but means that all
   information we read from the presence or absence of those arguments needs to be
   passed into the `mark.minimizer` decorator instead. Therefore, `mark.minimizer` gets
   the following new arguments:

- supports_bounds: bool
- supports_nonlinear_constraints: bool
- supports_linear_constraints: bool (currently not used, but we should add it)
- needs_derivative: bool This is a breaking change that we will implement without
  deprecation cycle. There are probably very few users of estimagic who use custom
  algorithms and are affected by this.

2. The minimize function returns an `InternalOptimizeResult` instead of a dictionary. We
   can use a deprecation cycle for this.
1. Instead of simply attaching information to a function, the `mark.minimizer` decorator
   dynamically creates a suclass of `em.Algorithm`. This class contains all information
   that is currently stored in `._algorithm_info` and supports the configuration methods
   `with_stopping`, `with_convergence` and `with_option`. This is not a breaking change
   as it was never documented what `mark_minimizer` does.

```{note}
This proposal violates several best practices of object oriented programming:
1. Classes are dynamically generated, which will probably require the use of `exec`.
It seems justifiable since it gets rid of all duplication in the signatures of
`__init__` and the `with_...` methods and thus produces a very nice autocomplete
behavior.
2. The `with_...` methods have different signatures across subclasses and thus
violate the Liskov Substitution Principle. However, it seems justifiable since they are
basically constructors.
3. `InternalProblem` is a relative imprecise type as it needs to work for all problems
that can be solved by estimagic (constrained vs. unconstrained, least-squares vs. scalar,
...). We could work with a Class hierarchy here to get better static information, but
opt against it because it would complicate the internal algorithm interface.
```

```{note}
We can use this deprecation cycle to get rid of `primary_criterion_entry` (which could
take the values "value", "contributions" or "root_contributions") and use
`problem_type` instead, which can take the values `em.ProblemType.scalar`,
`em.ProblemType.least_squares` and `em.ProblemType.likelihood`.
```

```{note}
Should we continue to infer `parallelizes` from the presence of `batch_evaluator` in
the function signature?
```

Here are a few code snippets to make things concrete. The `InternalProblem` will be an
immutable dataclass that looks as follows:

```python
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Callable, Tuple
import estimagic as em


@dataclass(frozen=True)
class ScalarProblemFunctions:
    f: Callable[[NDArray[float]], float]
    jac: Callable[[NDArray[float]], NDArray[float]]
    f_and_jac: Callable[[NDArray[float]], Tuple[float, NDArray[float]]]


@dataclass(frozen=True)
class LeastSquaresProblemFunctions:
    f: Callable[[NDArray[float]], NDArray[float]]
    jac: Callable[[NDArray[float]], NDArray[float]]
    f_and_jac: Callable[[NDArray[float]], Tuple[NDArray[float], NDArray[float]]]


@dataclass(frozen=True)
class LikelihoodProblemFunctions:
    f: Callable[[NDArray[float]], NDArray[float]]
    jac: Callable[[NDArray[float]], NDArray[float]]
    f_and_jac: Callable[[NDArray[float]], Tuple[NDArray[float], NDArray[float]]]


@dataclass(frozen=True)
class InternalProblem:
    scalar: ScalarProblemFunctions
    least_squares: LeastSquaresProblemFunctions
    likelihood: LikelihoodProblemFunctions
    bounds: em.Bounds | None
    linear_constraints: list[em.LinearConstraint] | None
    nonlinear_constraints: list[em.NonlinearConstraint] | None
```

The `InternalOptimizeResult` formalizes the current dictionary solution:

```python
@dataclass(frozen=True)
class InternalOptimizeResult:
    solution_x: NDArray[float]
    solution_criterion: float
    n_criterion_evaluations: int | None
    n_derivative_evaluations: int | None
    n_iterations: int | None
    success: bool | None
    message: str | None
```

Explicitly writing out all optional arguments of `mark.minimizer`, the wrapper for
`scipy_neldermead` changes to:

```python
@em.mark.minimizer(
    name="scipy_neldermead",
    problem_type=em.ProblemType.scalar,
    needs_scaling=True,
    is_available=IS_SCIPY_AVAILABLE,
    is_global=False,
    disable_history=False,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    needs_derivative=False,
)
def scipy_neldermead(
    internal_problem: InternalProblem,
    x: NDArray[float],
    stopping_max_iterations=1_000_000,
    stopping_max_criterion_evaluations=1_000_000,
    convergence_absolute_criterion_tolerance=1e-8,
    convergence_absolute_params_tolerance=1e-8,
    adaptive=False,
) -> InternalOptimizeResult:
    pass
```

The abstract base class `Algorithm` is:

```python

```

If we were to define the dynamically generated `ScipyNeldermead` class ourselves, the
code would roughly look as follows:

```python

```

## Numerical differentiation

#### Current situation

**Things we want to keep** **Problems**

#### Proposal

## Benchmarking

### Benchmark problems

#### Current situation

**Things we want to keep** **Problems**

#### Proposal

### Benchmark results

#### Current situation

**Things we want to keep** **Problems**

#### Proposal

## Internal changes

### Internal criterion and derivative

## Type checkers and their configuration

## Runtime type checking

Since most of our users do not use static type checkers we will still need to check the
type of most user inputs so we can give them early feedback when problems arise.

## Summary of design philosophy

## Changes in documentation

- No type hints in docstrings anymore
- Only show new recommended ways of doing things, not deprecated ones

## Breaking changes

## Summary of deprecations

The following deprecations become active in version `0.5.0`. The functionality will be
removed in version `0.6.0` which should be scheduled for approximately half a year after
the realease of `0.5.0`.

- Returning a `dict` in the `criterion` function io deprecated. Return `CriterionValue`
  instead or use `em.mark.least_squares` or `em.mark.likelihood` to create your
  criterion function.
- The arguments `lower_bounds`, `upper_bounds`, `soft_lower_bounds` and
  `soft_uppper_bounds` are deprecated. Use `bounds` instead.
- Selecting an algorithm by strings is deprecated. Pass an `Algorithm` instead.
