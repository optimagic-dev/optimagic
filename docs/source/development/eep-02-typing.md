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
overarching goals of the proposal are the following:

- Better discoverability and autocomplete for users of estimagic.
- Better readability of code due to type hints.
- More robust code due to static type checking and use of stricter types in internal
  functions.

Achieving these goals requires more than adding type hints. estimagic is currently
mostly [stringly typed](https://wiki.c2.com/?StringlyTyped) and full of dictionaries
with a fixed set of required keys (e.g.
[constraints](https://estimagic.readthedocs.io/en/latest/how_to_guides/optimization/how_to_specify_constraints.html),
[option dictionaries](https://estimagic.readthedocs.io/en/latest/how_to_guides/optimization/how_to_specify_algorithm_and_algo_options.html),
etc.).

This enhancement proposal outlines how we can accommodate the changes needed to reap the
benefits of static typing without breaking users' code in too many places.

A few deprecations and breaking changes will, however, be unavoidable. Since we are
already interrupting users, we can use this deprecation cycle as a chance to better
align some names in estimagic with SciPy and other optimization libraries where we think
it can improve the user experience. These changes will be marked as independent of the
core proposal and summarized in [aligning names](aligning-names).

## Motivation and resources

- [Writing Python like it's Rust](https://kobzol.github.io/rust/python/2023/05/20/writing-python-like-its-rust.html).
  A very good blogpost that summarizes the drawbacks of "stringly-typed" Python code and
  shows how to incorporate typing philosophies from Rust into Python projects. Read this
  if you don't have time to read the other resources.
- [Robust Python](https://www.oreilly.com/library/view/robust-python/9781098100650/), an
  excellent book that discusses how to design code around types and provides an
  introduction to static type checkers in Python.
- [jax enhancement proposal](https://jax.readthedocs.io/en/latest/jep/12049-type-annotations.html)
  for adopting static typing. It has a very good discussion on benefits of static
  typing.
- [Subclassing in Python Redux](https://hynek.me/articles/python-subclassing-redux/)
  explains which types of subclassing are considered harmful and was very helpful for
  designing this proposal.

## Changes for optimization

The following changes apply to all functions that are directly related to optimization,
i.e. `maximize`, `minimize`, `slice_plot`, `criterion_plot`, `params_plot`,
`count_free_params`, `check_constraints` and `OptimizeResult`.

### The objective function

#### Current situation

The objective or criterion function is the function being optimized. Currently, it is
called `criterion` in estimagic.

The `criterion` function maps a pytree of parameters into a criterion value.

An important feature of estimagic is that the same criterion function can work for
scalar, least-squares and likelihood optimizers. Moreover, a criterion function can
return additional data that is stored in the log file (if logging is active). All of
this is achieved by returning a dictionary instead of just a scalar float.

The conventions for the return value of the criterion function are as follows:

- For the simplest case, where only scalar optimizers are used, `criterion` returns a
  float or a dictionary containing the key "value" which is a float.
- For likelihood functions, `criterion` returns a dictionary that contains at least the
  key `"contributions"` which can be any pytree that can be flattened into a numpy array
  of floats.
- For least-squares problems, `criterion` returns a dictionary that contains at least
  the key `"root_contributions"` which can be any pytree that can be flattened into a
  numpy array of floats.
- In any case the returned dictionary can have an arbitrary number of additional keys.
  The corresponding information will be stored in the log database (if logging is used).

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
- Using the same criterion function for scalar, likelihood and least-squares optimizers
  makes it easy to try out and compare very different algorithms with minimal code
  changes.
- We do not need to restrict the type of additional arguments of the criterion function.
- The simplest form of our criterion functions is also compatible with scipy.optimize

**Problems**

- Most users of estimagic find it hard to write criterion functions that return the
  correct dictionary. Therefore, they don't use the logging feature and we often get
  questions about specifying least-squares problems correctly.
- Internally we can make almost no assumptions about the output of a criterion function,
  making the code that processes the criterion output very complex and full of if
  conditions.
- The best typing information we could get for the output of the criterion function is
  `float | dict[str, Any]`, which is not very useful.
- We only know if the specified criterion function is compatible with the selected
  optimizer after we evaluate it once. This means that errors for users are raised very
  late.
- While optional, in least-squares problems it is possible that a user specifies
  `root_contributions`, `contributions` and `value` even though any of them could be
  constructed out of the `root_contributions`. This means we need to check that all of
  them are compatible.
- The name `criterion` is not used very widely in the Python optimization ecosystem.

#### Proposal

`params` and additional keyword arguments of the criterion function stay unchanged. The
output of the criterion function becomes `float | CriterionValue`. There are decorators
that help the user to write valid criterion functions without making an explicit
instance of `CriterionValue`.

The first two previous examples remain valid. The third one will be deprecated and
should be replaced by:

```python
@em.mark.least_squares
def least_squares_sphere(params: np.ndarray) -> em.CriterionValue:
    out = CriterionValue(
        residuals=params, info={"p_mean": params.mean, "p_std": params.std()}
    )
    return out
```

We can exploit this deprecation cycle to rename `root_contributions` to `residuals`
which is more in line with the literature.

Since there is no need to modify instances of `CriterionValue`, it should be immutable.

If a user only wants to express the least-squares structure of the problem without
logging any additional information, they can only return the least-squares residuals as
a pytree or vector. Since the decorator was used, we know how to interpret the output.

```python
@em.mark.least_squares
def decorated_sphere(params: np.ndarray) -> np.ndarray:
    return params
```

In this last syntax, the criterion function is implemented the same way as in existing
least-squares optimizers (e.g. DFO-LS), which will make it very easy for new users of
estimagic. Similarly, `em.mark.likelihood` will be available for creating criterion
functions that are compatible with the BHHH optimizer.

For completeness, we can implement an `em.mark.scalar` decorator, but this will be
optional, i.e. if none of the decorators was used we'll assume that the problem is a
scalar problem.

```{note}
In principle, we could make the usage of the decorator optional if a `CriterionValue`
instance is returned. However, then we still would need one criterion evaluation until
we know whether the criterion function is compatible with the selected optimizer.
```

```{note}
A more modern alternative to a `mark` decorator would be to use `typing.Annotated` to
add the relevant information to the return type. However, I find that harder for
beginner users.
```

On top of the described changes, we suggest to rename `criterion` to `fun` to align the
naming with `scipy.optimize`

### Bundling bounds

#### Current situation

Currently we have four arguments of `maximize`, `minimize`, and related functions that
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

Each of them is a pytree that mirrors the structure of `params` or `None`

**Problems**

- Usually, all of these arguments are used together and passing them around individually
  is annoying.
- The names are very long because the word `bounds` is repeated.

#### Proposal

We bundle the bounds together in a `Bounds` type:

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

Since there is no need to modify instances of `Bounds`, it should be immutable.

To improve the alignment with SciPy, we can also allow users to pass a
`scipy.optimize.Bounds` object as bounds. Internally, this will be converted to our
`Bounds` object.

### Constraints

#### Current situation

Currently, constraints are dictionaries with a set of required keys. The exact
requirements depend on the type of constraints and even on the structure of `params`.

Each constraint needs a way to select the parameters to which the constraint applies.
There are three dictionary keys for this:

- `"loc"`, which works if params are numpy arrays, `pandas.Series` or
  `pandas.DataFrame`.
- `"query"`, which works only if `params` are `pandas.DataFrame`
- `"Selector"`, which works for all valid formats of `params`.

Moreover, each constraint needs to specify its type using the `"type"` key.

Some constraints have additional required keys:

- Linear constraints have `"weights"`, `"lower_bound"`, `"upper_bound"`, and `"value"`.
- Nonlinear constraints have `"func"`, `"lower_bound"`, `"upper_bound"`, and `"value"`.

Details and examples can be found
[here](https://estimagic.readthedocs.io/en/latest/how_to_guides/optimization/how_to_specify_constraints.html).

**Things we want to keep**

- The constraints interface is very declarative; Constraints purely collect information
  and are completely separate from the implementation.
- All three ways of selecting parameters have their strength and can be very concise and
  readable in specific applications.

**Problems**

- Constraints are hard to document and generally not understood by most users.
- Having multiple ways of selecting parameters (not all compatible with all `params`
  formats) is confusing for users and annoying when processing constraints. We have to
  handle the case where no selection or multiple selections are specified.
- Dicts with required keys are brittle and do not provide autocomplete. This is made
  worse by the fact that each type of constraint requires different sets of keys.

#### Proposal

1. We implement simple dataclasses for each type of constraint.
1. We get rid of `loc` and `query` as parameter selection methods. Instead, we show in
   the documentation how both selection methods can be used inside a `selector`
   function.

Examples of the new syntax are:

```python
constraints = [
    em.constraints.FixedConstraint(selector=lambda x: x[0, 5]),
    em.constraints.IncreasingConstraint(selector=lambda x: x[1:4]),
]

res = em.minimize(
    fun=criterion,
    params=np.array([2.5, 1, 1, 1, 1, -2.5]),
    algorithm="scipy_lbfgsb",
    constraints=constraints,
)
```

Since there is no need to modify instances of constraints, they should be immutable.

All constraints can subclass `Constraint` which will only have the `selector` attribute.
During the deprecation phase, `Constraint` will also have `loc` and `query` attributes.

The current `cov` and `sdcorr` constraints apply to flattened covariance matrices, as
well as standard deviations and flattened correlation matrices. This comes from a time
where estimagic only supported an essentially flat parameter format (`DataFrames` with
`"value"` column). We can exploit the current deprecation cycle to rename the current
`cov` and `sdcorr` constraints to `FlatCovConstraint` and `FlatSdcorrConstraint`. This
prepares the introduction of a more natural `CovConstraint` and `SdcorrConstraint`
later.

(algorithm-selection)=

### Algorithm selection

#### Current situation

`algorithm` is a string or a callable that satisfies the internal algorithm interface.
If the user passes a string, we look up the algorithm implementation in a dictionary
containing all installed algorithms. We implement suggestions for typical typos based on
fuzzy matching of strings.

**Problems**

- There is no autocomplete.
- It is very easy to make typos and they only get caught at runtime.
- Users cannot select algorithms without reading the documentation.

#### Difficulties

The usual solution to selecting algorithms in an autocomplete friendly way is an `Enum`.
However, there are two difficulties that make this solution suboptimal:

1. The set of available algorithms depends on the packages a user has installed. Almost
   all algorithms come from optional dependencies and very few users install all
   optional dependencies.

1. We already have more than 50 algorithms and plan to add many more. A simple
   autocomplete is not very helpful. Instead, the user would have to be able to filter
   the autocomplete results according to the problem properties (e.g. least-squares,
   gradient-based, local, ...). However, it is not clear which filters are relevant and
   in which order a user wants to apply them.

#### Proposal

We continue to support passing algorithms as strings. This is important because
estimagic promises to work "just like SciPy" for simple things. On top, we offer a new
way of specifying algorithms that is less prone to typos, supports autocomplete and will
be useful for advanced algorithm configuration.

To exemplify the new approach, assume a simplified situation with 5 algorithms. We only
consider whether an algorithm is gradient free or gradient based. One algorithm is not
installed, so should never show up anywhere. Here is the fictitious list:

- `neldermead`: installed, `gradient_free`
- `bobyqa`: installed, `gradient_free`
- `lbfgs`: installed, `gradient_based`
- `slsqp`: installed, `gradient_based`
- `ipopt`: not installed, `gradient_based`

We want the following behavior:

The user types `em.algorithms.` and autocomplete shows

- `GradientBased`
- `GradientFree`
- `neldermead`
- `bobyqa`
- `lbfgs`
- `slsqp`

A user can either select one of the algorithms (lowercase) directly or filter further by
selecting a category (CamelCase). This would look as follows:

The user types `em.algorithms.GradientFree.` and autocomplete shows

- `neldermead`
- `bobyqa`

Once the user arrives at an algorithm, a subclass of `Algorithm` is returned. This class
will be passed to `minimize` or `maximize`. Passing configured instances of `Algorithm`s
will be discussed in [Algorithm Options](algorithm-options).

In practice, we would have a lot more algorithms and a lot more categories. Some
categories might be mutually exclusive, in that case the second category is omitted
after the first one is selected.

We have the following categories:

- `GradientBased` vs. `GradientFree`
- `Local` vs. `Global`
- `Bounded` vs. `Unbounded`
- `Scalar` vs. `LeastSquares` vs. `Likelihood`
- `LinearConstrained` vs. `NonlinearConstrained` vs. `Unconstrained`

Potentially, we could also offer a `.All` attribute that returns a list of all currently
selected algorithms. That way a user could for example loop over all `Bounded` and
`GradientBased` `LeastSquares` algorithms and compare them in a criterion plot.

These categories match nicely with our
[algorithm selection tutorials](https://effective-programming-practices.vercel.app/scientific_computing/optimization_algorithms/objectives_materials.html).

We can use
[`dataclasses.make_dataclass`](https://docs.python.org/3/library/dataclasses.html#dataclasses.make_dataclass)
to programmatically build up a data structure with the autocomplete behavior described
above. `make_dataclass` also supports type hints.

```{note}
The first solution I found when
playing with this is eager, i.e. the complete data structure is created at
import time, no matter what the user does. A lazy solution where only the branches of
the data structure we need are created would be nicer. Maybe, this can be achieved with
properties, but I don't know yet how easy it is to add properties via `make_dataclass`
and whether it would break some of the autocomplete behavior we want.
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
options, `nag_dfols` supports very specific restarting strategies, ...).

While nothing can be changed about the fact that every algorithm supports different
options (e.g. there is simply no trustregion radius in a genetic algorithm), we go very
far in harmonizing `algo_options` across optimizers:

1. Options that are the same in spirit (e.g. stop after a specific number of iterations)
   get the same name across all optimizers wrapped in estimagic. Most of them even get
   the same default value.
1. Options that have non-descriptive (and often heavily abbreviated) names in their
   original implementation get more readable names, even if they appear only in a single
   algorithm.
1. Options that are specific to a well known optimizer (e.g. `ipopt`) are not renamed

#### Current situation

The user passes `algo_options` as a dictionary of keyword arguments. All options that
are not supported by the selected algorithm are discarded with a warning. The names of
most options are very descriptive (even though a bit too long at times).

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

```{note}
The `.` notation in `stopping.max_iterations` is just syntactic sugar. Internally, the
option is called `stopping_max_iterations` because all options need to be valid
Python variable names.
```

**Things we want to keep**

- Mixing the options for all optimizers in a single dictionary and discarding options
  that do not apply to the selected optimizer allows to loop very efficiently over very
  different algorithms (without `if` conditions in the user's code). This is very good
  for quick experimentation, e.g. solving the same problem with three different
  optimizers and limiting each optimizer to 100 function evaluations.
- The basic namespaces help to quickly see what is influenced by a specific option. This
  works especially well to distinguish stopping options and convergence criteria from
  other tuning parameters of the algorithms. However, it would be enough to keep them as
  a naming convention if we find it hard to support the `.` notation.
- All options are documented in the estimagic documentation, i.e. we do not link to the
  docs of original packages.

**Problems**

- There is no autocomplete and the only way to find out which options are supported is
  the documentation.
- A small typo in an option name can easily lead to the option being discarded.
- Option dictionaries can grow very big.
- The fact that option dictionaries are mutable can lead to errors, for example when a
  user wants to try out a grid of values for one tuning parameter while keeping all
  other options constant.

**Secondary problems**

The following problems are not related to the specific goals of this enhancement
proposal but it might be a good idea to address them in the same deprecation cycle.

- In an effort to make everything very descriptive, some names got too long. For example
  `"convergence.absolute_gradient_tolerance"` is very long but most people are so
  familiar with reading `"gtol_abs"` (from SciPy and NLopt) that
  `"convergence.gtol_abs"` would be a better name.
- It might have been a bad idea to harmonize default values for similar options that
  appear in multiple optimizers. Sometimes, the options, while similar in spirit, are
  defined slightly differently and usually algorithm developers will set all tuning
  parameters to maximize performance on a benchmark set they care about. If we change
  how options are handled in estimagic, we should consider just harmonizing names and
  not default values.

#### Proposal

In total, we want to offer four entry points for the configuration of optimizers:

1. Instead of passing an `Algorithm` class (as described in
   [Algorithm Selection](algorithm-selection)) the user can create an instance of their
   selected algorithm. When creating the instance, they have autocompletion for all
   options supported by the selected algorithm. `Algorithm`s are immutable.
1. Given an instance of an `Algorithm`, a user can easily create a modified copy of that
   instance by using the `with_option` method.
1. We can provide additional methods `with_stopping` and `with_convergence` that call
   `with_option` internally but provide two additional features:
   1. They validate that the option is indeed a stopping/convergence criterion.
   1. They allow to omit the `convergence_` or `stopping_` at the beginning of the
      option name and can thus reduce repetition in the option names.
1. As before, the user can pass a global set of options to `maximize` or `minimize`. We
   continue to support option dictionaries but also allow `AlgorithmOption` objects that
   enable better autocomplete and immutability. They can be constructed dynamically
   using `make_dataclass`. Global options override the options that were directly passed
   to an optimizer. For consistency, `AlgorithmOptions` can offer the `with_stopping`,
   `with_convergence` and `with_option` copy-constructors, so users can modify options
   safely.

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

```{note}
In my currently planned implementation, autocomplete will not work reliably for the
copy constructors (`with_option`, `with_stopping` and `with_convergence). The main
reason is that most editors do not play well with `functools.wraps` or any other means
of dynamic signature creation. For more details, see the discussions about the
[Internal Algorithm Interface](algorithm-interface).
```

### Custom derivatives

Providing custom derivatives to estimagic is slightly complicated because we support
scalar, likelihood and least-squares problems in the same interface. Moreover, we allow
to either provide a `derivative` function or a joint `criterion_and_derivative` function
that allow users to exploit synergies between evaluating the criterion and the
derivative.

#### Current situation

The `derivative` argument can currently be one of three things:

- A `callable`: This is assumed to be the relevant derivative of `criterion`. If a
  scalar optimizer is used, it is the gradient of the criterion value w.r.t. params. If
  a likelihood optimizer is used, it is the jacobian of the likelihood contributions
  w.r.t. params. If a least-squares optimizer is used, it is the jacobian of the
  residuals w.r.t. params.
- A `dict`: The dict must have three keys `"value"`, `"contributions"` and
  `"root_contributions"`. The corresponding values are the three callables described
  above.
- `None`: In this case, a numerical derivative is calculated.

The `criterion_and_derivative` argument exactly mirrors `derivative` but each callable
returns a tuple of the criterion value and the derivative instead.

**Things we want to keep**

- It is good that synergies between `criterion` and `derivative` can be exploited.
- The three arguments (`criterion`, `derivative`, `criterion_and_derivative`) make sure
  that every algorithm can run efficiently when looping over algorithms and keeping
  everything else equal. With SciPy's approach of setting `jac=True` if one wants to use
  a joint criterion and derivative function, a gradient free optimizer would have no
  chance of evaluating just the criterion.
- We want to support scalar, least-squares and likelihood problems in one interface.

**Problems**

- A dict with required keys is brittle
- Autodiff needs to be handled completely outside of estimagic
- The names `criterion`, `derivative` and `criterion_and_derivative` are not aligned
  with scipy and very long.

#### Proposal

1. We keep the three arguments but rename them to `fun`, `jac` and `fun_and_jac`.
1. `jac` can also be a string `"jax"` or a more autocomplete friendly enum
   `em.autodiff_backend.JAX`. This can be used to signal that the objective function is
   jax compatible and jax should be used to calculate its derivatives. In the long run
   we can add PyTorch support and more. Since this is mostly about a signal of
   compatibility, it would be enough to set one of the two arguments to `"jax"`, the
   other one can be left at `None`.
1. The dictionaries of callables get replaced by appropriate dataclasses. We align the
   names with the names in `CriterionValue` (e.g. rename `root_contributions` to
   `residuals`).

### Other option dictionaries

#### Current situation

We often allow to switch on some behavior with a bool or a string value and then
configure the behavior with an option dictionary. Examples are:

- `logging` (`str | pathlib.Path | False`) and `log_options` (dict)
- `scaling` (`bool`) and `scaling_options` (dict)
- `error_handling` (`Literal\["raise", "continue"\]`) and `error_penalty` (dict)
- `multistart` (`bool`) and `multistart_options`

Moreover we have option dictionaries whenever we have nested invocations of estimagic
functions. Examples are:

- `numdiff_options` in `minimize` and `maximize`
- `optimize_options` in `estimate_msm` and `estimate_ml`

**Things we want to keep**

- It is nice that complex behavior like logging or multistart can be switched on in
  extremely simple ways, without importing anything and without looking up supported
  options.
- The interfaces are very declarative and decoupled from our implementation.

**Problems**

- Option dictionaries are brittle and don't support autocomplete.
- It can be confusing if someone provided `scaling_options` or `multistart_options` but
  they take no effect because `scaling` or `multistart` were not set to `True`.

#### Proposal

We want to keep a simple way of enabling complex behavior (with some default options)
but get rid of having two separate arguments (one to switch the behavior on and one to
configure it). This means that we have to be generous regarding input types.

##### Logging

Currently we only implement logging via an sqlite database. All `log_options` are
specific to this type of logging. However, logging is slow and we should support more
types of logging. For this, we can implement a simple `Logger` abstraction. Advanced
users could implement their own logger.

After the changes, `logging` can be any of the following:

- `False` (or anything Falsy): No logging is used.
- A `str` or `pathlib.Path`: Logging is used at default options.
- An instance of `estimagic.Logger`. There will be multiple subclasses, e.g.
  `SqliteLogger` which allow us to switch out the logging backend. Each subclass might
  have different optional arguments.

The `log_options` are deprecated. Using dictionaries instead of `Option` objects will be
supported during a deprecation cycle.

##### Scaling, error handling and multistart

In contrast to logging, scaling, error handling and multistart are deeply baked into
estimagic's minimize function. Therefore, it does not make sense to create abstractions
for these features that would make them replaceable components that can be switched out
for other implementations by advanced users. Most of these features are already
perceived as advanced and allow for a lot of configuration.

We therefore suggest the following argument types:

- `scaling`: `bool | ScalingOptions`
- `error_handling`: `bool | ErrorHandlingOptions`
- `multistart`: `bool | MultistartOptions`

All of the Option objects are simple dataclasses that mirror the current dictionaries.
All `_options` arguments are deprecated.

##### `numdiff_options` and similar

Replace the current dictionaries by dataclasses. Dictionaries are supported during a
deprecation cycle.

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
1. `@mark_minimizer` collects the following information via keyword arguments:

- Is the algorithm a scalar, least-squares or likelihood optimizer?
- The algorithm name.
- Does the algorithm requires well scaled problems?
- Is the algorithm currently installed?
- Is the algorithm global or local?
- Should the history tracking be disabled (e.g. because the algorithm tracks its own
  history)?
- Does the algorithm parallelize criterion evaluations?

A slightly simplified example of the current internal algorithm interface is:

```python
@mark_minimizer(
    name="scipy_neldermead",
    needs_scaling=False,
    primary_criterion_entry="value",
    is_available=IS_SCIPY_AVAILABLE,
    is_global=False,
    disable_history=False,
)
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
    options = {
        "maxiter": stopping_max_iterations,
        "maxfev": stopping_max_criterion_evaluations,
        # both tolerances seem to have to be fulfilled for Nelder-Mead to converge.
        # if not both are specified it does not converge in our tests.
        "xatol": convergence_absolute_params_tolerance,
        "fatol": convergence_absolute_criterion_tolerance,
        "adaptive": adaptive,
    }

    res = scipy.optimize.minimize(
        fun=criterion,
        x0=x,
        bounds=_get_scipy_bounds(lower_bounds, upper_bounds),
        method="Nelder-Mead",
        options=options,
    )

    return process_scipy_result(res)
```

The first two arguments (`criterion` and `x`) are mandatory. The lack of any arguments
related to derivatives signifies that `scipy_neldermead` is a gradient free algorithm.
The bounds show that it supports box constraints. The remaining arguments define the
supported stopping criteria and algorithm options as well as their default values.

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

- Writing `minimize` functions is very simple and in many cases we only need minimal
  wrappers around optimizer libraries.
- The internal interface has proven flexible enough for many optimizers we had not
  wrapped when we designed it. It is easy to add more optional arguments to the
  decorator without breaking any existing code.
- The decorator approach completely hides how we represent algorithms internally.
- Since we read a lot of information from function signatures (as opposed to registering
  options somewhere), there is no duplicated information. If we change the approach to
  collecting information, we still need to ensure there is no duplication or possibility
  to provide wrong information to estimagic.

**Problems**

- Type checkers complain about the `._algorithm_info` hack.
- All computations and signature checking are done eagerly for all algorithms at import
  time. This is one of the reasons why imports are slow.
- The first few arguments to the minimize functions follow a naming scheme and any typo
  in those names would lead to situations that are hard to debug (e.g. if `lower_bound`
  was miss-typed as `lower_buond` we would assume that the algorithm does not support
  lower bounds but has a tuning parameter called `lower_buond`).

#### Proposal

We first show the proposed new algorithm interface and discuss the changes later.

```python
@em.mark.minimizer(
    name="scipy_neldermead",
    needs_scaling=False,
    problem_type=em.ProblemType.Scalar,
    is_available=IS_SCIPY_AVAILABLE,
    is_global=False,
    disable_history=False,
    needs_derivatives=False,
    needs_parallelism=False,
    supports_bounds=True,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
)
@dataclass(frozen=True)
class ScipyNelderMead(Algorithm):
    stopping_max_iterations: int = 1_000_000
    stopping_max_criterion_evaluations: int = 1_000_000
    convergence_absolute_criterion_tolerance: float = 1e-8
    convergence_absolute_params_tolerance: float = 1e-8
    adaptive = False

    def __post_init__(self):
        # check everything that cannot be handled by the type system
        assert self.convergence_absolute_criterion_tolerance > 0
        assert self.convergence_absolute_params_tolerance > 0

    def _solve_internal_problem(
        self, problem: InternalProblem, x0: NDArray[float]
    ) -> InternalOptimizeResult:

        options = {
            "maxiter": self.stopping_max_iterations,
            "maxfev": self.stopping_max_criterion_evaluations,
            "xatol": self.convergence_absolute_params_tolerance,
            "fatol": self.convergence_absolute_criterion_tolerance,
            "adaptive": self.adaptive,
        }

        res = minimize(
            fun=problem.scalar.fun,
            x0=x,
            bounds=_get_scipy_bounds(problem.bounds),
            method="Nelder-Mead",
            options=options,
        )

        return process_scipy_result(res)
```

1. The new internal algorithms are dataclasses, where all algorithm options are
   dataclass fields. This enables us to obtain information about the options via the
   `__dataclass_fields__` attribute without inspecting signatures or imposing naming
   conventions on non-option arguments.
1. The `_solve_internal_problem` method receives an instance of `InternalProblem` and
   `x0` (the start values) as arguments. `InternalProblem` collects the criterion
   function, its derivatives, bounds, etc. This again avoids any potential for typos in
   argument names.
1. The `mark.minimizer` decorator collects all the information that was previously
   collected via optional arguments with naming conventions. This information is
   available while constructing the instance of `InternalProblem`. Thus we can make sure
   that attributes that were not requested (e.g. derivatives if `needs_derivative` is
   `False`) raise an `AttributeError` if used.
1. The minimize function returns an `InternalOptimizeResult` instead of a dictionary.

The copy constructors (`with_option`, `with_convergence`, and `with_stopping`) are
inherited from `estimagic.Algorithm`. This means, that they will have `**kwargs` as
signature and thus do not support autocomplete. However, they can check that all
specified options are actually in the `__dataclass_fields__` and thus provide feedback
before an optimization is run.

All breaking changes of the internal algorithm interface are done without deprecation
cycle.

To make things more concrete, here are prototypes for components related to the
`InternalProblem` and `InternalOptimizeResult`.

```python
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Callable, Tuple
import estimagic as em


@dataclass(frozen=True)
class ScalarProblemFunctions:
    fun: Callable[[NDArray[float]], float]
    jac: Callable[[NDArray[float]], NDArray[float]]
    fun_and_jac: Callable[[NDArray[float]], Tuple[float, NDArray[float]]]


@dataclass(frozen=True)
class LeastSquaresProblemFunctions:
    fun: Callable[[NDArray[float]], NDArray[float]]
    jac: Callable[[NDArray[float]], NDArray[float]]
    fun_and_jac: Callable[[NDArray[float]], Tuple[NDArray[float], NDArray[float]]]


@dataclass(frozen=True)
class LikelihoodProblemFunctions:
    fun: Callable[[NDArray[float]], NDArray[float]]
    jac: Callable[[NDArray[float]], NDArray[float]]
    fun_and_jac: Callable[[NDArray[float]], Tuple[NDArray[float], NDArray[float]]]


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

## Numerical differentiation

#### Current situation

The following proposal applies to the functions `first_derivative` and
`second_derivative`. Both functions have an interface that has grown over time and both
return a relatively complex result dictionary. There are several arguments that govern
which entries are stored in the result dictionary.

The functions `first_derivative` and `second_derivative` allow params to be arbitrary
pytrees. They work for scalar and vector valued functions and a `key` argument makes
sure that they work for `criterion` functions that return a dict containing
`"value", `"contributions", and `"root_contributions"`.

In contrast to optimization, all pytree handling (for params and function outputs) is
mixed with the calculation of the numerical derivatives. This can produce more
informative error messages and save some memory. However it increases complexity
extremely because we can make very few assumptions on types. There are many if
conditions to deal with this situation.

The interface is further complicated by supporting Richardson Extrapolation. This
feature was inspired by [numdifftools](https://numdifftools.readthedocs.io/en/latest/)
but has not produced convincing results in benchmarks.

**Things we want to keep**

- `params` and function values can be pytrees
- support for estimagic `criterion` functions (now functions that return
  `CriterionValue`)
- Many optional arguments to influence the details of the numerical differentiation
- Rich output format that helps to get insights on the precision of the numerical
  differentiation
- Ability to optionally pass in a function evaluation at `params` or return a function
  evaluation at `params`

**Problems**

- We can make no assumptions on types inside the function because pytree handling is
  mixed with calculations
- Support for Richardson extrapolation complicates the interface but has not been
  convincing in benchmarks
- Pytree handling is acatually incomplete (`base_steps`, `mi@` and `step_ratio` are
  assumed to be flat numpy arrays)
- Many users expect the output of a function for numerical differentiation to be just
  the gradient, jacobian or hessian, not a more complex result object.

#### Proposal

##### Separation of calculations and pytree handling

As in numerical optimization, we should implement the core functionality for first and
second derivative for functions that map from 1-Dimensional numpy arrays to
1-Dimensional numpy arrays. All pytree handling or other handling of function outputs
(e.g. functions that return a `CriterionValue`) should be done outside of the core
functions.

##### Deprecate Richardson Extrapolation (and prepare alternatives)

The goal of implementing Richardson Extrapolation was to get more precise estimates of
numerical derivatives when it is hard to find an optimal step size. Example use-cases we
had in mind were:

- Optimization of a function that is piecewise flat, e.g. the likelihood function of a
  naively implemented multinomial probit
- Optimization or standard error estimation of slightly noisy functions, e.g. functions
  of an MSM estimation problem
- Standard error estimation of wiggly functions where the slope and curvature at the
  minimum does not yield reasonable standard errors and confidence intervals

Unfortunately, the computational cost of Richardson extrapolation is too high for any
application during optimization. Moreover, our practical experience with Richardson
Extrapolation was not positive and it seems that Richardson extrapolation is not
designed for our use-cases. It is designed as a sequence acceleration method that
reduces roundoff error while shrinking a step size to zero, whereas in our application
it might often be better to take a larger step size (for example, the success of
derivative free trust-region optimizers suggest less local slope and curvature
information is more useful than actual derivatives for optimization; similarly,
numerical derivatives with larger step sizes could be seen as an estimate of a
[quasi jacobian](https://arxiv.org/abs/1907.13093) and inference based on it might have
good statistical properties).

We therefore propose to remove Richardson extrapolation and open an Issue to work on
alternatives. Examples for alternatives could be:

- [Moré and Wild (2010)](https://www.mcs.anl.gov/papers/P1785.pdf) propose an approach
  to calculate optimal step sizes for finite difference differentiation of noisy
  functions
- We could think about aggregating derivative estimates at multiple step sizes in a way
  that produces worst case standard errors and confidence intervals
- ...

```{note}
Richardson extrapolation was only completed for first derivatives, even though it is
already prepared in the interface for second derivatives.
```

##### Better `NumdiffResult` object

The result dictionary will be replaced by a `NumdiffResult` object. All arguments that
govern which results are stored will be removed. If some of the formerly optional
results require extra computation that we wanted to avoid by making them optional, they
can be properties or methods of the result object.

##### Jax inspired high-level interfaces

Since our `first_derivative` and `second_derivative` functions need to fulfill very
specific requirements for use during optimization, they need to return a complex result
object. However, this can be annoying in simple situations where users just want a
gradient, jacobian or hessian.

To cover these simple situations and provide a high level interface to our numdiff
functions, we can provide a set of jax inspired decorators:

- `@grad`
- `@value_and_grad`
- `@jac` (no distinction between `@jacrev` and `jacfwd` necessary)
- `@value_and_jac`
- `@hessian`
- `@value_and_hessian`

All of these will be very simple wrappers around `first_derivative` and
`second_derivative` with very low implementation and maintenance costs.

## Benchmarking

### `get_benchmark_problems`

#### Current situation

As other functions in estimagic, `get_benchmark_problems` follows a design where
behavior can be switched on by a bool and configured by an options dictionary. The
following arguments are related to this:

- `additive_noise` and `additive_noise_options`
- `multiplicative_noise` and `multiplicative_noise_options`
- `scaling` and `scaling_options`

All of them have the purpose of adding some difficult characteristics to an existing
benchmark set, so we can analyze how well an optimizer can deal with this situation.

The name of the benchmark set is passed in as a string.

The return value of `get_benchmark_problems` is a nested dictionary. The keys in the
outer dictionary are the names of benchmark problems. The inner dictionaries represent
benchmark problems.

**Things we want to keep**

- Collecting benchmark problems in a dictionary is good because it makes it easy to
  merge problems from multiple benchmark sets or filter benchmark sets. A fixed field
  data structure would not work here.

**Problems**

- As discussed before, having separate arguments for switching-on behavior and
  configuring it can be dangerous
- Each single benchmark problem should not be represented as a dictionary
- Adding noise or scaling problems should be made more flexible and generic

#### Proposal

##### Add noise to benchmark problems

The four arguments `additive_noise`, `multiplicative_noise`, `additive_noise_options`,
and `multiplicative_noise_options` are combined in one `noise` argument. This `noise`
argument can be `bool | BenchmarkNoise`. If `False`, no noise is added. If `True`,
standard normal noise is added.

We implement several subclasses of `BenchmarkNoise` to cover the current use cases. As
syntactic sugar, we can make `BenchmarkNoise` instances addable (by implementing an
`__add__` method) so multiple sources of noise can be combined.

A rough prototype for `BenchmarkNoise` looks as follows:

```python
FvalType = TypeVar("FvalType", bound=float | NDArray[float])


class BenchmarkNoise(ABC):

    @abstractmethod
    def draw_noise(
        self, fval: FvalType, params: NDArray, size: int, rng: np.random.Generator
    ) -> FvalType:
        pass

    def __add__(self, other: BenchmarkNoise):
        pass
```

Passing `fval` and `params` to `draw_noise` enables use to implement multiplicative
noise (i.e. noise where the standard deviation scales with the function value) and
stochastic or deterministic wiggle (e.g. a sine curve that depends on params).
Therefore, this proposal does not just cover everything that is currently implemented
but also adds new functionality we wanted to implement.

##### Add scaling issues to benchmark problems

The `scaling_options` argument is deprecated. The `scaling` argument can be
`bool | BenchmarkScaler`. We implement `LinspaceBenchmarkScaler` to cover everything
that is implemented right now but more types of scaling can be implemented in the
future. A rough prototype of `BenchmarkScaler` looks as follows:

```python
class BenchmarkScaler(ABC):
    @abstractmethod
    def scale(self, params: NDArray) -> NDArray:
        pass

    @abstractmethod
    def unscale(self, params: NDArray) -> NDArray:
        pass
```

##### Representing benchmark problems

Instead of the fixed-field dictionary we will have a dataclass with corresponding
fields. This would roughly look as follows:

```python
@dataclass
class BenchmarkProblem:
    fun: Callable[[NDArray], CriterionValue]
    start_x: NDArray
    solution_x: NDArray | None
    start_fun: float
    solution_fun: float
```

### `run_benchmark`

`run_benchmark` takes `benchmark_problems` (covered in the previous section),
`optimize_options` and a few other arguments and returns a nested dictionary
representing benchmark results.

`optimize_options` can be a list of algorithm names, a dict with algorithm names as
values or a nested dict of

#### Current situation

**Things we want to keep**

- Collecting benchmark results in a dictionary is good because it makes it easy to merge
  results from multiple benchmark runs or filter results. A fixed field data structure
  would not work here.

**Problems**

- `optimize_options` are super flexible but error prone and hard to write as there is no
  autocomplete support
- Each single benchmark result should not be represented as a dictionary

#### Proposal

We restrict the typo of `optimize_options` to
`dict[str, Type[Algorithm] | Algorithm | OptimizeOptions]`. Here, `OptimizeOptions` will
be a simple dataclass that we need for `estimate_ml` and `estimate_msm` anyways.

Passing just lists of algorithm names is deprecated. Passing dicts as optimize options
is also deprecated. Most use-cases will be covered by passing dictionaries of configured
Algorithms as optimize options. Actually using the fully power of passing
`OptimizeOptions` will be rarely needed.

The return type of `run_benchmark` will be `dict[tuple[str], BenchmarkResult]`

`BenchmarkResult` is a dataclass with fields that mirror the keys of the current
dictionary. It will roughly look as follows:

```python
@dataclass
class BenchmarkResult:
    params_history: list[NDArray]
    fun_history: list[float]
    time_history: list[float]
    batches_history: list[int]
    solution: OptimizeResult
```

## Estimation

The changes to the estimation functions `estimate_ml` and `estimate_msm` will be
minimal:

- `lower_bounds` and `upper_bounds` are replaced by `bounds` (as in optimization)
- `numdiff_options` and `optimize_options` become dataclasses
- `logging` and `log_options` get aligned with our proposal for optimization

In the long run we plan a general overhaul of `MSM` estimation that provides better
access to currently internal objects such as the MSM objective function.

## Internal changes

This section will be fledged out when we start the implementation of this enhancement
proposal. Until then it serves as a collection of ideas.

- The `history` list becomes a class
- The internal criterion and derivative becomes a class instead of using multiple
  partialling.
- We add a logger abstraction that will enable alternatives to the sqlite database in
  the future
- ...

## Type checkers and their configuration

We choose mypy as static type checker and run it as part of our continuous integration.

Once this enhancement proposal is fully implemented, we want to use the following
settings:

```
check_untyped_defs = true
disallow_any_generics = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
```

## Runtime type checking

Since most of our users do not use static type checkers we will still need to check the
type of most user inputs so we can give them early feedback when problems arise.

We can investigate using `jaxtyping`'s pytest hooks to enable runtime typecheckers like
beartype during testing but it is not a priority for now.

## Summary of design philosophy

## Changes in documentation

- No type hints in docstrings anymore
- Only show new recommended ways of doing things, not deprecated ones

(aligning-names)=

## Aligning names

### Suggested changes

| **Old Name**                               | **Proposed Name**         | **Source** |
| ------------------------------------------ | ------------------------- | ---------- |
| `criterion`                                | `fun`                     | scipy      |
| `derivative`                               | `jac`                     | scipy      |
| `criterion_and_derivative`                 | `fun_and_jac`             | (follows)  |
| `stopping_max_criterion_evaluations`       | `stopping_maxfun`         | scipy      |
| `stopping_max_iterations`                  | `stopping_maxiter`        | scipy      |
| `convergence_absolute_criterion_tolerance` | `convergence_ftol_abs`    | NlOpt      |
| `convergence_relative_criterion_tolerance` | `convergence_ftol_rel`    | NlOpt      |
| `convergence_absolute_params_tolerance`    | `convergence_xtol_abs`    | NlOpt      |
| `convergence_relative_params_tolerance`    | `convergence_xtol_rel`    | NlOpt      |
| `convergence_absolute_gradient_tolerance`  | `convergence_gtol_abs`    | NlOpt      |
| `convergence_relative_gradient_tolerance`  | `convergence_gtol_rel`    | NlOpt      |
| `convergence_scaled_gradient_tolerance`    | `convergence_gtol_scaled` | (follows)  |

### Things we do not want to align

- We do not want to rename `algorithm` to `method` because our algorithm names are
  different from scipy, so people who switch over from scipy need to adjust their code
  anyways.
- We do not want to rename `algo_options` to `options` for the same reason.

### On the fence

- I am not sure if `params` should be renamed to `x0`. It would align estimagic more
  with scipy, but `params` is just easier to pronounce and use as a word than `x0`.

## Breaking changes

- The internal algorithm interface changes without deprecations
- The representation of benchmark problems and benchmark results changes without
  deprecations

## Summary of deprecations

The following deprecations become active in version `0.5.0`. The functionality will be
removed in version `0.6.0` which should be scheduled for approximately half a year after
the realease of `0.5.0`.

- Returning a `dict` in the `criterion` function io deprecated. Return `CriterionValue`
  instead or use `em.mark.least_squares` or `em.mark.likelihood` to create your
  criterion function.
- The arguments `lower_bounds`, `upper_bounds`, `soft_lower_bounds` and
  `soft_uppper_bounds` are deprecated. Use `bounds` instead.
