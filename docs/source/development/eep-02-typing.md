(eeppytrees)=

# EEP-01: Static typing

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
  A very good blogpost that the drawbacks of "stringly-typed" Python code and shows how
  to incorporate typing philosophies from Rust into Python projects. Read this if you
  don't have time to read the other ressources.
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
scalar algorithms will be used) or a dictionary that contains at the entries "value" (a
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


def least_squares_sphere(params: np.ndarray) -> dict[str:Any]:
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
  `float | dict[str: Any]` which is not very useful.

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
which is more in line with the literater.

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

- fixed constraints have "value"
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
  formats) is confusing for users and annoying when processing constraints.
- Dicts with required keys are brittle and do not provide autocomplete. This is made
  worse by the fact that each type of constraint requires different sets of keys.

#### Proposal

### Algorithm selection

#### Current situation

**Things we want to keep** **Problems**

#### Proposal

### Algorithm options

#### Current situation

**Things we want to keep** **Problems**

#### Proposal

### Custom derivatives

#### Current situation

**Things we want to keep** **Problems**

#### Proposal

### Other option dictionaries

#### Current situation

**Things we want to keep** **Problems**

#### Proposal

### The internal algorithm interface

#### Current situation

**Things we want to keep** **Problems**

#### Proposal

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
