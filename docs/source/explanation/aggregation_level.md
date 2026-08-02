(aggregation_level)=

# Problem types (`AggregationLevel`)

optimagic can optimize three kinds of objective functions: **scalar**,
**least-squares**, and **likelihood**. Internally these are represented by
{class}`~optimagic.typing.AggregationLevel`.

You tell optimagic which kind you have by marking the objective function with a
decorator from `optimagic.mark` (`@om.mark.least_squares`, `@om.mark.likelihood`, or
optionally `@om.mark.scalar`). That mark changes:

1. **What your function should return** (a single number vs a vector of contributions or
   residuals).
1. **Which specialized optimizers you can use** (for example pounders for least-squares,
   or BHHH for likelihood).
<<<<<<< HEAD
1. **How error penalties and derivatives are interpreted** when something goes wrong or
   when a scalar optimizer is used on a specialized problem.
=======
3. **How error penalties and derivatives are interpreted** when something goes wrong
   (see {ref}`how-to-errors`) or when a scalar optimizer is used on a specialized
   problem.
>>>>>>> 04ce067 (docs: address review on AggregationLevel explanation)

Any marked function can still be solved with a normal scalar optimizer; optimagic
aggregates the vector output when needed (sum of squares for least-squares, sum of
contributions for likelihood).

## Scalar problems

This is the default. Your function returns a **single number** — the value to minimize
(or maximize).

```python
import optimagic as om
import numpy as np


# @om.mark.scalar is optional; unmarked functions are treated as scalar
def sphere(params):
    return params @ params


om.minimize(sphere, params=np.arange(3), algorithm="scipy_lbfgsb")
```

Use this whenever you do not have least-squares or likelihood structure to exploit.

## Least-squares problems

Mark the function with `@om.mark.least_squares` and return the **residuals** (a vector
or pytree), **not** the sum of squared residuals.

```python
@om.mark.least_squares
def ls_sphere(params):
    return params  # residuals; optimagic forms sum of squares if needed
```

**Why mark it?** Specialized least-squares solvers can use the residual structure and
are often much faster than treating $f(x)=\sum_i r_i(x)^2$ as a black-box scalar. If you
only return the scalar sum of squares, those solvers cannot be used.

See {ref}`how-to-fun` for a short usage example.

## Likelihood problems

Mark the function with `@om.mark.likelihood` and return a **vector (or pytree) of
per-observation log-likelihood contributions**, not a single summed log-likelihood.

```python
@om.mark.likelihood
def loglike_contributions(params):
    # return one log-density value per observation (an array), not their sum
    ...
```

**Sign / maximize vs minimize:** return the actual log-likelihood contributions (the
quantities you would sum to get $\ell(\theta)$). Prefer {func}`~optimagic.maximize` for
maximum likelihood; optimagic flips the sign internally for the solver. If you prefer
{func}`~optimagic.minimize`, return **negative** log-likelihood contributions instead.

Do **not** return only the summed scalar log-likelihood if you want likelihood-specific
optimizers — they need the contributions.

For estimation workflows built on likelihood functions, see also {ref}`estimagic`.

## How this relates to `AggregationLevel`

| Problem       | Decorator                 | Typical return      | `AggregationLevel` |
| ------------- | ------------------------- | ------------------- | ------------------ |
| Scalar        | none or `@om.mark.scalar` | `float`             | `SCALAR`           |
| Least-squares | `@om.mark.least_squares`  | residual vector     | `LEAST_SQUARES`    |
| Likelihood    | `@om.mark.likelihood`     | contribution vector | `LIKELIHOOD`       |

Optimizers are also tagged with a `solver_type` of the same enum (see
{ref}`internal_optimizer_interface`). Matching the mark on your function to the solver
type is what lets optimagic pick the right internal representation.
