# How to document optimizers

This guide shows you how to document algorithms in optimagic using our new documentation
system. We'll walk through the process step-by-step using the `ScipyLBFGSB` optimizer as
a complete example.

## When to Use This Guide

Use this guide when you need to:

- Document a new algorithm you've added to optimagic
- Migrate existing algorithm documentation from the old split system (docstrings +
  `algorithms.md`) to the new system
- Update or improve existing algorithm documentation

If you're adding a completely new optimizer to optimagic, start with the "How to Add
Optimizers guide" first, then use this guide to document your algorithm properly.

## Why the New Documentation System?

Previously, algorithm documentation was scattered across multiple places:

- Basic descriptions in the algorithm class docstrings
- Detailed parameter descriptions in `algorithms.md`
- Usage examples separate from the algorithm definitions

This made it hard to maintain consistency and keep documentation up-to-date. The new
system centralizes nearly all documentation in the algorithm code itself, making it:

- Easier to maintain (documentation lives next to code)
- More consistent (unified format across all algorithms)
- Auto-generated (parameter lists appear automatically in docs)
- Type-safe (documentation matches actual parameter types)

## The Documentation System Components

Our documentation system has three main parts:

1. **Algorithm Class Documentation**: A comprehensive docstring in the algorithm
   dataclass that explains what the algorithm does, how it works, and when to use it
1. **Parameter Documentation**: Detailed docstrings for each parameter with mathematical
   formulations when needed
1. **Usage Integration**: A section in `algorithms.md` that show how to use the
   algorithm

Let's walk through documenting an algorithm from start to finish.

## Example: Documenting ScipyLBFGSB

We'll use the `ScipyLBFGSB` optimizer to show you exactly how to document an algorithm.
This is a real example from the optimagic codebase, so you can follow along and see the
results.

### Step 1: Understand Your Algorithm

Before writing documentation, make sure you understand:

- What the algorithm does mathematically
- What problems it's designed to solve
- How its parameters affect behavior
- Any performance characteristics or limitations

For L-BFGS-B, this means understanding it's a quasi-Newton method for bound-constrained
optimization that approximates the Hessian using gradient history.

```{eval-rst}

.. note::
    If you are simply migrating an existing algorithm, you can mostly rely on the
    existing documentation in the algorithm class docstring and `algorithms.md`.

```

### Step 2: Write the Algorithm Class Documentation

The algorithm class docstring is the most important part. It should give users
everything they need to decide whether to use this algorithm.

Here's how we document `ScipyLBFGSB`:

```python
# src/optimagic/optimizers/scipy_optimizers.py
class ScipyLBFGSB(Algorithm):
    """Minimize a scalar differentiable function using the L-BFGS-B algorithm.

    The optimizer is taken from scipy, which calls the Fortran code written by the
    original authors of the algorithm. The Fortran code includes the corrections
    and improvements that were introduced in a follow up paper.

    lbfgsb is a limited memory version of the original bfgs algorithm, that deals with
    lower and upper bounds via an active set approach.

    The lbfgsb algorithm is well suited for differentiable scalar optimization problems
    with up to several hundred parameters.

    It is a quasi-newton line search algorithm. At each trial point it evaluates the
    criterion function and its gradient to find a search direction. It then approximates
    the hessian using the stored history of gradients and uses the hessian to calculate
    a candidate step size. Then it uses a gradient based line search algorithm to
    determine the actual step length. Since the algorithm always evaluates the gradient
    and criterion function jointly, the user should provide a ``fun_and_jac`` function
    that exploits the synergies in the calculation of criterion and gradient.

    The lbfgsb algorithm is almost perfectly scale invariant. Thus, it is not necessary
    to scale the parameters.

    """
```

**What makes this docstring effective:**

- **Clear first line**: States exactly what the algorithm does
- **Implementation details**: Explains it uses scipy's Fortran implementation
- **Algorithm classification**: Identifies it as a quasi-Newton method
- **Problem suitability**: Explains what problems it's good for
- **How it works**: Brief explanation of the algorithm's approach
- **Performance characteristics**: Mentions scale invariance
- **Usage advice**: Suggests using `fun_and_jac` for efficiency

### Step 3: Document Individual Parameters

Each parameter needs clear documentation explaining what it controls and how it affects
the algorithm's behavior.

```python
# Basic parameter documentation
stopping_maxiter: PositiveInt = STOPPING_MAXITER
"""Maximum number of iterations."""

# Parameter with mathematical formulation
convergence_ftol_rel: NonNegativeFloat = CONVERGENCE_FTOL_REL
r"""Converge if the relative change in the objective function is less than this
value. More formally, this is expressed as.

.. math::

    \frac{f^k - f^{k+1}}{\max\{|f^k|, |f^{k+1}|, 1\}} \leq
    \textsf{convergence_ftol_rel}.

"""

# Parameter with external library context
limited_memory_storage_length: PositiveInt = LIMITED_MEMORY_STORAGE_LENGTH
"""The maximum number of variable metric corrections used to define the limited
memory matrix. This is the 'maxcor' parameter in the SciPy documentation.

The default value is taken from SciPy's L-BFGS-B implementation. Larger values use
more memory but may converge faster for some problems.

"""
```

**Key principles for parameter documentation:**

- **Start with a clear description** of what the parameter controls
- **Add mathematical formulations** when they clarify the exact meaning (use `r"""` for
  raw strings with LaTeX)
- **Include external library context** when relevant (e.g., "Default value is taken from
  SciPy")
- **Explain performance implications** when they matter
- **Use proper type annotations** that match the parameter's constraints

> **Note:** If your optimizer module uses type hints (e.g., `PositiveInt`,
> `NonNegativeInt`), include the following at the top of your optimizer module:

```python
from __future__ import annotations
```

Without this, type hints such as `PositiveInt` may appear decomposed in the
documentation (e.g., as `Annotated[int, Gt(gt=0)]`).

### Step 4: Integrate into `algorithms.md`

The final step is integrating your documented algorithm into the main documentation.
This creates a dropdown section that shows users how to use the algorithm.

Add the following to `docs/source/algorithms.md` in an `eval-rst` block:

```text
.. dropdown::  scipy_lbfgsb

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        om.minimize(
          fun=lambda x: x @ x,
          params=[1.0, 2.0, 3.0],
          algorithm=om.algos.scipy_lbfgsb(stopping_maxiter=1_000, ...),
        )
        
    or using the string interface:
        
    .. code-block:: python

        om.minimize(
          fun=lambda x: x @ x,
          params=[1.0, 2.0, 3.0],
          algorithm="scipy_lbfgsb",
          algo_options={"stopping_maxiter": 1_000, ...},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.scipy_optimizers.ScipyLBFGSB
```

**What this section provides:**

- **The dropdown button and title**: Makes it easy to find the algorithm
- **Concrete usage examples** showing both the object and string interfaces
- **Algorithm-specific parameter** in the usage example
- **Auto-generated documentation** via the `autoclass` directive that pulls in your
  docstrings

## Working with Existing Documentation

If you're migrating an algorithm that already has documentation:

### Finding Existing Content

Look for existing documentation in:

- **Algorithm class docstrings**: Usually basic descriptions
- **`docs/source/algorithms.md`**: Detailed parameter descriptions and examples
- **Research papers**: For mathematical formulations and background
- **External library docs**: For default values and parameter meanings

### Migration Strategy

1. **Start with the algorithm class**: Move the best description from `algorithms.md` to
   the class docstring
1. **Update and expand**: Add missing information about performance, usage, etc.
1. **Move parameter docs**: Transfer parameter descriptions from `algorithms.md` to
   individual parameter docstrings
1. **Verify accuracy**: Check that all information is current and correct
1. **Create new integration**: Replace the old `algorithms.md` section with the new
   dropdown format

## Common Pitfalls to Avoid

- **Don't copy-paste generic descriptions**: Each algorithm needs specific, detailed
  documentation
- **Don't skip mathematical formulations**: When convergence criteria or parameters have
  precise mathematical definitions, include them
- **Don't ignore external library context**: Always mention where default values come
  from
- **Don't use vague parameter descriptions**: "Controls the algorithm behavior" is not
  helpful
- **Don't forget performance implications**: Users need to understand trade-offs between
  parameters

## Getting Help

If you're stuck or need clarification:

- Look at existing well-documented algorithms like `ScipyLBFGSB`
- Check the {ref}`style_guide` for coding conventions
- Ask questions in GitHub issues or discussions

The goal is to make optimagic's algorithm documentation the best resource for
understanding and using optimization algorithms effectively.
