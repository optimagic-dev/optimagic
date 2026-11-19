(how-to-document-optimizers)=

# How to document optimizers

This guide shows how to document an algorithm in optimagic. Use it when you add a new
algorithm or improve the documentation of an existing one. If you add a new optimizer,
start with [how to add optimizers](how_to_add_optimizers.md).

The goal of the documentation is that users understand what an option does and how to
choose its value, without reading the paper. It should not be academic or low-level.

## Where the documentation lives

All documentation of an algorithm is in the code:

1. The **docstring of the algorithm class** explains what the algorithm does and when to
   use it.
1. The **docstring of each option** explains the effect of the option. Option groups are
   documented in the same way: one docstring for the class and one per attribute.
1. An **entry in `algorithms.md`** shows how to select the algorithm and renders the
   docstrings.

## The docstring of the algorithm class

The class docstring gives users what they need to decide whether to use the algorithm:

- The first line says what the algorithm does.
- For which problems it is a good choice, and for which it is not. Compare it to related
  algorithms.
- How it works, in a few intuitive sentences.
- Where the implementation comes from, how to install it and which papers to cite.
- Which features of the library are not supported and why.
- A link to the tutorial, if there is one.

Here is an excerpt from the docstring of `Fides`:

```python
class Fides(Algorithm):
    """Minimize a scalar function using the Fides trust-region optimizer.

    Fides is a gradient-based local optimizer for smooth scalar functions with or
    without bounds. It is a good choice if evaluating the function and its gradient
    is expensive, because it spends more effort than line-search methods such as
    L-BFGS-B on choosing each step and typically needs fewer evaluations in return.
    The price is a matrix factorization per iteration, so it is less suited for
    problems with thousands of parameters.

    ...

    The following features of fides are not supported yet:

    - Exact Hessians and the hybrid Hessian approximations that need them
      (``HybridFixed``, ``HybridFraction``), because optimagic does not pass
      user-provided Hessians to optimizers.
    - ...

    """
```

(option-docstring-template)=

## The docstring of an option

Every option docstring has three parts:

1. **One plain sentence that says what the option controls.**
1. **The effect of the values and when to change them.** For a number: what happens with
   larger and with smaller values. For a choice: what each choice does. Say in which
   situation a user should deviate from the default.
1. **A closing line with the name of the option in the library and the source of the
   default.** If the default differs from the default of the library, say why.

```python
trustregion_initial_radius: PositiveFloat = 1.0
"""Radius of the trust region in the first iteration.

The radius limits the length of the first step, measured in the units of the
parameters. Decrease it if the first steps lead into regions where the function
is not defined or behaves very differently than at the start parameters.
Increase it if the start parameters are far from the optimum and the first
steps are too cautious. After the first iterations the radius adapts itself.

This is the ``delta_init`` option of fides and has the same default.

"""
```

Rules for the wording:

- Describe effects, not formulas. Use a formula only where it is the definition of the
  option, which is mostly the case for convergence criteria.
- Use the symbols of the paper only in the closing line. Write "the radius of the trust
  region", not "delta".
- Explain technical terms in half a sentence when you first use them, for example "the
  quality of a step, i.e. the actual improvement of the function divided by the
  improvement the model predicted".
- Only describe effects you have verified in the code of the library or in an
  experiment. Documentation of libraries is sometimes out of date.
- For convergence criteria, say whether the criterion measures closeness to an optimum
  or only progress.
- If the default is `None` because the library chooses a value that depends on the
  problem, state the rule the library uses.

An option with a formula and a default that differs from the library looks like this:

```python
convergence_gtol_rel: NonNegativeFloat = CONVERGENCE_GTOL_REL
r"""Stop when the norm of the gradient is at most this value times the absolute
function value.

This makes the gradient criterion independent of the scale of the function,
i.e. fides stops when :math:`\|g_k\| \leq \text{gtol\_rel} \cdot |f_k|`. It has
no effect if the function value at the optimum is zero.

This is the ``grtol`` option of fides. The default of 1e-8 is harmonized across
optimagic; fides defaults to 0, which switches the criterion off.

"""
```

## Type hints

Option values are validated against the type hints, so the type hints are part of the
documentation. Use the types from `optimagic.typing` that describe valid values, such as
`PositiveInt` or `NonNegativeFloat`, and `Literal` for choices.

```{eval-rst}

.. warning::
    Include the following at the top of your optimizer module:

    .. code-block:: python

        from __future__ import annotations

    Without this, type hints such as ``PositiveInt`` may appear decomposed in the
    documentation (e.g., as ``Annotated[int, Gt(gt=0)]``).

.. warning::
    Option values are validated with pydantic, which resolves the annotations at
    runtime. Every name used in a field annotation must therefore be importable at
    runtime -- a name that is only imported inside an ``if TYPE_CHECKING:`` block
    will raise an error when the algorithm is instantiated. If an optional
    dependency provides a type you want to use in an annotation, add a runtime
    fallback next to the ``TYPE_CHECKING`` import, as is done for
    ``HessianApproximation`` in ``optimagic/optimizers/fides.py``.

    Since annotations are enforced, they must also be honest: document and annotate
    the values the optimizer actually accepts, rather than a narrower set.

```

## The entry in `algorithms.md`

Add a dropdown to `docs/source/algorithms.md` in an `eval-rst` block:

```text
.. dropdown:: scipy_lbfgsb

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.scipy_lbfgsb(stopping_maxiter=1_000),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="scipy_lbfgsb",
            algo_options={"stopping_maxiter": 1_000},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.scipy_optimizers.ScipyLBFGSB
        :members:
        :inherited-members: Algorithm, object
```

- Each code block is self-contained and runs as is. Pass `params` as a NumPy array so
  that `x @ x` works, do not use `...` placeholders, add finite `bounds` if the
  algorithm needs them, and mark least-squares or likelihood functions with
  `om.mark.least_squares` or `om.mark.likelihood`.
- Set an option that is specific to the algorithm in the examples.
- The `:inherited-members: Algorithm, object` option is required: `mark.minimizer`
  re-creates the algorithm class as a pydantic dataclass that inherits from your class,
  so without it autodoc treats your documented fields as inherited and does not show
  them.
- If the algorithm has option groups, add an `autoclass` directive with `:members:` and
  `:inherited-members: object` for each group class below the algorithm. See the entry
  of `nag_dfols` for an example.

## Check the result

Build the documentation with `pixi run -e docs build-docs` and read the rendered page of
your algorithm. The build fails on warnings, for example on broken references.

Common problems are:

- Descriptions that only repeat the name of the option, such as "Controls the shrinking
  of the radius".
- Descriptions that are copied from the paper and can only be understood with the paper.
- Missing information on where a default comes from.
- Missing advice on when to change an option.
