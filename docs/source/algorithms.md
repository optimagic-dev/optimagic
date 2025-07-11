(list_of_algorithms)=

# Optimizers

Check out {ref}`how-to-select-algorithms` to see how to select an algorithm and specify
`algo_options` when using `maximize` or `minimize`.

## Optimizers from scipy

(scipy-algorithms)=

optimagic supports most [SciPy](https://scipy.org/) algorithms and SciPy is
automatically installed when you install optimagic.

```{eval-rst}
.. dropdown::  ``scipy_lbfgsb``

    **How to use this algorithm:**

    .. code-block::

        import optimagic as om
        om.minimize(
          ...,
          algorithm=om.algos.scipy_lbfgsb(stopping_maxiter=1_000, ...)
        )
        
    or
        
    .. code-block::

        om.minimize(
          ...,
          algorithm="scipy_lbfgsb",
          algo_options={"stopping_maxiter": 1_000, ...}
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.scipy_optimizers.ScipyLBFGSB

```

## References

```{eval-rst}
.. bibliography:: refs.bib
    :labelprefix: algo_
    :filter: docname in docnames
    :style: unsrt
```
