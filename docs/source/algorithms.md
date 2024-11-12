(list_of_algorithms)=

# Optimizers

Check out {ref}`how-to-select-algorithms` to see how to select an algorithm and specify
`algo_options` when using `maximize` or `minimize`.

## Optimizers from scipy

(scipy-algorithms)=

optimagic supports most `scipy` algorithms and scipy is automatically installed when you
install optimagic.

```{eval-rst}
.. dropdown::  scipy_lbfgsb

    .. code-block::

        "scipy_lbfgsb"

    .. autoclass:: optimagic.optimizers.scipy_optimizers.ScipyLBFGSB
      :members:

```

```{eval-rst}
Minimize a scalar function of one or more variables using the L-BFGS-B algorithm.

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
    and criterion function jointly, the user should provide a
    ``criterion_and_derivative`` function that exploits the synergies in the
    calculation of criterion and gradient.

    The lbfgsb algorithm is almost perfectly scale invariant. Thus, it is not necessary
    to scale the parameters.

    - **convergence.ftol_rel** (float): Stop when the relative improvement
      between two iterations is smaller than this. More formally, this is expressed as

    .. math::

        \frac{(f^k - f^{k+1})}{\\max{{|f^k|, |f^{k+1}|, 1}}} \leq
        \text{relative_criterion_tolerance}


    - **convergence.gtol_abs** (float): Stop if all elements of the projected
      gradient are smaller than this.
    - **stopping.maxfun** (int): If the maximum number of function
      evaluation is reached, the optimization stops but we do not count this as convergence.
    - **stopping.maxiter** (int): If the maximum number of iterations is reached,
      the optimization stops, but we do not count this as convergence.
    - **limited_memory_storage_length** (int): Maximum number of saved gradients used to approximate the hessian matrix.

```

## References

```{eval-rst}
.. bibliography:: refs.bib
    :labelprefix: algo_
    :filter: docname in docnames
    :style: unsrt
```
