(internal_optimizer_interface)=

# Internal optimizers for optimagic

optimagic provides a large collection of optimization algorithm that can be used by
passing the algorithm name as `algorithm` into `maximize` or `minimize`. Advanced users
can also use optimagic with their own algorithm, as long as it conforms with the
internal optimizer interface.

The advantages of using the algorithm with optimagic over using it directly are:

- You can collect the optimizer history and create criterion_plots and params_plots.
- You can use flexible formats for your start parameters (e.g. nested dicts or
  namedtuples)
- optimagic turns unconstrained optimizers into constrained ones.
- You can use logging.
- You get great error handling for exceptions in the criterion function or gradient.
- You get a parallelized and customizable numerical gradient if you don't have a closed
  form gradient.
- You can compare your optimizer with all the other optimagic optimizers on our
  benchmark sets.

All of this functionality is achieved by transforming a more complicated user provided
problem into a simpler problem and then calling "internal optimizers" to solve the
transformed problem.

(functions_and_classes_for_internal_optimizers)=

## Functions and classes for internal optimizers

The functions and classes below are everything you need to know to add an optimizer to
optimagic. To see them in action look at
[this guide](../how_to/how_to_add_optimizers.ipynb)

```{eval-rst}
.. currentmodule:: optimagic.mark
```

```{eval-rst}
.. dropdown:: mark.minimizer

    The `mark.minimizer` decorator is used to provide algorithm specific information to
    optimagic. This information is used in the algorithm selection tool, for better
    error handling and for processing of the user provided optimization problem.

    .. autofunction:: minimizer
```

```{eval-rst}
.. currentmodule:: optimagic.optimization.internal_optimization_problem
```

```{eval-rst}


.. dropdown:: InternalOptimizationProblem

    The `InternalOptimizationProblem` is optimagic's internal representation of objective
    functions, derivatives, bounds, constraints, and more. This representation is already
    pretty close to what most algorithms expect (e.g. parameters and bounds are flat
    numpy arrays, no matter which format the user provided).

    .. autoclass:: InternalOptimizationProblem()
        :members:

```

```{eval-rst}
.. currentmodule:: optimagic.optimization.algorithm
```

```{eval-rst}

.. dropdown:: InternalOptimizeResult

    This is what you need to create from the output of a wrapped algorithm.

    .. autoclass:: InternalOptimizeResult
        :members:

```

```{eval-rst}

.. dropdown:: Algorithm

    .. autoclass:: Algorithm
        :members:
        :exclude-members: with_option_if_applicable

```

(naming-conventions)=

## Naming conventions for algorithm specific arguments

Many optimizers have similar but slightly different names for arguments that configure
the convergence criteria, other stopping conditions, and so on. We try to harmonize
those names and their default values where possible.

Since some optimizers support many tuning parameters we group some of them by the first
part of their name (e.g. all convergence criteria names start with `convergence`). See
{ref}`list_of_algorithms` for the signatures of the provided internal optimizers.

The preferred default values can be imported from `optimagic.optimization.algo_options`
which are documented in {ref}`algo_options`. If you add a new optimizer to optimagic you
should only deviate from them if you have good reasons.

Note that a complete harmonization is not possible nor desirable, because often
convergence criteria that clearly are the same are implemented slightly different for
different optimizers. However, complete transparency is possible and we try to document
the exact meaning of all options for all optimizers.

## Algorithms that parallelize

(to be written)

## Nonlinear constraints

(to be written)
