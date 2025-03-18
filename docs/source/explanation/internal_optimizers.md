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

To make switching between different algorithm as simple as possible, we align the names
of commonly used convergence and stopping criteria. We also align the default values for
stopping and convergence criteria as much as possible.

You can find the harmonized names and value [here](algo_options_docs).

To align the names of other tuning parameters as much as possible with what is already
there, simple have a look at the optimizers we already wrapped. For example, if you are
wrapping a bfgs or lbfgs algorithm from some libray, try to look at all existing
wrappers of bfgs algorithms and use the same names for the same options.

## Algorithms that parallelize

Algorithms that evaluate the objective function or derivatives in parallel should only
do so via `InternalOptimizationProblem.batch_fun`,
`InternalOptimizationProblem.batch_jac` or
`InternalOptimizationProblem.batch_fun_and_jac`.

If you parallelize in any other way, the automatic history collection will stop to work.

In that case, call `om.mark.minimizer` with `disable_history=True`. In that case you can
either do your own history collection and add that history to `InternalOptimizeResult`
or the user has to rely on logging.

## Nonlinear constraints

(to be written)
