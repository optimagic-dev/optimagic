# optimagic API

```{eval-rst}
.. currentmodule:: optimagic
```

(maximize-and-minimize)=

## Optimization

```{eval-rst}
.. dropdown:: maximize

    .. autofunction:: maximize
```

```{eval-rst}
.. dropdown:: minimize

    .. autofunction:: minimize

```

```{eval-rst}
.. dropdown:: slice_plot

    .. autofunction:: slice_plot

```

```{eval-rst}
.. dropdown:: criterion_plot

    .. autofunction:: criterion_plot

```

```{eval-rst}
.. dropdown:: params_plot

    .. autofunction:: params_plot


```

```{eval-rst}
.. dropdown:: OptimizeResult

    .. autoclass:: OptimizeResult
        :members:

```

```{eval-rst}
.. dropdown:: Bounds

    .. autoclass:: Bounds
        :members:

```

```{eval-rst}
.. dropdown:: Constraints

    .. autoclass:: FixedConstraint
        :members:

    .. autoclass:: IncreasingConstraint
        :members:

    .. autoclass:: DecreasingConstraint
        :members:

    .. autoclass:: EqualityConstraint
        :members:

    .. autoclass:: ProbabilityConstraint
        :members:

    .. autoclass:: PairwiseEqualityConstraint
        :members:

    .. autoclass:: FlatCovConstraint
        :members:

    .. autoclass:: FlatSDCorrConstraint
        :members:

    .. autoclass:: LinearConstraint
        :members:

    .. autoclass:: NonlinearConstraint
        :members:

```

```{eval-rst}
.. dropdown:: NumdiffOptions

    .. autoclass:: NumdiffOptions
        :members:

```

```{eval-rst}
.. dropdown:: MultistartOptions

    .. autoclass:: MultistartOptions
        :members:

```

```{eval-rst}
.. dropdown:: ScalingOptions

    .. autoclass:: ScalingOptions
        :members:

```

```{eval-rst}
.. dropdown:: LogOptions

    .. autoclass:: SQLiteLogOptions
        :members:

```

```{eval-rst}
.. dropdown:: History

    .. autoclass:: History
        :members:

```

```{eval-rst}
.. dropdown:: count_free_params

    .. autofunction:: count_free_params

```

```{eval-rst}
.. dropdown:: check_constraints

    .. autofunction:: check_constraints

```

(first_derivative)=

## Derivatives

```{eval-rst}
.. dropdown:: first_derivative

    .. autofunction:: first_derivative

```

```{eval-rst}
.. dropdown:: second_derivative

    .. autofunction:: second_derivative

```

(benchmarking)=

## Benchmarks

```{eval-rst}
.. dropdown:: get_benchmark_problems

    .. autofunction:: get_benchmark_problems
```

```{eval-rst}
.. dropdown:: run_benchmark

    .. autofunction:: run_benchmark
```

```{eval-rst}
.. dropdown:: profile_plot

    .. autofunction:: profile_plot
```

```{eval-rst}
.. dropdown:: convergence_plot

    .. autofunction:: convergence_plot


```

(logreading)=

## Log reading

```{eval-rst}
.. dropdown:: OptimizeLogReader

    .. autoclass:: OptimizeLogReader



```

## Other:

```{toctree}
---
maxdepth: 1
---
utilities
algo_options
batch_evaluators
```
