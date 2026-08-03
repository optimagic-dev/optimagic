(list_of_algorithms)=

# Optimizers

Check out {ref}`how-to-select-algorithms` to see how to select an algorithm and specify
`algo_options` when using `maximize` or `minimize`. The default algorithm options are
discussed in {ref}`algo_options` and their type hints are documented in {ref}`typing`.

## Optimizers from SciPy

(scipy-algorithms)=

optimagic supports most [SciPy](https://scipy.org/) algorithms and SciPy is
automatically installed when you install optimagic.

```{eval-rst}
.. dropdown::  scipy_lbfgsb

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

```{eval-rst}
.. dropdown::  scipy_slsqp

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.scipy_slsqp(stopping_maxiter=1_000, ...),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="scipy_slsqp",
            algo_options={"stopping_maxiter": 1_000, ...},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.scipy_optimizers.ScipySLSQP
```

```{eval-rst}
.. dropdown::  scipy_neldermead

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.scipy_neldermead(stopping_maxiter=1_000, ...),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="scipy_neldermead",
            algo_options={"stopping_maxiter": 1_000, ...},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.scipy_optimizers.ScipyNelderMead
```

```{eval-rst}
.. dropdown::  scipy_powell

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.scipy_powell(stopping_maxiter=1_000, ...),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="scipy_powell",
            algo_options={"stopping_maxiter": 1_000, ...},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.scipy_optimizers.ScipyPowell
```

```{eval-rst}
.. dropdown::  scipy_bfgs

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.scipy_bfgs(stopping_maxiter=1_000, ...),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="scipy_bfgs",
            algo_options={"stopping_maxiter": 1_000, ...},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.scipy_optimizers.ScipyBFGS
```

```{eval-rst}
.. dropdown::  scipy_conjugate_gradient

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.scipy_conjugate_gradient(stopping_maxiter=1_000, ...),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="scipy_conjugate_gradient",
            algo_options={"stopping_maxiter": 1_000, ...},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.scipy_optimizers.ScipyConjugateGradient
```

```{eval-rst}
.. dropdown::  scipy_newton_cg

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.scipy_newton_cg(stopping_maxiter=1_000, ...),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="scipy_newton_cg",
            algo_options={"stopping_maxiter": 1_000, ...},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.scipy_optimizers.ScipyNewtonCG
```

```{eval-rst}
.. dropdown::  scipy_cobyla

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.scipy_cobyla(stopping_maxiter=1_000, ...),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="scipy_cobyla",
            algo_options={"stopping_maxiter": 1_000, ...},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.scipy_optimizers.ScipyCOBYLA
```

```{eval-rst}
.. dropdown::  scipy_truncated_newton

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.scipy_truncated_newton(stopping_maxfun=100_000, ...),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="scipy_truncated_newton",
            algo_options={"stopping_maxfun": 100_000, ...},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.scipy_optimizers.ScipyTruncatedNewton
```

```{eval-rst}
.. dropdown::  scipy_trust_constr

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.scipy_trust_constr(stopping_maxiter=1_000, ...),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="scipy_trust_constr",
            algo_options={"stopping_maxiter": 1_000, ...},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.scipy_optimizers.ScipyTrustConstr
```

```{eval-rst}
.. dropdown::  scipy_ls_dogbox

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om


        @om.mark.least_squares
        def fun(x):
            return x


        om.minimize(
            fun=fun,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.scipy_ls_dogbox(stopping_maxfun=1_000, ...),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=fun,
            params=[1.0, 2.0, 3.0],
            algorithm="scipy_ls_dogbox",
            algo_options={"stopping_maxfun": 1_000, ...},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.scipy_optimizers.ScipyLSDogbox
```

```{eval-rst}
.. dropdown::  scipy_ls_trf

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om


        @om.mark.least_squares
        def fun(x):
            return x


        om.minimize(
            fun=fun,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.scipy_ls_trf(stopping_maxfun=1_000, ...),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=fun,
            params=[1.0, 2.0, 3.0],
            algorithm="scipy_ls_trf",
            algo_options={"stopping_maxfun": 1_000, ...},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.scipy_optimizers.ScipyLSTRF
```

```{eval-rst}
.. dropdown::  scipy_ls_lm

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om


        @om.mark.least_squares
        def fun(x):
            return x


        om.minimize(
            fun=fun,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.scipy_ls_lm(stopping_maxfun=1_000, ...),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=fun,
            params=[1.0, 2.0, 3.0],
            algorithm="scipy_ls_lm",
            algo_options={"stopping_maxfun": 1_000, ...},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.scipy_optimizers.ScipyLSLM
```

```{eval-rst}
.. dropdown::  scipy_basinhopping

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.scipy_basinhopping(n_local_optimizations=10, ...),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="scipy_basinhopping",
            algo_options={"n_local_optimizations": 10, ...},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.scipy_optimizers.ScipyBasinhopping
```

```{eval-rst}
.. dropdown::  scipy_brute

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        import numpy as np
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.scipy_brute(n_grid_points=50, ...),
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="scipy_brute",
            algo_options={"n_grid_points": 50, ...},
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.scipy_optimizers.ScipyBrute
```

```{eval-rst}
.. dropdown::  scipy_differential_evolution

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        import numpy as np
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.scipy_differential_evolution(
                population_size_multiplier=10, ...
            ),
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="scipy_differential_evolution",
            algo_options={"population_size_multiplier": 10, ...},
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.scipy_optimizers.ScipyDifferentialEvolution
```

```{eval-rst}
.. dropdown::  scipy_shgo

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.scipy_shgo(n_sampling_points=256, ...),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="scipy_shgo",
            algo_options={"n_sampling_points": 256, ...},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.scipy_optimizers.ScipySHGO
```

```{eval-rst}
.. dropdown::  scipy_dual_annealing

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        import numpy as np
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.scipy_dual_annealing(stopping_maxiter=1_000, ...),
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="scipy_dual_annealing",
            algo_options={"stopping_maxiter": 1_000, ...},
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.scipy_optimizers.ScipyDualAnnealing
```

```{eval-rst}
.. dropdown::  scipy_direct

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        import numpy as np
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.scipy_direct(stopping_maxfun=10_000, ...),
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="scipy_direct",
            algo_options={"stopping_maxfun": 10_000, ...},
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.scipy_optimizers.ScipyDirect
```

(own-algorithms)=

## Own optimizers

We implement a few algorithms from scratch. They are currently considered experimental.

```{eval-rst}
.. dropdown::  bhhh

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om


        @om.mark.likelihood
        def fun(x):
            return x**2


        om.minimize(
            fun=fun,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.bhhh(stopping_maxiter=1_000),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=fun,
            params=[1.0, 2.0, 3.0],
            algorithm="bhhh",
            algo_options={"stopping_maxiter": 1_000},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.bhhh.BHHH
```

```{eval-rst}
.. dropdown::  neldermead_parallel

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.neldermead_parallel(n_cores=2),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="neldermead_parallel",
            algo_options={"n_cores": 2},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.neldermead.NelderMeadParallel
```

```{eval-rst}
.. dropdown::  pounders

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om


        @om.mark.least_squares
        def fun(x):
            return x


        om.minimize(
            fun=fun,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.pounders(stopping_maxiter=1_000),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=fun,
            params=[1.0, 2.0, 3.0],
            algorithm="pounders",
            algo_options={"stopping_maxiter": 1_000},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.pounders.Pounders
```

(tao-algorithms)=

## Optimizers from the Toolkit for Advanced Optimization (TAO)

We wrap the pounders algorithm from the Toolkit of Advanced optimization. To use it you
need to have [petsc4py](https://pypi.org/project/petsc4py/) installed.

```{eval-rst}
.. dropdown::  tao_pounders

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om


        @om.mark.least_squares
        def fun(x):
            return x


        om.minimize(
            fun=fun,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.tao_pounders(stopping_maxiter=1_000, ...),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=fun,
            params=[1.0, 2.0, 3.0],
            algorithm="tao_pounders",
            algo_options={"stopping_maxiter": 1_000, ...},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.tao_optimizers.TAOPounders
```

(nag-algorithms)=

## Optimizers from the Numerical Algorithms Group (NAG)

We wrap two algorithms from the numerical algorithms group. To use them, you need to
install each of them separately:

- `pip install DFO-LS`
- `pip install Py-BOBYQA`

```{eval-rst}
.. dropdown::  nag_dfols

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om


        @om.mark.least_squares
        def fun(x):
            return x


        om.minimize(
            fun=fun,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.nag_dfols(stopping_maxfun=10_000, ...),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=fun,
            params=[1.0, 2.0, 3.0],
            algorithm="nag_dfols",
            algo_options={"stopping_maxfun": 10_000, ...},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nag_optimizers.NagDFOLS
```

```{eval-rst}
.. dropdown::  nag_pybobyqa

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.nag_pybobyqa(
                stopping_max_criterion_evaluations=10_000, ...
            ),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="nag_pybobyqa",
            algo_options={"stopping_max_criterion_evaluations": 10_000, ...},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nag_optimizers.NagPyBOBYQA
```

(pygmo-algorithms)=

## PYGMO2 Optimizers

Please cite {cite}`Biscani2020` in addition to optimagic when using pygmo. optimagic
supports the following [pygmo2](https://esa.github.io/pygmo2) optimizers.

```{eval-rst}
.. dropdown::  pygmo_gaco

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        import numpy as np
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.pygmo_gaco(stopping_maxiter=1_000, ...),
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="pygmo_gaco",
            algo_options={"stopping_maxiter": 1_000, ...},
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.pygmo_optimizers.PygmoGaco
```

```{eval-rst}
.. dropdown::  pygmo_bee_colony

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        import numpy as np
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.pygmo_bee_colony(stopping_maxiter=1_000, ...),
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="pygmo_bee_colony",
            algo_options={"stopping_maxiter": 1_000, ...},
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.pygmo_optimizers.PygmoBeeColony
```

```{eval-rst}
.. dropdown::  pygmo_de

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        import numpy as np
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.pygmo_de(stopping_maxiter=1_000, ...),
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="pygmo_de",
            algo_options={"stopping_maxiter": 1_000, ...},
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.pygmo_optimizers.PygmoDe
```

```{eval-rst}
.. dropdown::  pygmo_sea

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        import numpy as np
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.pygmo_sea(stopping_maxiter=5_000, ...),
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="pygmo_sea",
            algo_options={"stopping_maxiter": 5_000, ...},
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.pygmo_optimizers.PygmoSea
```

```{eval-rst}
.. dropdown::  pygmo_sga

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        import numpy as np
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.pygmo_sga(stopping_maxiter=1_000, ...),
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="pygmo_sga",
            algo_options={"stopping_maxiter": 1_000, ...},
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.pygmo_optimizers.PygmoSga
```

```{eval-rst}
.. dropdown::  pygmo_sade

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        import numpy as np
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.pygmo_sade(stopping_maxiter=1_000, ...),
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="pygmo_sade",
            algo_options={"stopping_maxiter": 1_000, ...},
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.pygmo_optimizers.PygmoSade
```

```{eval-rst}
.. dropdown::  pygmo_cmaes

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        import numpy as np
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.pygmo_cmaes(stopping_maxiter=1_000, ...),
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="pygmo_cmaes",
            algo_options={"stopping_maxiter": 1_000, ...},
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.pygmo_optimizers.PygmoCmaes
```

```{eval-rst}
.. dropdown::  pygmo_simulated_annealing

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        import numpy as np
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.pygmo_simulated_annealing(start_temperature=5.0, ...),
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="pygmo_simulated_annealing",
            algo_options={"start_temperature": 5.0, ...},
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.pygmo_optimizers.PygmoSimulatedAnnealing
```

```{eval-rst}
.. dropdown::  pygmo_pso

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        import numpy as np
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.pygmo_pso(stopping_maxiter=1_000, ...),
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="pygmo_pso",
            algo_options={"stopping_maxiter": 1_000, ...},
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.pygmo_optimizers.PygmoPso
```

```{eval-rst}
.. dropdown::  pygmo_pso_gen

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        import numpy as np
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.pygmo_pso_gen(stopping_maxiter=1_000, ...),
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="pygmo_pso_gen",
            algo_options={"stopping_maxiter": 1_000, ...},
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.pygmo_optimizers.PygmoPsoGen
```

```{eval-rst}
.. dropdown::  pygmo_mbh

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        import numpy as np
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.pygmo_mbh(perturbation=0.05, ...),
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="pygmo_mbh",
            algo_options={"perturbation": 0.05, ...},
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.pygmo_optimizers.PygmoMbh
```

```{eval-rst}
.. dropdown::  pygmo_xnes

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        import numpy as np
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.pygmo_xnes(stopping_maxiter=1_000, ...),
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="pygmo_xnes",
            algo_options={"stopping_maxiter": 1_000, ...},
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.pygmo_optimizers.PygmoXnes
```

```{eval-rst}
.. dropdown::  pygmo_gwo

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        import numpy as np
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.pygmo_gwo(stopping_maxiter=1_000, ...),
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="pygmo_gwo",
            algo_options={"stopping_maxiter": 1_000, ...},
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.pygmo_optimizers.PygmoGwo
```

```{eval-rst}
.. dropdown::  pygmo_compass_search

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        import numpy as np
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.pygmo_compass_search(stopping_maxfun=1_000, ...),
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="pygmo_compass_search",
            algo_options={"stopping_maxfun": 1_000, ...},
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.pygmo_optimizers.PygmoCompassSearch
```

```{eval-rst}
.. dropdown::  pygmo_ihs

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        import numpy as np
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.pygmo_ihs(stopping_maxiter=1_000, ...),
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="pygmo_ihs",
            algo_options={"stopping_maxiter": 1_000, ...},
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.pygmo_optimizers.PygmoIhs
```

```{eval-rst}
.. dropdown::  pygmo_de1220

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        import numpy as np
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.pygmo_de1220(stopping_maxiter=1_000, ...),
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="pygmo_de1220",
            algo_options={"stopping_maxiter": 1_000, ...},
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.pygmo_optimizers.PygmoDe1220
```

(ipopt-algorithm)=

## The Interior Point Optimizer (ipopt)

optimagic's support for the Interior Point Optimizer ({cite}`Waechter2005`,
{cite}`Waechter2005a`, {cite}`Waechter2005b`, {cite}`Nocedal2009`) is built on
[cyipopt](https://cyipopt.readthedocs.io/en/latest/index.html), a Python wrapper for the
[Ipopt optimization package](https://coin-or.github.io/Ipopt/index.html).

To use ipopt, you need to have
[cyipopt installed](https://cyipopt.readthedocs.io/en/latest/index.html)
(`conda install cyipopt`).

```{eval-rst}
.. dropdown::  ipopt

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.ipopt(stopping_maxiter=1_000, ...),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="ipopt",
            algo_options={"stopping_maxiter": 1_000, ...},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.ipopt.Ipopt
```

(fides-algorithm)=

## The Fides Optimizer

optimagic supports the
[Fides Optimizer](https://fides-optimizer.readthedocs.io/en/latest). To use Fides, you
need to have [the fides package](https://github.com/fides-dev/fides) installed
(`pip install fides>=0.7.4`, make sure you have at least 0.7.1).

```{eval-rst}
.. dropdown::  fides

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.fides(hessian_update_strategy="bfgs"),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="fides",
            algo_options={"hessian_update_strategy": "bfgs"},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.fides.Fides
```

## The NLOPT Optimizers (nlopt)

optimagic supports the following [NLOPT](https://nlopt.readthedocs.io/en/latest/)
algorithms. Please add the
[appropriate citations](https://nlopt.readthedocs.io/en/latest/Citing_NLopt/) in
addition to optimagic when using an NLOPT algorithm. To install nlopt run
`conda install nlopt`.

```{eval-rst}
.. dropdown::  nlopt_bobyqa

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.nlopt_bobyqa(stopping_maxfun=10_000, ...),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="nlopt_bobyqa",
            algo_options={"stopping_maxfun": 10_000, ...},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nlopt_optimizers.NloptBOBYQA
```

```{eval-rst}
.. dropdown::  nlopt_neldermead

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.nlopt_neldermead(stopping_maxfun=10_000, ...),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="nlopt_neldermead",
            algo_options={"stopping_maxfun": 10_000, ...},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nlopt_optimizers.NloptNelderMead
```

```{eval-rst}
.. dropdown::  nlopt_praxis

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.nlopt_praxis(stopping_maxfun=10_000, ...),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="nlopt_praxis",
            algo_options={"stopping_maxfun": 10_000, ...},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nlopt_optimizers.NloptPRAXIS
```

```{eval-rst}
.. dropdown::  nlopt_cobyla

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.nlopt_cobyla(stopping_maxfun=10_000, ...),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="nlopt_cobyla",
            algo_options={"stopping_maxfun": 10_000, ...},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nlopt_optimizers.NloptCOBYLA
```

```{eval-rst}
.. dropdown::  nlopt_sbplx

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.nlopt_sbplx(stopping_maxfun=10_000, ...),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="nlopt_sbplx",
            algo_options={"stopping_maxfun": 10_000, ...},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nlopt_optimizers.NloptSbplx
```

```{eval-rst}
.. dropdown::  nlopt_newuoa

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.nlopt_newuoa(stopping_maxfun=10_000, ...),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="nlopt_newuoa",
            algo_options={"stopping_maxfun": 10_000, ...},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nlopt_optimizers.NloptNEWUOA
```

```{eval-rst}
.. dropdown::  nlopt_tnewton

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.nlopt_tnewton(stopping_maxfun=10_000, ...),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="nlopt_tnewton",
            algo_options={"stopping_maxfun": 10_000, ...},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nlopt_optimizers.NloptTNewton
```

```{eval-rst}
.. dropdown::  nlopt_lbfgsb

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.nlopt_lbfgsb(stopping_maxfun=10_000, ...),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="nlopt_lbfgsb",
            algo_options={"stopping_maxfun": 10_000, ...},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nlopt_optimizers.NloptLBFGSB
```

```{eval-rst}
.. dropdown::  nlopt_ccsaq

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.nlopt_ccsaq(stopping_maxfun=10_000, ...),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="nlopt_ccsaq",
            algo_options={"stopping_maxfun": 10_000, ...},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nlopt_optimizers.NloptCCSAQ
```

```{eval-rst}
.. dropdown::  nlopt_mma

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.nlopt_mma(stopping_maxfun=10_000, ...),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="nlopt_mma",
            algo_options={"stopping_maxfun": 10_000, ...},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nlopt_optimizers.NloptMMA
```

```{eval-rst}
.. dropdown::  nlopt_var

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.nlopt_var(rank_1_update=False, ...),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="nlopt_var",
            algo_options={"rank_1_update": False, ...},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nlopt_optimizers.NloptVAR
```

```{eval-rst}
.. dropdown::  nlopt_slsqp

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.nlopt_slsqp(stopping_maxfun=10_000, ...),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="nlopt_slsqp",
            algo_options={"stopping_maxfun": 10_000, ...},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nlopt_optimizers.NloptSLSQP
```

```{eval-rst}
.. dropdown::  nlopt_direct

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        import numpy as np
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.nlopt_direct(locally_biased=True, ...),
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="nlopt_direct",
            algo_options={"locally_biased": True, ...},
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nlopt_optimizers.NloptDirect
```

```{eval-rst}
.. dropdown::  nlopt_esch

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        import numpy as np
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.nlopt_esch(stopping_maxfun=10_000, ...),
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="nlopt_esch",
            algo_options={"stopping_maxfun": 10_000, ...},
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nlopt_optimizers.NloptESCH
```

```{eval-rst}
.. dropdown::  nlopt_isres

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        import numpy as np
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.nlopt_isres(stopping_maxfun=10_000, ...),
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="nlopt_isres",
            algo_options={"stopping_maxfun": 10_000, ...},
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nlopt_optimizers.NloptISRES
```

```{eval-rst}
.. dropdown::  nlopt_crs2_lm

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        import numpy as np
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.nlopt_crs2_lm(population_size=100, ...),
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="nlopt_crs2_lm",
            algo_options={"population_size": 100, ...},
            bounds=om.Bounds(lower=np.array([-5, -5, -5]), upper=np.array([5, 5, 5])),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nlopt_optimizers.NloptCRS2LM
```

## Optimizers from iminuit

optimagic supports the [IMINUIT MIGRAD Optimizer](https://iminuit.readthedocs.io/). To
use MIGRAD, you need to have
[the iminuit package](https://github.com/scikit-hep/iminuit) installed
(`pip install iminuit`).

```{eval-rst}
.. dropdown::  iminuit_migrad

    **How to use this algorithm:**

    .. code-block::

        import optimagic as om
        om.minimize(
          ...,
          algorithm=om.algos.iminuit_migrad(stopping_maxfun=10_000, ...)
        )
        
    or
        
    .. code-block::

        om.minimize(
          ...,
          algorithm="iminuit_migrad",
          algo_options={"stopping_maxfun=10_000, ...}
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.iminuit_migrad.IminuitMigrad

```

## Nevergrad Optimizers

optimagic supports following algorithms from the
[Nevergrad](https://facebookresearch.github.io/nevergrad/index.html) library. To use
these optimizers, you need to have
[the nevergrad package](https://github.com/facebookresearch/nevergrad) installed.
(`pip install nevergrad`).\
Two algorithms from nevergrad are not available in optimagic.\
`SPSA (Simultaneous Perturbation Stochastic Approximation)` - This is WIP in nevergrad
and hence imprecise.\
`AXP (AX-platfofm)` - Very slow and not recommended.

```{eval-rst}
.. dropdown::  nevergrad_pso

    **How to use this algorithm:**

    .. code-block::

        import optimagic as om
        om.minimize(
          ...,
          algorithm=om.algos.nevergrad_pso(stopping_maxfun=1_000, ...)
        )
        
    or
        
    .. code-block::

        om.minimize(
          ...,
          algorithm="nevergrad_pso",
          algo_options={"stopping_maxfun": 1_000, ...}
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nevergrad_optimizers.NevergradPSO

```

```{eval-rst}
.. dropdown::  nevergrad_cmaes

    **How to use this algorithm:**

    .. code-block::

        import optimagic as om
        om.minimize(
          ...,
          algorithm=om.algos.nevergrad_cmaes(stopping_maxfun=1_000, ...)
        )
        
    or
        
    .. code-block::

        om.minimize(
          ...,
          algorithm="nevergrad_cmaes",
          algo_options={"stopping_maxfun": 1_000, ...}
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nevergrad_optimizers.NevergradCMAES

```

```{eval-rst}
.. dropdown:: nevergrad_oneplusone

    **How to use this algorithm:**

    .. code-block::

        import optimagic as om
        om.minimize(
          ...,
          algorithm=om.algos.nevergrad_oneplusone(stopping_maxfun=1_000, ...)
        )

    or

    .. code-block::

        om.minimize(
          ...,
          algorithm="nevergrad_oneplusone",
          algo_options={"stopping_maxfun": 1_000, ...}
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nevergrad_optimizers.NevergradOnePlusOne
```

```{eval-rst}
.. dropdown:: nevergrad_de

    **How to use this algorithm:**

    .. code-block::

        import optimagic as om
        om.minimize(
          ...,
          algorithm=om.algos.nevergrad_de(population_size="large", ...)
        )

    or

    .. code-block::

        om.minimize(
          ...,
          algorithm="nevergrad_de",
          algo_options={"population_size": "large", ...}
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nevergrad_optimizers.NevergradDifferentialEvolution
```

```{eval-rst}
.. dropdown::  nevergrad_bo

    .. note::

        Using this optimizer requires the `bayes-optim` package to be installed as well.
        This can be done with `pip install bayes-optim`.

    **How to use this algorithm:**

    .. code-block::

        import optimagic as om
        om.minimize(
          ...,
          algorithm=om.algos.nevergrad_bo(stopping_maxfun=1_000, ...)
        )

    or

    .. code-block::

        om.minimize(
          ...,
          algorithm="nevergrad_bo",
          algo_options={"stopping_maxfun": 1_000, ...}
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nevergrad_optimizers.NevergradBayesOptim
```

```{eval-rst}
.. dropdown:: nevergrad_emna

    **How to use this algorithm:**

    .. code-block::

        import optimagic as om
        om.minimize(
          ...,
          algorithm=om.algos.nevergrad_emna(noise_handling=False, ...)
        )

    or

    .. code-block::

        om.minimize(
          ...,
          algorithm="nevergrad_emna",
          algo_options={"noise_handling": False, ...}
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nevergrad_optimizers.NevergradEMNA
```

```{eval-rst}
.. dropdown:: nevergrad_cga

    **How to use this algorithm:**

    .. code-block::

        import optimagic as om
        om.minimize(
          ...,
          algorithm=om.algos.nevergrad_cga(stopping_maxfun=10_000)
        )

    or

    .. code-block::

        om.minimize(
          ...,
          algorithm="nevergrad_cga",
          algo_options={"stopping_maxfun": 10_000}
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nevergrad_optimizers.NevergradCGA
```

```{eval-rst}
.. dropdown:: nevergrad_eda

    **How to use this algorithm:**

    .. code-block::

        import optimagic as om
        om.minimize(
          ...,
          algorithm=om.algos.nevergrad_eda(stopping_maxfun=10_000)
        )

    or

    .. code-block::

        om.minimize(
          ...,
          algorithm="nevergrad_eda",
          algo_options={"stopping_maxfun": 10_000}
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nevergrad_optimizers.NevergradEDA
```

```{eval-rst}
.. dropdown:: nevergrad_tbpsa

    **How to use this algorithm:**

    .. code-block::

        import optimagic as om
        om.minimize(
          ...,
          algorithm=om.algos.nevergrad_tbpsa(noise_handling=False, ...)
        )

    or

    .. code-block::

        om.minimize(
          ...,
          algorithm="nevergrad_tbpsa",
          algo_options={"noise_handling": False, ...}
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nevergrad_optimizers.NevergradTBPSA
```

```{eval-rst}
.. dropdown:: nevergrad_randomsearch

    **How to use this algorithm:**

    .. code-block::

        import optimagic as om
        om.minimize(
          ...,
          algorithm=om.algos.nevergrad_randomsearch(opposition_mode="quasi", ...)
        )

    or

    .. code-block::

        om.minimize(
          ...,
          algorithm="nevergrad_randomsearch",
          algo_options={"opposition_mode": "quasi", ...}
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nevergrad_optimizers.NevergradRandomSearch
```

```{eval-rst}
.. dropdown:: nevergrad_samplingsearch

    **How to use this algorithm:**

    .. code-block::

        import optimagic as om
        om.minimize(
          ...,
          algorithm=om.algos.nevergrad_samplingsearch(sampler="Hammersley", scrambled=True)
        )

    or

    .. code-block::

        om.minimize(
          ...,
          algorithm="nevergrad_samplingsearch",
          algo_options={"sampler": "Hammersley", "scrambled": True}
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nevergrad_optimizers.NevergradSamplingSearch
```

```{eval-rst}
.. dropdown:: nevergrad_ngopt

    **How to use this algorithm:**

    .. code-block::

        import optimagic as om
        om.minimize(
          ...,
          algorithm=om.algos.nevergrad_ngopt(optimizer="NGOptRW", ...)
        )

    or

    .. code-block::

        om.minimize(
          ...,
          algorithm="nevergrad_ngopt",
          algo_options={"optimizer": "NGOptRW", ...}
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nevergrad_optimizers.NevergradNGOpt
```

```{eval-rst}
.. dropdown:: nevergrad_meta

    **How to use this algorithm:**

    .. code-block::

        import optimagic as om
        om.minimize(
          ...,
          algorithm=om.algos.nevergrad_meta(optimizer="BFGSCMAPlus", ...)
        )

    or

    .. code-block::

        om.minimize(
          ...,
          algorithm="nevergrad_meta",
          algo_options={"optimizer": "BFGSCMAPlus", ...}
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nevergrad_optimizers.NevergradMeta
```

## Bayesian Optimization

We wrap the
[BayesianOptimization](https://github.com/bayesian-optimization/BayesianOptimization)
package. To use it, you need to have
[bayesian-optimization](https://pypi.org/project/bayesian-optimization/) installed.
Note: This optimizer requires `bayesian_optimization > 2.0.0` to be installed which is
incompatible with `nevergrad > 1.0.3`.

```{eval-rst}
.. dropdown::  bayes_opt

    **How to use this algorithm:**

    .. code-block::

        import optimagic as om
        om.minimize(
          ...,
          algorithm=om.algos.bayes_opt(n_iter=50, ...)
        )
        
    or
        
    .. code-block::

        om.minimize(
          ...,
          algorithm="bayes_opt",
          algo_options={"n_iter": 50, ...}
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.bayesian_optimizer.BayesOpt

```

## Gradient Free Optimizers

Optimizers from the
[gradient_free_optimizers](https://github.com/SimonBlanke/Gradient-Free-Optimizers?tab=readme-ov-file)
package are available in optimagic. To use it, you need to have
[gradient_free_optimizers](https://pypi.org/project/gradient_free_optimizers) installed.

```{eval-rst}
.. dropdown:: gfo_hillclimbing

  **How to use this algorithm.**

  .. code-block:: python

    import optimagic as om
    import numpy as np
    om.minimize(
      fun=lambda x: x @ x,
      params=[1.0, 2.0, 3.0],
      algorithm=om.algos.gfo_hillclimbing(stopping_maxiter=1_000, ...),
      bounds = om.Bounds(lower = np.array([1,1,1]), upper=np.array([5,5,5]))
    )

  or using the string interface:
      
  .. code-block:: python

    om.minimize(
      fun=lambda x: x @ x,
      params=[1.0, 2.0, 3.0],
      algorithm="gfo_hillclimbing",
      algo_options={"stopping_maxiter": 1_000, ...},
      bounds = om.Bounds(lower = np.array([1,1,1]), upper=np.array([5,5,5]))
    )

  **Description and available options:**

  .. autoclass:: optimagic.optimizers.gfo_optimizers.GFOHillClimbing
    :members:
    :inherited-members: Algorithm, object

```

```{eval-rst}
.. dropdown:: gfo_stochastichillclimbing

  **How to use this algorithm.**

  .. code-block:: python

    import optimagic as om
    import numpy as np
    om.minimize(
      fun=lambda x: x @ x,
      params=[1.0, 2.0, 3.0],
      algorithm=om.algos.gfo_stochastichillclimbing(stopping_maxiter=1_000, ...),
      bounds = om.Bounds(lower = np.array([1,1,1]), upper=np.array([5,5,5]))
    )

  or using the string interface:
      
  .. code-block:: python

    om.minimize(
      fun=lambda x: x @ x,
      params=[1.0, 2.0, 3.0],
      algorithm="gfo_stochastichillclimbing",
      algo_options={"stopping_maxiter": 1_000, ...},
      bounds = om.Bounds(lower = np.array([1,1,1]), upper=np.array([5,5,5]))
    )

  **Description and available options:**

  .. autoclass:: optimagic.optimizers.gfo_optimizers.GFOStochasticHillClimbing
    :members:
    :inherited-members: Algorithm, object  
    :member-order: bysource

```

```{eval-rst}
.. dropdown:: gfo_repulsinghillclimbing

  **How to use this algorithm.**

  .. code-block:: python

    import optimagic as om
    import numpy as np
    om.minimize(
      fun=lambda x: x @ x,
      params=[1.0, 2.0, 3.0],
      algorithm=om.algos.gfo_repulsinghillclimbing(stopping_maxiter=1_000, ...),
      bounds = om.Bounds(lower = np.array([1,1,1]), upper=np.array([5,5,5]))
    )

  or using the string interface:
      
  .. code-block:: python

    om.minimize(
      fun=lambda x: x @ x,
      params=[1.0, 2.0, 3.0],
      algorithm="gfo_repulsinghillclimbing",
      algo_options={"stopping_maxiter": 1_000, ...},
      bounds = om.Bounds(lower = np.array([1,1,1]), upper=np.array([5,5,5]))
    )

  **Description and available options:**

  .. autoclass:: optimagic.optimizers.gfo_optimizers.GFORepulsingHillClimbing
    :members:
    :inherited-members: Algorithm, object  
    :member-order: bysource

```

```{eval-rst}
.. dropdown:: gfo_simulatedannealing

  **How to use this algorithm.**

  .. code-block:: python

    import optimagic as om
    import numpy as np
    om.minimize(
      fun=lambda x: x @ x,
      params=[1.0, 2.0, 3.0],
      algorithm=om.algos.gfo_simulatedannealing(stopping_maxiter=1_000, ...),
      bounds = om.Bounds(lower = np.array([1,1,1]), upper=np.array([5,5,5]))
    )

  or using the string interface:
      
  .. code-block:: python

    om.minimize(
      fun=lambda x: x @ x,
      params=[1.0, 2.0, 3.0],
      algorithm="gfo_simulatedannealing",
      algo_options={"stopping_maxiter": 1_000, ...},
      bounds = om.Bounds(lower = np.array([1,1,1]), upper=np.array([5,5,5]))
    )

  **Description and available options:**

  .. autoclass:: optimagic.optimizers.gfo_optimizers.GFOSimulatedAnnealing
    :members:
    :inherited-members: Algorithm, object  
    :member-order: bysource

```

```{eval-rst}
.. dropdown:: gfo_downhillsimplex

  **How to use this algorithm.**

  .. code-block:: python

    import optimagic as om
    import numpy as np
    om.minimize(
      fun=lambda x: x @ x,
      params=[1.0, 2.0, 3.0],
      algorithm=om.algos.gfo_downhillsimplex(stopping_maxiter=1_000, ...),
      bounds = om.Bounds(lower = np.array([1,1,1]), upper=np.array([5,5,5]))
    )

  or using the string interface:
      
  .. code-block:: python

    om.minimize(
      fun=lambda x: x @ x,
      params=[1.0, 2.0, 3.0],
      algorithm="gfo_downhillsimplex",
      algo_options={"stopping_maxiter": 1_000, ...},
      bounds = om.Bounds(lower = np.array([1,1,1]), upper=np.array([5,5,5]))
    )

  **Description and available options:**

  .. autoclass:: optimagic.optimizers.gfo_optimizers.GFODownhillSimplex
    :members:
    :inherited-members: Algorithm, object  
    :member-order: bysource

```

```{eval-rst}
.. dropdown:: gfo_powells_method

  **How to use this algorithm.**

  .. code-block:: python

    import optimagic as om
    import numpy as np
    om.minimize(
      fun=lambda x: x @ x,
      params=[1.0, 2.0, 3.0],
      algorithm=om.algos.gfo_powells_method(stopping_maxiter=1_000, ...),
      bounds = om.Bounds(lower = np.array([1,1,1]), upper=np.array([5,5,5]))
    )

  or using the string interface:
      
  .. code-block:: python

    om.minimize(
      fun=lambda x: x @ x,
      params=[1.0, 2.0, 3.0],
      algorithm="gfo_powells_method",
      algo_options={"stopping_maxiter": 1_000, ...},
      bounds = om.Bounds(lower = np.array([1,1,1]), upper=np.array([5,5,5]))
    )

  **Description and available options:**

  .. autoclass:: optimagic.optimizers.gfo_optimizers.GFOPowellsMethod
    :members:
    :inherited-members: Algorithm, object  
    :member-order: bysource

```

```{eval-rst}
.. dropdown:: gfo_pso

  **How to use this algorithm.**

  .. code-block:: python

    import optimagic as om
    import numpy as np
    om.minimize(
      fun=lambda x: x @ x,
      params=[1.0, 2.0, 3.0],
      algorithm=om.algos.gfo_pso(stopping_maxiter=1_000, ...),
      bounds = om.Bounds(lower = np.array([1,1,1]), upper=np.array([5,5,5]))
    )

  or using the string interface:
      
  .. code-block:: python

    om.minimize(
      fun=lambda x: x @ x,
      params=[1.0, 2.0, 3.0],
      algorithm="gfo_pso",
      algo_options={"stopping_maxiter": 1_000, ...},
      bounds = om.Bounds(lower = np.array([1,1,1]), upper=np.array([5,5,5]))
    )

  **Description and available options:**

  .. autoclass:: optimagic.optimizers.gfo_optimizers.GFOParticleSwarmOptimization
    :members:
    :inherited-members: Algorithm, object  
    :member-order: bysource

```

```{eval-rst}

.. dropdown:: gfo_parallel_tempering

  **How to use this algorithm.**

  .. code-block:: python

    import optimagic as om
    import numpy as np
    om.minimize(
      fun=lambda x: x @ x,
      params=np.array([1.0, 2.0, 3.0]),
      algorithm=om.algos.gfo_parallel_tempering(population_size=15, n_iter_swap=5),
      bounds = om.Bounds(lower = np.array([1,1,1]), upper=np.array([5,5,5]))
    )

  or using the string interface:
      
  .. code-block:: python

    om.minimize(
      fun=lambda x: x @ x,
      params=np.array([1.0, 2.0, 3.0]),
      algorithm="gfo_parallel_tempering",
      algo_options={"population_size": 15, "n_iter_swap": 5},
      bounds = om.Bounds(lower = np.array([1,1,1]), upper=np.array([5,5,5]))
    )

  **Description and available options:**

  .. autoclass:: optimagic.optimizers.gfo_optimizers.GFOParallelTempering
    :members:
    :inherited-members: Algorithm, object  
    :member-order: bysource
```

```{eval-rst}
.. dropdown:: gfo_spiral_optimization

  **How to use this algorithm.**

  .. code-block:: python

    import optimagic as om
    import numpy as np
    om.minimize(
      fun=lambda x: x @ x,
      params=np.array([1.0, 2.0, 3.0]),
      algorithm=om.algos.gfo_spiral_optimization(population_size=15, decay_rate=0.95),
      bounds = om.Bounds(lower = np.array([1,1,1]), upper=np.array([5,5,5]))
    )

  or using the string interface:
      
  .. code-block:: python

    om.minimize(
      fun=lambda x: x @ x,
      params=np.array([1.0, 2.0, 3.0]),
      algorithm="gfo_spiral_optimization",
      algo_options={"population_size": 15, "decay_rate": 0.95},
      bounds = om.Bounds(lower = np.array([1,1,1]), upper=np.array([5,5,5]))
    )

  **Description and available options:**

  .. autoclass:: optimagic.optimizers.gfo_optimizers.GFOSpiralOptimization
    :members:
    :inherited-members: Algorithm, object  
    :member-order: bysource
```

```{eval-rst}
.. dropdown:: gfo_genetic_algorithm

  **How to use this algorithm.**

  .. code-block:: python

    import optimagic as om
    import numpy as np
    om.minimize(
      fun=lambda x: x @ x,
      params=np.array([1.0, 2.0, 3.0]),
      algorithm=om.algos.gfo_genetic_algorithm(population_size=20, mutation_rate=0.6),
      bounds = om.Bounds(lower = np.array([1,1,1]), upper=np.array([5,5,5]))
    )

  or using the string interface:
      
  .. code-block:: python

    om.minimize(
      fun=lambda x: x @ x,
      params=np.array([1.0, 2.0, 3.0]),
      algorithm="gfo_genetic_algorithm",
      algo_options={"population_size": 20, "mutation_rate": 0.6},
      bounds = om.Bounds(lower = np.array([1,1,1]), upper=np.array([5,5,5]))
    )

  **Description and available options:**

  .. autoclass:: optimagic.optimizers.gfo_optimizers.GFOGeneticAlgorithm
    :members:
    :inherited-members: Algorithm, object  
    :member-order: bysource
```

```{eval-rst}
.. dropdown:: gfo_evolution_strategy

  **How to use this algorithm.**

  .. code-block:: python

    import optimagic as om
    import numpy as np
    om.minimize(
      fun=lambda x: x @ x,
      params=np.array([1.0, 2.0, 3.0]),
      algorithm=om.algos.gfo_evolution_strategy(population_size=15, crossover_rate=0.4),
      bounds = om.Bounds(lower = np.array([1,1,1]), upper=np.array([5,5,5]))
    )

  or using the string interface:
      
  .. code-block:: python

    om.minimize(
      fun=lambda x: x @ x,
      params=np.array([1.0, 2.0, 3.0]),
      algorithm="gfo_evolution_strategy",
      algo_options={"population_size": 15, "crossover_rate": 0.4},
      bounds = om.Bounds(lower = np.array([1,1,1]), upper=np.array([5,5,5]))
    )

  **Description and available options:**

  .. autoclass:: optimagic.optimizers.gfo_optimizers.GFOEvolutionStrategy
    :members:
    :inherited-members: Algorithm, object  
    :member-order: bysource
```

```{eval-rst}
.. dropdown:: gfo_differential_evolution

  **How to use this algorithm.**

  .. code-block:: python

    import optimagic as om
    import numpy as np
    om.minimize(
      fun=lambda x: x @ x,
      params=np.array([1.0, 2.0, 3.0]),
      algorithm=om.algos.gfo_differential_evolution(population_size=20, mutation_rate=0.8),
      bounds = om.Bounds(lower = np.array([1,1,1]), upper=np.array([5,5,5]))
    )

  or using the string interface:
      
  .. code-block:: python

    om.minimize(
      fun=lambda x: x @ x,
      params=np.array([1.0, 2.0, 3.0]),
      algorithm="gfo_differential_evolution",
      algo_options={"population_size": 20, "mutation_rate": 0.8},
      bounds = om.Bounds(lower = np.array([1,1,1]), upper=np.array([5,5,5]))
    )

  **Description and available options:**

  .. autoclass:: optimagic.optimizers.gfo_optimizers.GFODifferentialEvolution
    :members:
    :inherited-members: Algorithm, object  
    :member-order: bysource

```

## Pygad Optimizer

We wrap the pygad optimizer. To use it you need to have
[pygad](https://pygad.readthedocs.io/en/latest/) installed.

```{eval-rst}
.. dropdown::  pygad

    **How to use this algorithm:**

    .. code-block::

        import optimagic as om
        om.minimize(
          ...,
          algorithm=om.algos.pygad(num_generations=100, ...)
        )
        
    or
        
    .. code-block::

        om.minimize(
          ...,
          algorithm="pygad",
          algo_options={"num_generations": 100, ...}
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.pygad_optimizer.Pygad
```

## PySwarms Optimizers

optimagic supports the following continuous algorithms from the
[PySwarms](https://pyswarms.readthedocs.io/en/latest/) library: (GlobalBestPSO,
LocalBestPSO, GeneralOptimizerPSO). To use these optimizers, you need to have
[the pyswarms package](https://github.com/ljvmiranda921/pyswarms) installed.
(`pip install pyswarms`).

```{eval-rst}
.. dropdown::  pyswarms_global_best

    **How to use this algorithm:**

    .. code-block::

        import optimagic as om
        om.minimize(
          ...,
          algorithm=om.algos.pyswarms_global_best(n_particles=50, ...)
        )
        
    or
        
    .. code-block::

        om.minimize(
          ...,
          algorithm="pyswarms_global_best",
          algo_options={"n_particles": 50, ...}
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.pyswarms_optimizers.PySwarmsGlobalBestPSO
      :members:
      :inherited-members: Algorithm, object

```

```{eval-rst}
.. dropdown::  pyswarms_local_best

    **How to use this algorithm:**

    .. code-block::

        import optimagic as om
        om.minimize(
          ...,
          algorithm=om.algos.pyswarms_local_best(n_particles=50, k_neighbors=3, ...)
        )
        
    or
        
    .. code-block::

        om.minimize(
          ...,
          algorithm="pyswarms_local_best",
          algo_options={"n_particles": 50, "k_neighbors": 3, ...}
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.pyswarms_optimizers.PySwarmsLocalBestPSO
      :members:
      :inherited-members: Algorithm, object

```

```{eval-rst}
.. dropdown::  pyswarms_general

    **How to use this algorithm:**

    .. code-block::

        import optimagic as om
        om.minimize(
          ...,
          algorithm=om.algos.pyswarms_general(n_particles=50, topology_type="star", ...)
        )
        
    or
        
    .. code-block::

        om.minimize(
          ...,
          algorithm="pyswarms_general",
          algo_options={"n_particles": 50, "topology_type": "star", ...}
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.pyswarms_optimizers.PySwarmsGeneralPSO
      :members:
      :inherited-members: Algorithm, object

```

## The Tranquilo Optimizer

optimagic supports [tranquilo](https://github.com/OpenSourceEconomics/tranquilo), a
trust-region optimizer for noisy and/or computationally expensive black-box problems
that was developed by the optimagic developers. To use it, you need to have the
tranquilo package (version 0.1.0 or newer) installed, e.g. via `pip install tranquilo`
or `conda install -c conda-forge tranquilo`. `tranquilo` is the scalar version of the
algorithm; `tranquilo_ls` exploits the least-squares structure of the objective function
and should be preferred whenever your objective function can be expressed as a sum of
squared residuals.

```{eval-rst}
.. dropdown::  tranquilo

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om
        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.tranquilo(stopping_maxfun=5_000, ...),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=lambda x: x @ x,
            params=[1.0, 2.0, 3.0],
            algorithm="tranquilo",
            algo_options={"stopping_maxfun": 5_000, ...},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.tranquilo.Tranquilo
```

```{eval-rst}
.. dropdown::  tranquilo_ls

    **How to use this algorithm:**

    .. code-block:: python

        import optimagic as om


        @om.mark.least_squares
        def fun(x):
            return x


        om.minimize(
            fun=fun,
            params=[1.0, 2.0, 3.0],
            algorithm=om.algos.tranquilo_ls(stopping_maxfun=5_000, ...),
        )

    or using the string interface:

    .. code-block:: python

        om.minimize(
            fun=fun,
            params=[1.0, 2.0, 3.0],
            algorithm="tranquilo_ls",
            algo_options={"stopping_maxfun": 5_000, ...},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.tranquilo.TranquiloLS
```

## References

```{eval-rst}
.. bibliography:: refs.bib
    :labelprefix: algo_
    :filter: docname in docnames
    :style: unsrt
```
