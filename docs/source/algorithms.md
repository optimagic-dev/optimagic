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

```{eval-rst}
.. dropdown:: scipy_slsqp

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.scipy_slsqp(stopping_maxiter=1_000),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="scipy_slsqp",
            algo_options={"stopping_maxiter": 1_000},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.scipy_optimizers.ScipySLSQP
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: scipy_neldermead

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.scipy_neldermead(stopping_maxiter=1_000),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="scipy_neldermead",
            algo_options={"stopping_maxiter": 1_000},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.scipy_optimizers.ScipyNelderMead
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: scipy_powell

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.scipy_powell(stopping_maxiter=1_000),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="scipy_powell",
            algo_options={"stopping_maxiter": 1_000},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.scipy_optimizers.ScipyPowell
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: scipy_bfgs

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.scipy_bfgs(stopping_maxiter=1_000),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="scipy_bfgs",
            algo_options={"stopping_maxiter": 1_000},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.scipy_optimizers.ScipyBFGS
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: scipy_conjugate_gradient

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.scipy_conjugate_gradient(stopping_maxiter=1_000),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="scipy_conjugate_gradient",
            algo_options={"stopping_maxiter": 1_000},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.scipy_optimizers.ScipyConjugateGradient
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: scipy_newton_cg

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.scipy_newton_cg(stopping_maxiter=1_000),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="scipy_newton_cg",
            algo_options={"stopping_maxiter": 1_000},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.scipy_optimizers.ScipyNewtonCG
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: scipy_cobyla

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.scipy_cobyla(stopping_maxiter=1_000),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="scipy_cobyla",
            algo_options={"stopping_maxiter": 1_000},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.scipy_optimizers.ScipyCOBYLA
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: scipy_truncated_newton

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.scipy_truncated_newton(stopping_maxfun=100_000),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="scipy_truncated_newton",
            algo_options={"stopping_maxfun": 100_000},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.scipy_optimizers.ScipyTruncatedNewton
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: scipy_trust_constr

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.scipy_trust_constr(stopping_maxiter=1_000),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="scipy_trust_constr",
            algo_options={"stopping_maxiter": 1_000},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.scipy_optimizers.ScipyTrustConstr
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: scipy_ls_dogbox

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om


        @om.mark.least_squares
        def fun(x):
            return x


        om.minimize(
            fun=fun,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.scipy_ls_dogbox(stopping_maxfun=1_000),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om


        @om.mark.least_squares
        def fun(x):
            return x


        om.minimize(
            fun=fun,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="scipy_ls_dogbox",
            algo_options={"stopping_maxfun": 1_000},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.scipy_optimizers.ScipyLSDogbox
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: scipy_ls_trf

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om


        @om.mark.least_squares
        def fun(x):
            return x


        om.minimize(
            fun=fun,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.scipy_ls_trf(stopping_maxfun=1_000),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om


        @om.mark.least_squares
        def fun(x):
            return x


        om.minimize(
            fun=fun,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="scipy_ls_trf",
            algo_options={"stopping_maxfun": 1_000},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.scipy_optimizers.ScipyLSTRF
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: scipy_ls_lm

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om


        @om.mark.least_squares
        def fun(x):
            return x


        om.minimize(
            fun=fun,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.scipy_ls_lm(stopping_maxfun=1_000),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om


        @om.mark.least_squares
        def fun(x):
            return x


        om.minimize(
            fun=fun,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="scipy_ls_lm",
            algo_options={"stopping_maxfun": 1_000},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.scipy_optimizers.ScipyLSLM
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: scipy_basinhopping

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.scipy_basinhopping(n_local_optimizations=10),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="scipy_basinhopping",
            algo_options={"n_local_optimizations": 10},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.scipy_optimizers.ScipyBasinhopping
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: scipy_brute

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.scipy_brute(n_grid_points=50),
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="scipy_brute",
            algo_options={"n_grid_points": 50},
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.scipy_optimizers.ScipyBrute
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: scipy_differential_evolution

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.scipy_differential_evolution(
                population_size_multiplier=10,
            ),
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="scipy_differential_evolution",
            algo_options={"population_size_multiplier": 10},
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.scipy_optimizers.ScipyDifferentialEvolution
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: scipy_shgo

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.scipy_shgo(n_sampling_points=256),
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="scipy_shgo",
            algo_options={"n_sampling_points": 256},
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.scipy_optimizers.ScipySHGO
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: scipy_dual_annealing

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.scipy_dual_annealing(stopping_maxiter=1_000),
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="scipy_dual_annealing",
            algo_options={"stopping_maxiter": 1_000},
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.scipy_optimizers.ScipyDualAnnealing
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: scipy_direct

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.scipy_direct(stopping_maxfun=10_000),
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="scipy_direct",
            algo_options={"stopping_maxfun": 10_000},
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.scipy_optimizers.ScipyDirect
        :members:
        :inherited-members: Algorithm, object
```

(own-algorithms)=

## Own optimizers

We implement a few algorithms from scratch. They are currently considered experimental.

```{eval-rst}
.. dropdown:: bhhh

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om


        @om.mark.likelihood
        def fun(x):
            return x**2


        om.minimize(
            fun=fun,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.bhhh(stopping_maxiter=1_000),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om


        @om.mark.likelihood
        def fun(x):
            return x**2


        om.minimize(
            fun=fun,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="bhhh",
            algo_options={"stopping_maxiter": 1_000},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.bhhh.BHHH
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: neldermead_parallel

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.neldermead_parallel(n_cores=2),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="neldermead_parallel",
            algo_options={"n_cores": 2},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.neldermead.NelderMeadParallel
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: pounders

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om


        @om.mark.least_squares
        def fun(x):
            return x


        om.minimize(
            fun=fun,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.pounders(stopping_maxiter=1_000),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om


        @om.mark.least_squares
        def fun(x):
            return x


        om.minimize(
            fun=fun,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="pounders",
            algo_options={"stopping_maxiter": 1_000},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.pounders.Pounders
        :members:
        :inherited-members: Algorithm, object
```

(tao-algorithms)=

## Optimizers from the Toolkit for Advanced Optimization (TAO)

We wrap the pounders algorithm from the Toolkit for Advanced Optimization. To use it you
need to have [petsc4py](https://pypi.org/project/petsc4py/) installed.

```{eval-rst}
.. dropdown:: tao_pounders

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om


        @om.mark.least_squares
        def fun(x):
            return x


        om.minimize(
            fun=fun,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.tao_pounders(stopping_maxiter=1_000),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om


        @om.mark.least_squares
        def fun(x):
            return x


        om.minimize(
            fun=fun,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="tao_pounders",
            algo_options={"stopping_maxiter": 1_000},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.tao_optimizers.TAOPounders
        :members:
        :inherited-members: Algorithm, object
```

(nag-algorithms)=

## Optimizers from the Numerical Algorithms Group (NAG)

We wrap two algorithms from the numerical algorithms group. To use them, you need to
install each of them separately:

- `pip install DFO-LS`
- `pip install Py-BOBYQA`

```{eval-rst}
.. dropdown:: nag_dfols

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om


        @om.mark.least_squares
        def fun(x):
            return x


        om.minimize(
            fun=fun,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.nag_dfols(stopping_maxfun=10_000),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om


        @om.mark.least_squares
        def fun(x):
            return x


        om.minimize(
            fun=fun,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="nag_dfols",
            algo_options={"stopping_maxfun": 10_000},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nag_optimizers.NagDFOLS
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: nag_pybobyqa

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.nag_pybobyqa(
                convergence_minimal_trustregion_radius_tolerance=1e-6,
            ),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="nag_pybobyqa",
            algo_options={"convergence_minimal_trustregion_radius_tolerance": 1e-6},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nag_optimizers.NagPyBOBYQA
        :members:
        :inherited-members: Algorithm, object
```

(pygmo-algorithms)=

## PYGMO2 Optimizers

Please cite {cite}`Biscani2020` in addition to optimagic when using pygmo. optimagic
supports the following [pygmo2](https://esa.github.io/pygmo2) optimizers.

```{eval-rst}
.. dropdown:: pygmo_gaco

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.pygmo_gaco(stopping_maxiter=1_000),
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="pygmo_gaco",
            algo_options={"stopping_maxiter": 1_000},
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.pygmo_optimizers.PygmoGaco
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: pygmo_bee_colony

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.pygmo_bee_colony(stopping_maxiter=1_000),
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="pygmo_bee_colony",
            algo_options={"stopping_maxiter": 1_000},
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.pygmo_optimizers.PygmoBeeColony
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: pygmo_de

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.pygmo_de(stopping_maxiter=1_000),
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="pygmo_de",
            algo_options={"stopping_maxiter": 1_000},
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.pygmo_optimizers.PygmoDe
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: pygmo_sea

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.pygmo_sea(stopping_maxiter=5_000),
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="pygmo_sea",
            algo_options={"stopping_maxiter": 5_000},
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.pygmo_optimizers.PygmoSea
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: pygmo_sga

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.pygmo_sga(stopping_maxiter=1_000),
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="pygmo_sga",
            algo_options={"stopping_maxiter": 1_000},
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.pygmo_optimizers.PygmoSga
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: pygmo_sade

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.pygmo_sade(stopping_maxiter=1_000),
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="pygmo_sade",
            algo_options={"stopping_maxiter": 1_000},
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.pygmo_optimizers.PygmoSade
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: pygmo_cmaes

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.pygmo_cmaes(stopping_maxiter=1_000),
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="pygmo_cmaes",
            algo_options={"stopping_maxiter": 1_000},
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.pygmo_optimizers.PygmoCmaes
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: pygmo_simulated_annealing

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.pygmo_simulated_annealing(start_temperature=5.0),
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="pygmo_simulated_annealing",
            algo_options={"start_temperature": 5.0},
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.pygmo_optimizers.PygmoSimulatedAnnealing
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: pygmo_pso

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.pygmo_pso(stopping_maxiter=1_000),
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="pygmo_pso",
            algo_options={"stopping_maxiter": 1_000},
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.pygmo_optimizers.PygmoPso
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: pygmo_pso_gen

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.pygmo_pso_gen(stopping_maxiter=1_000),
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="pygmo_pso_gen",
            algo_options={"stopping_maxiter": 1_000},
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.pygmo_optimizers.PygmoPsoGen
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: pygmo_mbh

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.pygmo_mbh(perturbation=0.05),
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="pygmo_mbh",
            algo_options={"perturbation": 0.05},
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.pygmo_optimizers.PygmoMbh
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: pygmo_xnes

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.pygmo_xnes(stopping_maxiter=1_000),
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="pygmo_xnes",
            algo_options={"stopping_maxiter": 1_000},
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.pygmo_optimizers.PygmoXnes
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: pygmo_gwo

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.pygmo_gwo(stopping_maxiter=1_000),
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="pygmo_gwo",
            algo_options={"stopping_maxiter": 1_000},
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.pygmo_optimizers.PygmoGwo
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: pygmo_compass_search

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.pygmo_compass_search(stopping_maxfun=1_000),
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="pygmo_compass_search",
            algo_options={"stopping_maxfun": 1_000},
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.pygmo_optimizers.PygmoCompassSearch
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: pygmo_ihs

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.pygmo_ihs(stopping_maxiter=1_000),
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="pygmo_ihs",
            algo_options={"stopping_maxiter": 1_000},
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.pygmo_optimizers.PygmoIhs
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: pygmo_de1220

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.pygmo_de1220(stopping_maxiter=1_000),
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="pygmo_de1220",
            algo_options={"stopping_maxiter": 1_000},
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.pygmo_optimizers.PygmoDe1220
        :members:
        :inherited-members: Algorithm, object
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
.. dropdown:: ipopt

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.ipopt(stopping_maxiter=1_000),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="ipopt",
            algo_options={"stopping_maxiter": 1_000},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.ipopt.Ipopt
        :members:
        :inherited-members: Algorithm, object
```

(fides-algorithm)=

## The Fides Optimizer

optimagic supports the
[Fides Optimizer](https://fides-optimizer.readthedocs.io/en/latest). To use Fides, you
need to have [the fides package](https://github.com/fides-dev/fides) installed
(`pip install fides>=0.7.4`).

```{eval-rst}
.. dropdown:: fides

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.fides(hessian_update_strategy="bfgs"),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="fides",
            algo_options={"hessian_update_strategy": "bfgs"},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.fides.Fides
        :members:
        :inherited-members: Algorithm, object
```

## The NLOPT Optimizers (nlopt)

optimagic supports the following [NLOPT](https://nlopt.readthedocs.io/en/latest/)
algorithms. Please add the
[appropriate citations](https://nlopt.readthedocs.io/en/latest/Citing_NLopt/) in
addition to optimagic when using an NLOPT algorithm. To install nlopt run
`conda install nlopt`.

PRAXIS, ESCH, ISRES, CRS2-LM, and randomized DIRECT accept a `seed` option. It must be
an integer between 0 and `2**32 - 1`, inclusive. When set, it initializes NLopt's random
number generator before the optimization. When `None`, optimagic leaves NLopt's existing
random state unchanged. The random state is shared by NLopt optimizers on the same
thread in builds with thread-local storage, and may be shared across threads in other
builds. Nested optimizations can therefore change the random stream of an outer
optimization. The deterministic DIRECT variants do not use the seed.

```{eval-rst}
.. dropdown:: nlopt_bobyqa

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.nlopt_bobyqa(stopping_maxfun=10_000),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="nlopt_bobyqa",
            algo_options={"stopping_maxfun": 10_000},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nlopt_optimizers.NloptBOBYQA
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: nlopt_neldermead

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.nlopt_neldermead(stopping_maxfun=10_000),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="nlopt_neldermead",
            algo_options={"stopping_maxfun": 10_000},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nlopt_optimizers.NloptNelderMead
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: nlopt_praxis

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.nlopt_praxis(stopping_maxfun=10_000),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="nlopt_praxis",
            algo_options={"stopping_maxfun": 10_000},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nlopt_optimizers.NloptPRAXIS
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: nlopt_cobyla

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.nlopt_cobyla(stopping_maxfun=10_000),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="nlopt_cobyla",
            algo_options={"stopping_maxfun": 10_000},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nlopt_optimizers.NloptCOBYLA
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: nlopt_sbplx

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.nlopt_sbplx(stopping_maxfun=10_000),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="nlopt_sbplx",
            algo_options={"stopping_maxfun": 10_000},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nlopt_optimizers.NloptSbplx
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: nlopt_newuoa

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.nlopt_newuoa(stopping_maxfun=10_000),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="nlopt_newuoa",
            algo_options={"stopping_maxfun": 10_000},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nlopt_optimizers.NloptNEWUOA
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: nlopt_tnewton

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.nlopt_tnewton(stopping_maxfun=10_000),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="nlopt_tnewton",
            algo_options={"stopping_maxfun": 10_000},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nlopt_optimizers.NloptTNewton
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: nlopt_lbfgsb

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.nlopt_lbfgsb(stopping_maxfun=10_000),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="nlopt_lbfgsb",
            algo_options={"stopping_maxfun": 10_000},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nlopt_optimizers.NloptLBFGSB
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: nlopt_ccsaq

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.nlopt_ccsaq(stopping_maxfun=10_000),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="nlopt_ccsaq",
            algo_options={"stopping_maxfun": 10_000},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nlopt_optimizers.NloptCCSAQ
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: nlopt_mma

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.nlopt_mma(stopping_maxfun=10_000),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="nlopt_mma",
            algo_options={"stopping_maxfun": 10_000},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nlopt_optimizers.NloptMMA
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: nlopt_var

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.nlopt_var(rank_1_update=False),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="nlopt_var",
            algo_options={"rank_1_update": False},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nlopt_optimizers.NloptVAR
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: nlopt_slsqp

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.nlopt_slsqp(stopping_maxfun=10_000),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="nlopt_slsqp",
            algo_options={"stopping_maxfun": 10_000},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nlopt_optimizers.NloptSLSQP
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: nlopt_direct

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.nlopt_direct(locally_biased=True),
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="nlopt_direct",
            algo_options={"locally_biased": True},
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nlopt_optimizers.NloptDirect
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: nlopt_esch

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.nlopt_esch(stopping_maxfun=10_000),
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="nlopt_esch",
            algo_options={"stopping_maxfun": 10_000},
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nlopt_optimizers.NloptESCH
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: nlopt_isres

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.nlopt_isres(stopping_maxfun=10_000),
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="nlopt_isres",
            algo_options={"stopping_maxfun": 10_000},
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nlopt_optimizers.NloptISRES
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: nlopt_crs2_lm

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.nlopt_crs2_lm(population_size=100),
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="nlopt_crs2_lm",
            algo_options={"population_size": 100},
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nlopt_optimizers.NloptCRS2LM
        :members:
        :inherited-members: Algorithm, object
```

## Optimizers from iminuit

optimagic supports the [IMINUIT MIGRAD Optimizer](https://iminuit.readthedocs.io/). To
use MIGRAD, you need to have
[the iminuit package](https://github.com/scikit-hep/iminuit) installed
(`pip install iminuit`).

```{eval-rst}
.. dropdown:: iminuit_migrad

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.iminuit_migrad(stopping_maxfun=10_000),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="iminuit_migrad",
            algo_options={"stopping_maxfun": 10_000},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.iminuit_migrad.IminuitMigrad
        :members:
        :inherited-members: Algorithm, object
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
`AXP (AX-platform)` - Very slow and not recommended.

```{eval-rst}
.. dropdown:: nevergrad_pso

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.nevergrad_pso(stopping_maxfun=1_000),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="nevergrad_pso",
            algo_options={"stopping_maxfun": 1_000},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nevergrad_optimizers.NevergradPSO
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: nevergrad_cmaes

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.nevergrad_cmaes(stopping_maxfun=1_000),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="nevergrad_cmaes",
            algo_options={"stopping_maxfun": 1_000},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nevergrad_optimizers.NevergradCMAES
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: nevergrad_oneplusone

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.nevergrad_oneplusone(stopping_maxfun=1_000),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="nevergrad_oneplusone",
            algo_options={"stopping_maxfun": 1_000},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nevergrad_optimizers.NevergradOnePlusOne
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: nevergrad_de

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.nevergrad_de(population_size="large"),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="nevergrad_de",
            algo_options={"population_size": "large"},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nevergrad_optimizers.NevergradDifferentialEvolution
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: nevergrad_bo

    .. note::

        Using this optimizer requires the `bayes-optim` package to be installed as well.
        This can be done with `pip install bayes-optim`.

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.nevergrad_bo(stopping_maxfun=1_000),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="nevergrad_bo",
            algo_options={"stopping_maxfun": 1_000},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nevergrad_optimizers.NevergradBayesOptim
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: nevergrad_emna

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.nevergrad_emna(noise_handling=False),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="nevergrad_emna",
            algo_options={"noise_handling": False},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nevergrad_optimizers.NevergradEMNA
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: nevergrad_cga

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.nevergrad_cga(stopping_maxfun=10_000),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="nevergrad_cga",
            algo_options={"stopping_maxfun": 10_000},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nevergrad_optimizers.NevergradCGA
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: nevergrad_eda

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.nevergrad_eda(stopping_maxfun=10_000),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="nevergrad_eda",
            algo_options={"stopping_maxfun": 10_000},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nevergrad_optimizers.NevergradEDA
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: nevergrad_tbpsa

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.nevergrad_tbpsa(noise_handling=False),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="nevergrad_tbpsa",
            algo_options={"noise_handling": False},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nevergrad_optimizers.NevergradTBPSA
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: nevergrad_randomsearch

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.nevergrad_randomsearch(opposition_mode="quasi"),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="nevergrad_randomsearch",
            algo_options={"opposition_mode": "quasi"},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nevergrad_optimizers.NevergradRandomSearch
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: nevergrad_samplingsearch

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.nevergrad_samplingsearch(
                sampler="Hammersley",
                scrambled=True,
            ),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="nevergrad_samplingsearch",
            algo_options={"sampler": "Hammersley", "scrambled": True},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nevergrad_optimizers.NevergradSamplingSearch
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: nevergrad_ngopt

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.nevergrad_ngopt(optimizer="NGOptRW"),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="nevergrad_ngopt",
            algo_options={"optimizer": "NGOptRW"},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nevergrad_optimizers.NevergradNGOpt
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: nevergrad_meta

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.nevergrad_meta(optimizer="BFGSCMAPlus"),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="nevergrad_meta",
            algo_options={"optimizer": "BFGSCMAPlus"},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.nevergrad_optimizers.NevergradMeta
        :members:
        :inherited-members: Algorithm, object
```

## Bayesian Optimization

We wrap the
[BayesianOptimization](https://github.com/bayesian-optimization/BayesianOptimization)
package. To use it, you need to have
[bayesian-optimization](https://pypi.org/project/bayesian-optimization/) installed.
Note: This optimizer requires `bayesian_optimization > 2.0.0` to be installed which is
incompatible with `nevergrad > 1.0.3`.

```{eval-rst}
.. dropdown:: bayes_opt

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.bayes_opt(stopping_maxiter=50),
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="bayes_opt",
            algo_options={"stopping_maxiter": 50},
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.bayesian_optimizer.BayesOpt
        :members:
        :inherited-members: Algorithm, object
```

## Gradient Free Optimizers

Optimizers from the
[gradient_free_optimizers](https://github.com/SimonBlanke/Gradient-Free-Optimizers?tab=readme-ov-file)
package are available in optimagic. To use it, you need to have
[gradient_free_optimizers](https://pypi.org/project/gradient_free_optimizers) installed.

```{eval-rst}
.. dropdown:: gfo_hillclimbing

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.gfo_hillclimbing(stopping_maxiter=1_000),
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="gfo_hillclimbing",
            algo_options={"stopping_maxiter": 1_000},
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.gfo_optimizers.GFOHillClimbing
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: gfo_stochastichillclimbing

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.gfo_stochastichillclimbing(stopping_maxiter=1_000),
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="gfo_stochastichillclimbing",
            algo_options={"stopping_maxiter": 1_000},
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.gfo_optimizers.GFOStochasticHillClimbing
        :members:
        :inherited-members: Algorithm, object
        :member-order: bysource
```

```{eval-rst}
.. dropdown:: gfo_repulsinghillclimbing

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.gfo_repulsinghillclimbing(stopping_maxiter=1_000),
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="gfo_repulsinghillclimbing",
            algo_options={"stopping_maxiter": 1_000},
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.gfo_optimizers.GFORepulsingHillClimbing
        :members:
        :inherited-members: Algorithm, object
        :member-order: bysource
```

```{eval-rst}
.. dropdown:: gfo_simulatedannealing

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.gfo_simulatedannealing(stopping_maxiter=1_000),
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="gfo_simulatedannealing",
            algo_options={"stopping_maxiter": 1_000},
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.gfo_optimizers.GFOSimulatedAnnealing
        :members:
        :inherited-members: Algorithm, object
        :member-order: bysource
```

```{eval-rst}
.. dropdown:: gfo_downhillsimplex

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.gfo_downhillsimplex(stopping_maxiter=1_000),
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="gfo_downhillsimplex",
            algo_options={"stopping_maxiter": 1_000},
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.gfo_optimizers.GFODownhillSimplex
        :members:
        :inherited-members: Algorithm, object
        :member-order: bysource
```

```{eval-rst}
.. dropdown:: gfo_powells_method

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.gfo_powells_method(stopping_maxiter=1_000),
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="gfo_powells_method",
            algo_options={"stopping_maxiter": 1_000},
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.gfo_optimizers.GFOPowellsMethod
        :members:
        :inherited-members: Algorithm, object
        :member-order: bysource
```

```{eval-rst}
.. dropdown:: gfo_pso

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.gfo_pso(stopping_maxiter=1_000),
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="gfo_pso",
            algo_options={"stopping_maxiter": 1_000},
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.gfo_optimizers.GFOParticleSwarmOptimization
        :members:
        :inherited-members: Algorithm, object
        :member-order: bysource
```

```{eval-rst}
.. dropdown:: gfo_parallel_tempering

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.gfo_parallel_tempering(
                population_size=15,
                n_iter_swap=5,
            ),
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="gfo_parallel_tempering",
            algo_options={"population_size": 15, "n_iter_swap": 5},
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.gfo_optimizers.GFOParallelTempering
        :members:
        :inherited-members: Algorithm, object
        :member-order: bysource
```

```{eval-rst}
.. dropdown:: gfo_spiral_optimization

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.gfo_spiral_optimization(
                population_size=15,
                decay_rate=0.95,
            ),
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="gfo_spiral_optimization",
            algo_options={"population_size": 15, "decay_rate": 0.95},
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.gfo_optimizers.GFOSpiralOptimization
        :members:
        :inherited-members: Algorithm, object
        :member-order: bysource
```

```{eval-rst}
.. dropdown:: gfo_genetic_algorithm

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.gfo_genetic_algorithm(
                population_size=20,
                mutation_rate=0.6,
            ),
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="gfo_genetic_algorithm",
            algo_options={"population_size": 20, "mutation_rate": 0.6},
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.gfo_optimizers.GFOGeneticAlgorithm
        :members:
        :inherited-members: Algorithm, object
        :member-order: bysource
```

```{eval-rst}
.. dropdown:: gfo_evolution_strategy

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.gfo_evolution_strategy(
                population_size=15,
                crossover_rate=0.4,
            ),
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="gfo_evolution_strategy",
            algo_options={"population_size": 15, "crossover_rate": 0.4},
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.gfo_optimizers.GFOEvolutionStrategy
        :members:
        :inherited-members: Algorithm, object
        :member-order: bysource
```

```{eval-rst}
.. dropdown:: gfo_differential_evolution

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.gfo_differential_evolution(
                population_size=20,
                mutation_rate=0.8,
            ),
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="gfo_differential_evolution",
            algo_options={"population_size": 20, "mutation_rate": 0.8},
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
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
.. dropdown:: pygad

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.pygad(num_generations=100),
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="pygad",
            algo_options={"num_generations": 100},
            bounds=om.Bounds(lower=np.full(3, -5.0), upper=np.full(3, 5.0)),
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.pygad_optimizer.Pygad
        :members:
        :inherited-members: Algorithm, object
```

## PySwarms Optimizers

optimagic supports the following continuous algorithms from the
[PySwarms](https://pyswarms.readthedocs.io/en/latest/) library: (GlobalBestPSO,
LocalBestPSO, GeneralOptimizerPSO). To use these optimizers, you need to have
[the pyswarms package](https://github.com/ljvmiranda921/pyswarms) installed.
(`pip install pyswarms`).

```{eval-rst}
.. dropdown:: pyswarms_global_best

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.pyswarms_global_best(n_particles=50),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="pyswarms_global_best",
            algo_options={"n_particles": 50},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.pyswarms_optimizers.PySwarmsGlobalBestPSO
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: pyswarms_local_best

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.pyswarms_local_best(n_particles=50),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="pyswarms_local_best",
            algo_options={"n_particles": 50},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.pyswarms_optimizers.PySwarmsLocalBestPSO
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: pyswarms_general

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.pyswarms_general(n_particles=50, topology="star"),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="pyswarms_general",
            algo_options={"n_particles": 50, "topology": "star"},
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
.. dropdown:: tranquilo

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.tranquilo(stopping_maxfun=5_000),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om

        om.minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="tranquilo",
            algo_options={"stopping_maxfun": 5_000},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.tranquilo.Tranquilo
        :members:
        :inherited-members: Algorithm, object
```

```{eval-rst}
.. dropdown:: tranquilo_ls

    **How to use this algorithm:**

    .. code-block:: python

        import numpy as np
        import optimagic as om


        @om.mark.least_squares
        def fun(x):
            return x


        om.minimize(
            fun=fun,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm=om.algos.tranquilo_ls(stopping_maxfun=5_000),
        )

    or using the string interface:

    .. code-block:: python

        import numpy as np
        import optimagic as om


        @om.mark.least_squares
        def fun(x):
            return x


        om.minimize(
            fun=fun,
            params=np.array([1.0, 2.0, 3.0]),
            algorithm="tranquilo_ls",
            algo_options={"stopping_maxfun": 5_000},
        )

    **Description and available options:**

    .. autoclass:: optimagic.optimizers.tranquilo.TranquiloLS
        :members:
        :inherited-members: Algorithm, object
```

## References

```{eval-rst}
.. bibliography:: refs.bib
    :labelprefix: algo_
    :filter: docname in docnames
    :style: unsrt
```
