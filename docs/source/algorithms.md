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

## Optimizers from iminuit

optimagic supports the [IMINUIT MIGRAD Optimizer](https://iminuit.readthedocs.io/). To
use MIGRAD, you need to have
[the iminuit package](https://github.com/scikit-hep/iminuit) installed (pip install
iminuit).

```{eval-rst}
.. dropdown::  iminuit_migrad

    .. code-block::

        "iminuit_migrad"

    `MIGRAD <https://iminuit.readthedocs.io/en/stable/reference.html#iminuit.Minuit.migrad>`_ is 
    the workhorse algorithm of the MINUIT optimization suite, which has been widely used in the 
    high-energy physics community since 1975. The IMINUIT package is a Python interface to the 
    Minuit2 C++ library developed by CERN.

    Migrad uses a quasi-Newton method, updating the Hessian matrix iteratively
    to guide the optimization. The algorithm adapts dynamically to challenging landscapes
    using several key techniques:

    - **Quasi-Newton updates**: The Hessian is updated iteratively rather than recalculated at 
      each step, improving efficiency.
    - **Steepest descent fallback**: When the Hessian update fails, Migrad falls back to steepest
      descent with line search.
    - **Box constraints handling**: Parameters with bounds are transformed internally to ensure 
      they remain within allowed limits.
    - **Heuristics for numerical stability**: Special cases such as flat gradients or singular 
      Hessians are managed using pre-defined heuristics.
    - **Stopping criteria based on Estimated Distance to Minimum (EDM)**: The optimization halts 
      when the predicted improvement becomes sufficiently small.
              
    For details see :cite:`JAMES1975343`.

    **Optimizer Parameters:**  

    - **stopping.maxfun** (int): Maximum number of function evaluations. If reached, the optimization stops 
      but this is not counted as successful convergence. Function evaluations used for numerical gradient 
      calculations do not count toward this limit. Default is 1,000,000.

    - **n_restarts** (int): Number of times to restart the optimizer if convergence is not reached.

      - A value of 1 (the default) indicates that the optimizer will only run once, disabling the restart feature.  
      - Values greater than 1 specify the maximum number of restart attempts.  
```

(nevergrad-algorithms)=

## Nevergrad Optimizers

optimagic supports some algorithms from the
[Nevergrad](https://facebookresearch.github.io/nevergrad/index.html) library. To use
these optimizers, you need to have
[the nevergrad package](https://github.com/facebookresearch/nevergrad) installed.
(`pip install nevergrad`).

```{eval-rst}
.. dropdown:: nevergrad_pso

    .. code-block::

        "nevergrad_pso"

    Minimize a scalar function using the Particle Swarm Optimization (PSO) algorithm.

    The Particle Swarm Optimization algorithm was originally proposed by
    :cite:`Kennedy1995`. The implementation in Nevergrad is based on
    :cite:`Zambrano2013`.

    Particle Swarm Optimization (PSO) solves a problem by having a population of
    candidate solutions, here dubbed particles, and moving these particles around in the
    search-space according to simple mathematical formulae over the particle's position
    and velocity. Each particle's movement is influenced by its local best known
    position (termed "cognitive" component), but is also guided toward the best known
    positions (termed "social" component) in the search-space, which are updated as
    better positions are found by other particles. This is expected to move the swarm
    toward the best solutions.

    - **transform** (str): The transform to use to map from PSO optimization space to
      R-space. Available options are:
      - "arctan" (default)
      - "identity"
      - "gaussian"
    - **population_size** (int): Population size of the particle swarm.
    - **n_cores** (int): Number of cores to use.
    - **seed** (int): Seed used by the internal random number generator.
    - **stopping.maxfun** (int): Maximum number of function evaluations.
    - **inertia** (float): Inertia weight. Denoted by :math:`\omega`.
      Default is 0.7213475204444817. To prevent divergence, the value must be smaller
      than 1. It controls the influence of the particle's previous velocity on its
      movement.
    - **cognitive** (float): Cognitive coefficient. Denoted by :math:`\phi_p`.
      Default is 1.1931471805599454. Typical values range from 1.0 to 3.0. It controls
      the influence of its own best known position on the particle's movement.
    - **social** (float): Social coefficient. Denoted by :math:`\phi_g`.
      Default is 1.1931471805599454. Typical values range from 1.0 to 3.0. It controls
      the influence of the swarm's best known position on the particle's movement.
    - **quasi_opp_init** (bool): Whether to use quasi-opposition initialization.
      Default is False.
    - **speed_quasi_opp_init** (bool): Whether to use quasi-opposition initialization
      for speed. Default is False.
    - **special_speed_quasi_opp_init** (bool): Whether to use special quasi-opposition
      initialization for speed. Default is False.
```

## References

```{eval-rst}
.. bibliography:: refs.bib
    :labelprefix: algo_
    :filter: docname in docnames
    :style: unsrt
```
