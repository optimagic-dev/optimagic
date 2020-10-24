===========================================
Compact guide on how to choose an optimizer
===========================================

1. Global vs. local vs. globalized optimizer
============================================

Most of the time, one is interested in finding a global optimum of a function. There are several approaches to finding it:

* Using a **local optimizer** if you know that your function only has one optimum, and thus the local optimum is also the global one
* Use a **global optimizer** (e.g. a genetic algorithm or bayesian optimization)
* Start local optimization algorithms from several starting points (**globalized optiimizer**)

All global algorithms have in common that they require lower and upper bounds on all parameters: No solutions outside the bounds are ever considered.
They then try to explore a large portion of parameter space.

As an example, a bayesian algorithm does so by building a global surrogate model of the criterion function, while a genetic algorithm samples an initial population
of points and then calculates new points out of the best ones in the current population.

Exploring the full parameter space becomes very hard for high dimensional problems (curse of dimensionality) and thus global algorithms can become inefficient.

Using local algorithms from several starting points instead, does not even try to get a complete picture of the parameter space, as each local optimizer is greedy
and only explores promising directions. If the computational budget is limited, it can be the better approach. This is especially true
if the problem is smooth and fast gradient-based local algorithms can be used.


2. Derivative-based vs. derivative-free
=======================================

Derivative-based optimizers use the first and potentially second derivative of the criterion function to determine search directions and step sizes.
Derivative-free methods do not use this information.

+-------------------------------------------------+----------------------------------------------------+
|          Use derivative-based method            |           Use derivative-free method               |
+=================================================+====================================================+
| Your criterion function is differentiable and   | Your criterion function is non differentiable or   |
| you have closed form derivatives                | not even deterministic                             |
+-------------------------------------------------+----------------------------------------------------+
| You have nonlinear constraints [*]_             | You have a problem structure for which specialized |
|                                                 | derivative free methods exist (e.g. a nonlinear    |
|                                                 | least squares problem).                            |
+-------------------------------------------------+----------------------------------------------------+
| You have a large computer: Then, estimagic can  | You suspect that numerical derivatives will not be |
| calculate the numerical derivatives in parallel | very accurate (e.g. because your criterion         |
|                                                 | function is differentiable but very wiggly)        |
+-------------------------------------------------+----------------------------------------------------+

.. [*] There simply are no derivative free methods that can handle nonlinear constraints we are aware of. If you have one, let us know!

When your function is differentiable, but you can only calculate the derivates numerically, the best approach is unclear.


2. Which derivative-free optimizer
==================================




3. Which derivative-based optimizer
===================================
