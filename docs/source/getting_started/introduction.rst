========
Overview
========


Optimization
============

Estimagic wraps a large collection of local and global optimizers from
`Pygmo <https://esa.github.io/pagmo2/>`_,
`Scipy <https://docs.scipy.org/doc/scipy/reference/optimize.html>`_,
`Nlopt <https://nlopt.readthedocs.io/en/latest/>`_ and
`TAO <https://tinyurl.com/yyjaefmk>`_ with a common interface.

All optimizers support bounds and several types of constraints.

While estimagic is by far not the only library that offers a large set
of optimizers, it has one distinguishing feature: The parameter vector over
which you optimize is not stored in an array but in a pandas DataFrame.

This allows access to parameters based on labels as opposed to position,
store additional information like bounds conveniently and express constraints
with a very intuitive and concise syntax.

You can monitor your optimizations in a beautiful interactive dashboard that
shows the evolution of the criterion function and all parameters in real time.

Numerical Differentiation
=========================

Estimagic wraps numdifftools to provide functions to calculate very
precise gradients, jacobians and hessians of functions. The precision is
achieved by evaluating numerical derivatives at different step sizes and using
Richardson extrapolations. While this increases the computational cost, it
works for any function, whereas other approaches that would yield a similar
precision, like complex step derivatives have stronger requirements.


Inference
=========

Estimagic provides functions to calculate standard errors for Maximum
Likelihood, GMM, Method of Simulated Moments and Indirect inference estimators.

Visualization
=============

We implement functions for interactive plots that help you to compare large
numbers of estimated parameters. The plots help you to diagnose how strongly
your results depend on constraints, start values or the choice of optimizer.
Moreover, you can see whether the size of those effects is large or small in
comparison to your statistical standard errors.
