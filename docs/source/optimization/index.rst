============
Optimization
============


Optimization is usually the most difficult and time consuming part of a research
project in structural econometrics. Usually, difficulties arise from three sources:

1. Different problems require different optimizers and it is not known ex-ante which
optimizer will perform best.

2. Most optimizers are black boxes that only return when they are done. Meanwhile,
they don't give you any information on the progress.

3. The way constraints are specified depends strongly on the algorithm. This makes it
cumbersome to experiment with several optimizers.

Estimagic reduces or erases these difficulties:

1. Estimagic makes it very easy to try out a large number of optimizers, either
sequentially or in parallel by just switching a few arguments of the
``maximize`` or ``minimize`` functions.

2. Estimagic stores all parameters and function values in a database.
The estimagic dashboard visualizes this database. With the dashboard you can monitor the
optimization while it is running or explore your optimizer's behavior after the
algorithm terminated.

3. Estimagic transforms your constrained problem by applying bijective and
differentiable reparametrizations. These transformations make sure that the optimizer
can only try parameter vectors that satisfy all constraints and can sometimes reduce
the dimensionality of the optimization problem substantially.

In the following we first explain the basic interface of estimagic's
optimization functions. Then, we explain the arguments in detail. This section
is also relevant for you if you call the ``minimize`` or ``maximize`` functions
indirectly, e.g. via the a ``fit`` method of a package that uses estimagic.
While those packages might implement different defaults that are tailored to the
problem at hand, they will usually let you pass your own arguments through to
estimagic if you desire to do so.

.. toctree::
   :maxdepth: 1

   interface
   params
   constraints/index
   dashboard
   logging
   algorithms
   general_options
