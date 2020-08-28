.. _params:

===============================
How to Specify Start Parameters
===============================

The ``params`` DataFrame is an important concept in estimagic. It collects
information on the dimensionality of the optimization problem, lower and upper
bounds, categories of parameters and valid ranges for randomly sampled parameter
vectors. Moreover, it is the mandatory first argument of any criterion function
optimized with estimagic.


If you haven't done so yet, you should check out our `Ordered Logit Example`_,
so you see one small params DataFrame in action.

.. _Ordered Logit Example: ../getting_started/ordered_logit_example.ipynb


The Index
=========

If you are not familiar with a pandas ``MultiIndex`` we strongly suggest
to read up on this and get some practice before you continue. Good resourses are
the `documentation <https://tinyurl.com/yxhr362e>`_ or Wes McKinney's
`book <https://tinyurl.com/cfvqsy5>`_.

The choice of a good index is very important to reap all benefits estimagic
offers. If you choose a good one, you can easily select parameters you need
to select and express constraints on the parameters in just one line.

Since this is a very project specific choice, estimagic makes absolutely no
assumptions on your index, so you are completely free to choose whatever you
want. Below we have a few tips to help you in that choice:

1. **Choose as many levels as you need to select your parameters in all
partitions you ever need.** In the ordered logit example this was achieved by
two levels, where the first distinguished cutoffs vs utility parameters and the
second was the parameter name. In dynamic models with time varying parameters,
you often need another level for the period. But, of course, your index should
also be as parsimonious as possible. In practice, we always use between 2 and
4 levels.

2. To decide what your levels should be, it is often helpful to make a list of the
quantities into which you have to parse your parameters. Then make a list of all
constraints you want to express. Build an index that makes those two steps easiest.

The ``"value"`` column
======================


The ``"value"`` column is the only mandatory column in ``params``. It contains
what most other optimization libraries call ``x``, i.e. the start parameters
for the optimization.

The result of the optimization will contain a copy of ``params`` where the
original ``"value"`` column has been replaced by the optimal parameters.

The "lower_bound" and "upper_bound" columns
===============================

``"lower_bound"`` and ``"upper_bound"`` are optional columns with box constraints on the
parameters. You can also provide just one of them. For parameters that don't
have a bound, you can fill them with ``-np.inf`` and ``np.inf`` respectively.

Note that all optimizers in estimagic can deal with box constraints. However,
not all more complicated constraints (e.g. "covariance" constraints) are
compatible with box constraints. If you select an invalid combination of box constraints
and other constraints you will get an error.


The "draw_lower" and "draw_upper" columns
=========================================

``"draw_lower"`` and ``"draw_upper"`` are optional columns that are only used
if random start values are drawn, for example in genetic algorithms or when
starting a local optimization from several start values. We distinguish this
from the box constraints because you might want to leave some parameters
unconstrained but still generate random start values.



The "group" column
==================

``"group"`` is an optional column of strings that is only used for visualization
purposes. It can be used to partition the parameter into groups that have
similar magnitudes and/or are otherwise related. Those parameters will then
be grouped in the same sub-plot in the dashboard or the convergence plot.

Parameters whose ``"group"`` is ``None`` are typically not plotted. This can
be used to save resources when using the dashboard on vary large optimizations.


Invalid columns
===============

Some names are reserved for internal use in estimagic. Currently those are:

``'_fixed_value'``, ``'_is_fixed_to_value'``, ``'_is_fixed_to_other'``,
``'_pre_replacements'``, ``'_post_replacements'`` as well any name that starts with
``_internal``.
