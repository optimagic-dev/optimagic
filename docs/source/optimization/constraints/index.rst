.. _constraints:

==========================
The *constraints* Argument
==========================


Basic Structure of Constraints
==============================

``minimize`` and ``maximize`` can take a list with any number of constraints.
A constraint in estimagic is a dictionary. The following keys are mandatory for all
types of constraints:

1. ``"loc"`` or ``"query"`` but not both: This will select the subset of parameters to
which the constraint applies. If you use "loc", the corresponding value can be any
expression that is valid for ``DataFrame.loc``. If you are not familiar with these
methods, check out our `tutorial on selecting parameters <selecting_parameters.ipynb>`_.

2. ``"type"``: This can take any of the following values:

- **"fixed"**: The selected parameters are fixed to a value.
- **"probability"**: The selected parameters sum to one and are between zero and one.
- **"increasing"**: The selected parameters are increasing.
- **"decreasing"**: The seletced parameters are decreasing.
- **"equality"**: The selected parameters are equal to each other.
- **"pairwise_equality"**: Several sets of parameters are pairwise equal to each other.
- **"covariance"**: The selected parameters are variances and covariances.
- **"sdcorr"**: The selected parameters are standard deviations and correlations.
- **"linear"**: The selected parameters satisfy a linear constraint with equality or
  inequalities.


Depending on the type of constraint, some additional entries in the constraint
dictionary might be required. The details are explained on the next pages:

Details and Tutorials
=====================

.. toctree::
    :maxdepth: 1

    selecting_parameters.ipynb
    constraints.ipynb
    implementation
