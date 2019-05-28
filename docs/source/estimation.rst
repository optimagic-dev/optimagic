==========
Estimation
==========


Overview
--------

Here we will have estimation functions for the following types of models:

- maximum likelihood (ml)
- generalized method of moments (gmm)
- method of simulated moments (msm)
- indirect inference (ii)

Most of these functions will just construct a criterion frunction from some
building blocks and then call our :ref:`minimize`.

Below we describe some concepts a user has to understand in order to use the
minimize function or any of the estimation functions.


.. _params_df:

The params DataFrame
--------------------

.. todo:: Add probit example

params_df is an important concept in estimagic. It collects information on the
dimensionality of the optimization problem, lower and upper bounds, fixed
parameters, categories of parameters and valid ranges for randomly sampled
parameter vectors. Moreover, it's 'value' column is the mandatory first
argument of any criterion function optimized with estimagic's :ref:`minimize`.

It can have the following columns (most of them being optional)

- ``'value'`` (dtype=float): Start values for the optimization. This column is
  not optional. These values are taken as start values for local optimizers
  and are in the initial population of genetic algorithms or pseudo global
  optimizers.
- ``'lower'`` (dtype=float): Lower bounds enforced in optimization. Can take the
  values - np.inf or np.nan for unbounded parameters.
- ``'upper'`` (dtype=float): Upper bounds enforced in optimization. Can take the
  values np.inf or np.nan for unbounded parameters.
- ``'draw_lower'`` and ``'draw_upper'`` (dtype=float): bounds used to randomly draw
  parameters. They are ignored or only enforced approximately for some
  constrained parameters. These columns are required for genetic or pseudo
  global optimizers.
- ``'group'`` (dtype=str or None): Indicates in which group (if any)
  a parameter's values will be plotted in the convergence tab of the dashboard.
  Parameters with value None are not plotted.

It is important to distinguish three related but different concepts:

- ``params_df``: the DataFrame described above, sometimes just called ``params``.
- ``params_sr``: the ``'value'`` column of params. This is the first argument of any
  criterion function optimized with estimagic's minimize function.
- ``internal_params``: a reparametrized version of params that is only used
  internally in order to enforce some types of constraints during the
  optimization. It is often shorter than params and has a different index.
  Moreover, the columns for lower ad upper bounds might be differnet.
  internal_params is only relevant if you want to read the source code or want
  to extend estimagic.

.. _constraints:

Specification of Constraints
----------------------------

.. todo:: Add probit example

The user can specify a list with any number of constraints. Each constraint is
a dictionary. The dictionary must contain the following entries:

- ``'loc'`` or ``'query'`` but not both. This specifies to which subset of the
  parameters the constraint applies. The value corresponding to 'loc' will be
  passed to df.loc and the value corresponding to 'query' will be passed to
  df.query so you can provide whatever is accepted by those methods.
- ``'type'``, which can take the following values:
    - ``'covariance'``: a set of parameters forms a valid (i.e. positive
      semi-definite) covariance matrix. This is not compatible with any other
      constraints on the involved parameters.
    - ``'sdcorr'``: the first part of a set of parameters are standard deviations,
      the second part are the lower triangle (excluding the diagonal)
      of a correlation matrix. All parameters together can be used to construct
      a full covariance matrix but are more interpretable. This is not compatible
      with any other type of constraints on the involved parameters.
    - ``'sum'``: a set of parameters sums to a specified value. The last involved
      parameter can't have bounds. In this case the constraint dictionary also
      needs to contain a 'value' key.
    - ``'probability'``: a set of parameters is between 0 and 1 and sums to 1.
    - ``'increasing'``: a set of parameters is increasing. We check that the box
      constraints are compatible with the order.
    - ``'equality'``: a set of parameters is restricted to be equal to a
      particular value. The value has to be specified in the constraints
      dictionary.
    - ``'pairwise_equality'``: Two sets of parameters are pairwise equal. In this
      the constraint dictionary has to contain the keys ``locs`` and ``queries``
      instead of ``loc`` and ``query``. Both are lists of arbitrary length
      and each element in the list hast to be a valid argument to
      ``DataFrame.loc[]`` or ``DataFrame.query()``, respectively. Pairwise
      equality constraints are just syntactic sugar and are converted
      to normal equality constraints internally.
    - ``'fixed'``: A set of parameters is fixed to some values. In this case
      the constraints dictionary has to contain a ``'value'`` entry which can
      be a scalar or an iterable of suitable length.


Lower and upper bounds are specified in :ref:`params_df`.

The constraints are enforced by reparametrizations, additional bounds or
additional fixed parameters. For details see :ref:`reparametrize`


.. todo:: Implement a way to use nlopts and pygmo's general equality or
  inequality constraints for all algorithms that support this type of
  constraints.

.. todo:: Find out if box constraints are implemented efficiently in pygmo


.. _list_of_algorithms:

List of algorithms
------------------

.. todo:: Document the algorithms and their arguments. Provide links to the pygmo documentation.


- pygmo_gaco
- pygmo_de
- pygmo_sade
- pygmo_de1220
- pygmo_ihs
- pygmo_pso
- pygmo_pso_gen
- pygmo_sea
- pygmo_sga
- pygmo_simulated_annealing
- pygmo_bee_colony
- pygmo_cmaes
- pygmo_xnes
- pygmo_nsga2
- pygmo_moead
- nlopt_cobyla
- nlopt_bobyqa
- nlopt_newuoa
- nlopt_newuoa_bound
- nlopt_praxis
- nlopt_neldermead
- nlopt_sbplx
- nlopt_mma
- nlopt_ccsaq
- nlopt_slsqp
- nlopt_lbfgs
- nlopt_tnewton_precond_restart
- nlopt_tnewton_precond
- nlopt_tnewton_restart
- nlopt_tnewton
- nlopt_var2
- nlopt_var1
- nlopt_auglag
- nlopt_auglag_eq
- scipy_L-BFGS-B
- scipy_TNC
- scipy_SLSQP








