.. _algorithms:

Optimization algorithms in estimagic
====================================

The *algorithm* argument
------------------------

The ``algorithm`` argument can either be string with the name of a algorithm that is
implemented in estimagic, or a function that fulfills the
:ref:`internal_optimizer_interface`.


Which algorithms are available in estimagic depends on the packages a user has
installed:

.. _list_of_algorithms:

.. toctree::
    :maxdepth: 1

    The Scipy Algorithms Without Any Dependencies <scipy_algorithms>
    Tao Pounders With ``petsc4py`` installed <tao_algorithms>
    DF-OLS With ``dfols`` installed <nag_dfols>
    PyBOBYQA With ``pybobyqa`` installed <nag_pybobyqa>


.. With ``nlopt`` installed
.. With ``cyipopt`` installed
.. With ``pygmo`` installed


The *algo_options* argument
---------------------------

``algo_options`` is a dictionary with optional keyword arguments that are passed to the
optimizer.

We align the names of all ``algo_options`` across algorithms.

Since some optimizers support many tuning parameters we group some of them by the
first part of their name (e.g. all convergence criteria names start with
``convergence``).
Below, all option names contain only `_`.
However, to make the group membership more visible, you can also pass them with
dot notation.
For example, if you wanted to set some tuning parameters of `nag_dfols` you could
specify your algo options like this:

.. code-block:: python

    algo_options = {
        "convergence.scaled_criterion_tolerance": 0.0,
        "convergence.noise_corrected_criterion_tolerance": 1.1,
        "trustregion.shrinking_factor.not_successful": 0.4,
        "trustregion.shrinking_factor.lower_radius": 0.2,
        "trustregion.shrinking_factor.upper_radius": 0.8,
        "trustregion.threshold_successful": 0.2,
        "trustregion.threshold_very_successful": 0.9,
    }

Estimagic then automatically replaces the ``.`` with ``_`` before passing them to the
internal optimizer.

However, not all algorithms support all options. Which options are supported and
very specific details of
what the options mean are documented for each algorithm below.

To make it easier to switch between algorithms, we simply ignore non-supported options
and issue a warning that explains which options have been ignored.

To find more information on ``algo_options`` that more than one algorithm allows for
see :ref:`algo_options`.


How to read the algorithms documentation
----------------------------------------

Below we document the supported algorithms. The documentation refers to the internal
optimizer interface (see :ref:`internal_optimizer_interface`). However, those functions
are not intended to be called by the user. Instead users should call them by calling
``maximize`` or ``minimize`` with the corresponding algorithm argument.

The arguments ``criterion_and_derivative``, ``x``, ``lower_bound`` and ``upper_bound``
of the signatures below should simply be ignored.

The other arguments can be set as ``algo_options`` when calling ``maximize`` or
``minimize``.
