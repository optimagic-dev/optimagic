.. _nag_algorithms:

Algorithms of the Numerical Algorithms Group
=============================================

Currently, estimagic supports the
`Derivative-Free Optimizer for Least-Squares Minimization (DF-OLS)
<https://numericalalgorithmsgroup.github.io/dfols/>`_ and
`BOBYQA <https://numericalalgorithmsgroup.github.io/pybobyqa/>`_
by the `Numerical Algorithms Group <https://www.nag.com/>`_.

To use DF-OLS you need to have
`the dfols package <https://tinyurl.com/y5ztv4yc>`_ installed.
BOBYQA requires `the pybobyqa package <https://tinyurl.com/y67foub7>`_ .

Their following arguments are not supported as ``algo_options``:

- ``scaling_within_bounds``
- ``init.run_in_parallel``
- ``do_logging``, ``print_progress`` and all their advanced options.
  Use estimagic's database and dashboard instead to explore your criterion
  and algorithm.

.. raw:: html

    <div class="container">
    <div id="accordion" class="shadow tutorial-accordion">

        <div class="card tutorial-card">
            <div class="card-header collapsed card-link" data-toggle="collapse" data-target="#collapseOne">
                <div class="d-flex flex-row tutorial-card-header-1">
                    <div class="d-flex flex-row tutorial-card-header-2">
                        <button class="btn btn-dark btn-sm"></button>
                        DF-OLS
                    </div>
                    <span class="badge gs-badge-link">

.. raw:: html

                    </span>
                </div>
            </div>
            <div id="collapseOne" class="collapse" data-parent="#accordion">
                <div class="card-body">

.. autofunction:: estimagic.optimization.nag_optimizers.nag_dfols
.. raw:: html

                        </span>
                    </div>
                </div>
            </div>

            <div class="card tutorial-card">
                <div class="card-header collapsed card-link" data-toggle="collapse" data-target="#collapseTwo">
                    <div class="d-flex flex-row tutorial-card-header-1">
                        <div class="d-flex flex-row tutorial-card-header-2">
                            <button class="btn btn-dark btn-sm"></button>
                            BOBYQA
                        </div>
                        <span class="badge gs-badge-link">

.. raw:: html

                        </span>
                    </div>
                </div>
                <div id="collapseTwo" class="collapse" data-parent="#accordion">
                    <div class="card-body">




.. autofunction:: estimagic.optimization.nag_optimizers.nag_pybobyqa

.. raw:: html

                    </span>
                </div>
            </div>
        </div>
    </div>



**References**

.. bibliography:: ../../refs.bib
    :labelprefix: nag
    :filter: docname in docnames
    :style: unsrt
