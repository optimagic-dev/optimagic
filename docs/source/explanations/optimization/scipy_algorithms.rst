.. _scipy_algorithms:


The scipy optimizers
=====================

estimagic supports most ``scipy`` algorithms without requiring the user to install
additional dependencies.

The only ``scipy`` algorithms that are not supported are the following which
require the specification of the Hessian:

- dogleg
- trust-ncg
- trust-exact
- trust-krylov

The following arguments are not supported as part of ``algo_options``:

- ``disp``
    If set to True would print a convergence message.
    In estimagic it's always set to its default False.
    Refer to estimagic's result dictionary's "success" entry for the convergence
    message.
- ``return_all``
    If set to True, a list of the best solution at each iteration is returned.
    In estimagic it's always set to its default False.
    Use estimagic's database and dashboard instead to explore your criterion and
    algorithm.
- ``tol``
    This argument of minimize (not an options key) is passed as different types of
    tolerance (gradient, parameter or criterion, as well as relative or absolute)
    depending on the selected algorithm. We require the user to explicitely input
    the tolerance criteria or use our defaults instead.
- ``args``
    This argument of minimize (not an options key) is partialed into the function
    for the user. Specify ``criterion_kwargs`` in ``maximize`` or ``minimize`` to
    achieve the same behavior.
- ``callback``
    This argument would be called after each iteration and the algorithm would
    terminate if it returned True.

.. note::
    `scipy`'s COBYLA, SLSQP and trust-constr support general non linear constraints
    in principle. However, for the moment they are not supported.


.. raw:: html

    <div class="container">
    <div id="accordion" class="shadow tutorial-accordion">

        <div class="card tutorial-card">
            <div class="card-header collapsed card-link" data-toggle="collapse" data-target="#collapseOne">
                <div class="d-flex flex-row tutorial-card-header-1">
                    <div class="d-flex flex-row tutorial-card-header-2">
                        <button class="btn btn-dark btn-sm"></button>
                        L-BFGS-B
                    </div>
                    <span class="badge gs-badge-link">

.. raw:: html

                    </span>
                </div>
            </div>
            <div id="collapseOne" class="collapse" data-parent="#accordion">
                <div class="card-body">



.. autofunction:: estimagic.optimization.scipy_optimizers.scipy_lbfgsb


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
                            SLSQP
                        </div>
                        <span class="badge gs-badge-link">

.. raw:: html

                        </span>
                    </div>
                </div>
                <div id="collapseTwo" class="collapse" data-parent="#accordion">
                    <div class="card-body">


.. autofunction:: estimagic.optimization.scipy_optimizers.scipy_slsqp


.. raw:: html

                        </span>
                    </div>
                </div>
            </div>

            <div class="card tutorial-card">
                <div class="card-header collapsed card-link" data-toggle="collapse" data-target="#collapseThree">
                    <div class="d-flex flex-row tutorial-card-header-1">
                        <div class="d-flex flex-row tutorial-card-header-2">
                            <button class="btn btn-dark btn-sm"></button>
                            Nelder-Mead
                        </div>
                        <span class="badge gs-badge-link">

.. raw:: html

                        </span>
                    </div>
                </div>
                <div id="collapseThree" class="collapse" data-parent="#accordion">
                    <div class="card-body">

.. autofunction:: estimagic.optimization.scipy_optimizers.scipy_neldermead


.. raw:: html

                        </span>
                    </div>
                </div>
            </div>

            <div class="card tutorial-card">
                <div class="card-header collapsed card-link" data-toggle="collapse" data-target="#collapseFour">
                    <div class="d-flex flex-row tutorial-card-header-1">
                        <div class="d-flex flex-row tutorial-card-header-2">
                            <button class="btn btn-dark btn-sm"></button>
                            Modified Powell Method
                        </div>
                        <span class="badge gs-badge-link">

.. raw:: html

                        </span>
                    </div>
                </div>
                <div id="collapseFour" class="collapse" data-parent="#accordion">
                    <div class="card-body">

.. autofunction:: estimagic.optimization.scipy_optimizers.scipy_powell


.. raw:: html

                        </span>
                    </div>
                </div>
            </div>

            <div class="card tutorial-card">
                <div class="card-header collapsed card-link" data-toggle="collapse" data-target="#collapseFive">
                    <div class="d-flex flex-row tutorial-card-header-1">
                        <div class="d-flex flex-row tutorial-card-header-2">
                            <button class="btn btn-dark btn-sm"></button>
                            BFGS
                        </div>
                        <span class="badge gs-badge-link">

.. raw:: html

                        </span>
                    </div>
                </div>
                <div id="collapseFive" class="collapse" data-parent="#accordion">
                    <div class="card-body">

.. autofunction:: estimagic.optimization.scipy_optimizers.scipy_bfgs


.. raw:: html

                        </span>
                    </div>
                </div>
            </div>

            <div class="card tutorial-card">
                <div class="card-header collapsed card-link" data-toggle="collapse" data-target="#collapseSix">
                    <div class="d-flex flex-row tutorial-card-header-1">
                        <div class="d-flex flex-row tutorial-card-header-2">
                            <button class="btn btn-dark btn-sm"></button>
                            Conjugate Gradient
                        </div>
                        <span class="badge gs-badge-link">

.. raw:: html

                        </span>
                    </div>
                </div>
                <div id="collapseSix" class="collapse" data-parent="#accordion">
                    <div class="card-body">

.. autofunction:: estimagic.optimization.scipy_optimizers.scipy_conjugate_gradient


.. raw:: html

                        </span>
                    </div>
                </div>
            </div>

            <div class="card tutorial-card">
                <div class="card-header collapsed card-link" data-toggle="collapse" data-target="#collapseSeven">
                    <div class="d-flex flex-row tutorial-card-header-1">
                        <div class="d-flex flex-row tutorial-card-header-2">
                            <button class="btn btn-dark btn-sm"></button>
                            Newton's Conjugate Gradient
                        </div>
                        <span class="badge gs-badge-link">

.. raw:: html

                        </span>
                    </div>
                </div>
                <div id="collapseSeven" class="collapse" data-parent="#accordion">
                    <div class="card-body">

.. autofunction:: estimagic.optimization.scipy_optimizers.scipy_newton_cg



.. raw:: html

                        </span>
                    </div>
                </div>
            </div>

            <div class="card tutorial-card">
                <div class="card-header collapsed card-link" data-toggle="collapse" data-target="#collapseEight">
                    <div class="d-flex flex-row tutorial-card-header-1">
                        <div class="d-flex flex-row tutorial-card-header-2">
                            <button class="btn btn-dark btn-sm"></button>
                            COBYLA
                        </div>
                        <span class="badge gs-badge-link">

.. raw:: html

                        </span>
                    </div>
                </div>
                <div id="collapseEight" class="collapse" data-parent="#accordion">
                    <div class="card-body">

.. autofunction:: estimagic.optimization.scipy_optimizers.scipy_cobyla


.. raw:: html

                        </span>
                    </div>
                </div>
            </div>

            <div class="card tutorial-card">
                <div class="card-header collapsed card-link" data-toggle="collapse" data-target="#collapseNine">
                    <div class="d-flex flex-row tutorial-card-header-1">
                        <div class="d-flex flex-row tutorial-card-header-2">
                            <button class="btn btn-dark btn-sm"></button>
                            Truncated Newton
                        </div>
                        <span class="badge gs-badge-link">

.. raw:: html

                        </span>
                    </div>
                </div>
                <div id="collapseNine" class="collapse" data-parent="#accordion">
                    <div class="card-body">

.. autofunction:: estimagic.optimization.scipy_optimizers.scipy_truncated_newton


.. raw:: html

                        </span>
                    </div>
                </div>
            </div>

            <div class="card tutorial-card">
                <div class="card-header collapsed card-link" data-toggle="collapse" data-target="#collapseTen">
                    <div class="d-flex flex-row tutorial-card-header-1">
                        <div class="d-flex flex-row tutorial-card-header-2">
                            <button class="btn btn-dark btn-sm"></button>
                            Trust Region for Constrained Problems
                        </div>
                        <span class="badge gs-badge-link">

.. raw:: html

                        </span>
                    </div>
                </div>
                <div id="collapseTen" class="collapse" data-parent="#accordion">
                    <div class="card-body">

.. autofunction:: estimagic.optimization.scipy_optimizers.scipy_trust_constr



.. raw:: html

                    </span>
                </div>
            </div>
        </div>
    </div>


**References**

.. bibliography:: ../../refs.bib
    :labelprefix: scipy
    :filter: docname in docnames
    :style: unsrt
