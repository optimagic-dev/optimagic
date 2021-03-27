
.. _examples_of_simple_optimizers:

====================
Visualize simple optimizers
====================

At this page visualization of different types of optimizers are presented.

Derivative based algorithms 
===========================
.. _type_of_algorithms_for_based:

1. Line search
---------------------

Basic Idea:
    1. Use first derivative to get search direction
    2. Use approximated second derivative to guess step length
    3. Use a line search algorithm to see how far to go in the search direction

Potential Problems: 
    - Line search stays a 1d problem even with many parameters
    - Only solved approximately
    - Quite complicated if you really want to understand it
    - Most of the time accepts the first guess   

.. _example_algorithms_for_based_direct:

    .. raw:: html

        <div class="container">
        <div id="accordion" class="shadow tutorial-accordion">

            <div class="card tutorial-card">
                <div class="card-header collapsed card-link" data-toggle="collapse" data-target="#collapseOne">
                    <div class="d-flex flex-row tutorial-card-header-1">
                        <div class="d-flex flex-row tutorial-card-header-2">
                            <button class="btn btn-dark btn-sm"></button>
                            Stylized optimizer -- Derivate based line search 
                        </div>
                        <span class="badge gs-badge-link">

    .. raw:: html

                        </span>
                    </div>
                </div>
                <div id="collapseOne" class="collapse" data-parent="#accordion">
                    <div class="card-body">

    .. image:: ../../_static/images/final_result/derivate_based_line_search_algorithm.gif



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
                                Real optimizer -- L_BFGS_B
                            </div>
                            <span class="badge gs-badge-link">

    .. raw:: html

                            </span>
                        </div>
                    </div>
                    <div id="collapseTwo" class="collapse" data-parent="#accordion">
                        <div class="card-body">

    .. image:: ../../_static/images/final_result/L_BFGS_B.gif
    


2. Trust region
---------------------


Basic Idea:
    1. Fix a trust region radius
    2. Construct a Taylor expansion of the function based on function value, gradient, and (approximation to) Hessian 
    3. Minimize the Taylor expansion within the trust region
    4. Evaluate function again at the argmin of the Taylor expansion
    5. Compare expected and actual improvement
    6. Accept the new parameters if actual improvement is good enough
    7. Potentially modify the trust region radius (This is a very important and very complicated step)
    8.	Go back to 2.

Potential Problems: 
    - Most of the time, the approximation was not very good but sent us in the right direction
    - After a successful iteration, the trust region radius is increased
    - At some point it becomes too large and needs to be decreased
    - From now on the algorithm would converge soon because of a zero gradient
    - Even when it converges, the trust region radius does not shrink to zero       

.. _example_algorithms_for_based_trust:


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
                            Stylized optimizer -- Derivative based trust region 
                            </div>
                            <span class="badge gs-badge-link">

    .. raw:: html

                            </span>
                        </div>
                    </div>
                    <div id="collapseThree" class="collapse" data-parent="#accordion">
                        <div class="card-body">

    .. image:: ../../_static/images/final_result/derivative_based_trust_region_algorithm.gif



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
                            Real optimizer -- Trust_NCG
                            </div>
                            <span class="badge gs-badge-link">

    .. raw:: html

                            </span>
                        </div>
                    </div>
                    <div id="collapseFour" class="collapse" data-parent="#accordion">
                        <div class="card-body">

    .. image:: ../../_static/images/final_result/Trust_NCG.gif


Derivative free algorithms 
==========================
.. _type_of_algorithms_for_free:

1. Direct search
---------------------


Basic Idea:

    1. Explore parameter space around current point systematically and accept the best value
    2. Also called pattern search because the points at which the function is evaluated form a pattern
    3. Easiest example for one dimensional problems:

        - Evaluate function at current point and one other point
        - Switch direction of other point if you got a decrease
        - Make steps larger after success
        - Make steps smaller after failure


.. _example_algorithms_for_free_direct:



    .. raw:: html

        <div class="container">
        <div id="accordion" class="shadow tutorial-accordion">

            <div class="card tutorial-card">
                <div class="card-header collapsed card-link" data-toggle="collapse" data-target="#collapseFive">
                    <div class="d-flex flex-row tutorial-card-header-1">
                        <div class="d-flex flex-row tutorial-card-header-2">
                            <button class="btn btn-dark btn-sm"></button>
                            Stylized optimizer -- Derivate free direct search 
                        </div>
                        <span class="badge gs-badge-link">

    .. raw:: html

                        </span>
                    </div>
                </div>
                <div id="collapseFive" class="collapse" data-parent="#accordion">
                    <div class="card-body">

    .. image:: ../../_static/images/final_result/derivative_free_direct_search_algorithm.gif



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
                            Real optimizer -- Nelder_Mead
                            </div>
                            <span class="badge gs-badge-link">

    .. raw:: html

                            </span>
                        </div>
                    </div>
                    <div id="collapseSix" class="collapse" data-parent="#accordion">
                        <div class="card-body">

    .. image:: ../../_static/images/final_result/Nelder_Mead.gif





2. Trust region
---------------------

Basic Idea:

    1. Similar to derivative based trust region algorithm
    2. Instead of Taylor expansion, use a surrogate model based on interpolation or regression:

            - Interpolation: Function is evaluated at exactly as many points as you need to fit the model
            - Regression: Function is evaluated at more points than you strictly need. Better for noisy functions.
            - In general: Evaluation points are spread further out than for numerical derivatives.
    3. How the evaluation points are determined is complicated. It is also crucial for the efficiency of the algorithm.


.. _example_algorithms_for_free_trust:



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
                            Stylized optimizer -- Derivate free trust region 
                            </div>
                            <span class="badge gs-badge-link">

    .. raw:: html

                            </span>
                        </div>
                    </div>
                    <div id="collapseSeven" class="collapse" data-parent="#accordion">
                        <div class="card-body">

    .. image:: ../../_static/images/final_result/derivate_free_trust_region_algorithm.gif






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
                            Real optimizer -- Cobyla
                            </div>
                            <span class="badge gs-badge-link">

    .. raw:: html

                            </span>
                        </div>
                    </div>
                    <div id="collapseEight" class="collapse" data-parent="#accordion">
                        <div class="card-body">
    .. image:: ../../_static/images/final_result/Cobyla.gif

