==============
The Dashboard
==============

Overview
---------

Estimagic supports a dashboard that allows the user to inspect the optimization.
The dashboard is built and opened automatically
unless the dashboard argument is set to False in ref:`minimize`.


Tabs
----

The estimagic dashboard will provide four tabs:

1. **Convergence Monitor**:
    This tab plots the values of the fitness and
    (selected) parameter values in real time.

2. **Profiling Information**:
    This tab will provide a snakeviz profile of the criterion function.

3. **System Monitor**:
    This tab will show the resource usage.

4. **Optimization Log**:
    Output during the optimization will be logged here.


.. todo:: Implement the other tabs


Implementation
---------------

The dashboard is implemented using a Bokeh Server which is run
in a separate thread parallel to the optimization.

.. attention::
    The Bokeh server stops looking for updates
    when the optimization finishes. However, the server is
    only shut down when the Python script that
    called the optimization exits or the notebook that started
    the optimization is shut down or restarted.

``dashboard.py`` contains the main function ``run_dashboard``.
``run_dashboard`` builds the tabs and adds the callbacks for updating the plots.
It is the basis for the Bokeh Application that is run in the Bokeh Server.

The functions for building and updating each tab have their own module.

The functions for setting up and running the server are in ``server_functions.py``.
