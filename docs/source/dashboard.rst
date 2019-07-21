==============
The Dashboard
==============

Overview
---------

Estimagic provides a dashboard that allows to inspect an optimization.
All estimagic functions that run an optimization have an optional
boolean argument `dashboard`. When True, the function call will fire
up a new tab in your default browser that displays the dashboard.

Tabs
----

The estimagic dashboard will provide three tabs:

1. **Convergence Monitor**:
    This tab plots the values of the fitness and
    (selected) parameter values in real time.

2. **Profiling Information**:
    This tab will provide a snakeviz profile of the criterion function.

3. **Optimization Log**:
    Output during the optimization will be logged here.


.. todo:: Implement tab 2 and 3


Options
-------

All functions with the dashboard argument, also support an optional argument
db_options. It is a dictionary that allows to configure the dashboard. The
following entries are supported:

- `rollover (int)` : If <= 0, the complete history of criterion function
    evaluations and parameter values from the optimization is stored and
    displayed in the convergence plots. Otherwise, only the last `rollover`
    values are kept. This is recommended for very long running optimizations.
- `port (int)`: The port of the bokeh server
- `evaluations_to_skip (int)`: Only store and display data from a subset
    of the criterion function evaluations. This is recommended for very
    long running optimizations and for optimizers that take numerical
    derivatives.
- `time_between_updates (float)`: Seconds between each update
    of the convergence plot. A good value is
    `evaluations_to_skip` * approximate runtime of criterion function.
    Default is




Implementation
---------------

The dashboard is implemented using a Bokeh Server which is run
in a separate process parallel to the optimization. After the optimization
terminates, the updates to the dashboard are stopped, but the bokeh server
will keep running such that interactive features of the plots can still be
used.


``dashboard.py`` contains the main function ``run_dashboard``.
``run_dashboard`` builds the tabs and adds the callbacks for updating the plots.
It is the basis for the Bokeh Application that is run in the Bokeh Server.

The functions for building and updating each tab have their own module.

The functions for setting up and running the server are in ``server_functions.py``.
