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
   This tab plots the values of the fitness and (selected) parameter values in real
   time.

2. **Profiling Information**:
   This tab will provide a snakeviz profile of the criterion function.

3. **Optimization Log**:
   Output during the optimization will be logged here.


.. todo:: Implement tab 2 and 3


Options
-------

All functions with the dashboard argument, also support an optional argument db_options.
It is a dictionary that allows to configure the dashboard. The following entries are
supported:

- ``rollover (int)`` : If negative, the complete history of criterion function
  evaluations and parameter values from the optimization is stored and displayed in the
  convergence plots. Otherwise, only the last ``rollover`` values are kept. This is
  recommended for very long running optimizations.
- ``port (int)``: Defaults to a random port over which the notebook can be accessed.
- ``evaluations_to_skip (int)``: Only store and display data from a subset of the
  criterion function evaluations. This is recommended for very long running
  optimizations and for optimizers that take numerical derivatives.
- ``time_between_updates (float)``: Defaults to 0. Seconds between each update of the
  convergence plot. A good value is ``evaluations_to_skip`` multiplied with the
  approximate runtime of criterion function.
- ``open_browser (bool)``: Defaults to ``True``. On a remote server the dashboard should
  not launch a browser. See :ref:`remote-server` for more information.


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


.. _remote-server:

On a remote server
------------------

``estimagic`` is well suited to run on both, on local machines and on remote servers
with much more computational support. Normally, these servers do not offer a GUI and a
browser. Instead, the user must tunnel into the server via ``ssh`` and redirect the
notebook and the dashboard to the local machine. Here is a short instruction for how to
use ``estimagic`` on a remote server.

1. Open Bash, Powershell, CMD or Terminal.

2. We redirect the Jupyter Lab and ``estimagic``'s dashboard to our local machine
   so that we can access them as usual via ports, e.g., 10101 and 10102, respectively.

   .. code-block:: bash

       ssh -N -f -L localhost:10101:localhost:10101 username@server-address
       ssh -N -f -L localhost:10102:localhost:10102 username@server-address

   ``-N`` prevents to commands on the remote, ``-f`` hides the connection in the
   background, so the console windows is not blocked, ``-L`` is used to bind your local
   port to a remote port. At last, type your server user name and the server address
   separated with an ``@``. You are asked to enter your password to establish the
   connection.

3. Now, log into the remote server with

   .. code-block:: bash

       ssh username@server-address

   and enter your password.

4. One the remote, launch the Jupyter Lab with

   .. code-block:: bash

       jupyter lab --no-browser --port=10101

   Use a leading ``&`` in a Bash or Powershell v6 Terminal to hide the task in the
   background. If your terminal is blocked, open another one.

5. On your local machine, open ``localhost:10101`` and you should see the Jupyter Lab.

6. Use a notebook to run a maximization or minimization with ``estimagic``. Make sure to
   add among other options the following two to the ``db_options``.

   .. code-block:: python

       from estimagic.optimization.optimize import maximize, minimize


       maximize(..., db_options={"port": 10102, "open_browser": False}, ...)
       minimize(..., db_options={"port": 10102, "open_browser": False}, ...)

   ``"open_browser"`` is on by default, but it has to be disabled as the dashboard
   crashes if it does not find a browser.

7. That's it. For more information on ``ssh`` and how to conveniently configure it,
   check out `Working remotely in shell environments
   <https://github.com/OpenSourceEconomics/hackathon/blob/master/
   material/2019_08_20/17_shell_remote.pdf>`_.
