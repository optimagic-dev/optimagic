.. _logging:

=========================================
The *logging* and *log_options* Arguments
=========================================

Estimagic can keep a persistent log of the parameter and criterion values tried out by
an optimizer. For this we use an sqlite database, which makes it easy to read from and
write to the log-file from several processes or threads. Moreover, it is possible to
retrieve data from the log-file without ever loading it into memory, which might be
relevant for very long running optimizations.

The log-file is updated instantly when new information becomes available. Thus, no data
is lost when an optimization has to be aborted or a server is shut down for maintenance.

The sqlite database is also used to exchange data between the optimization and the
dashboard.

In addition to parameters and criterion values, we also save all arguments to an
maximize or minimize in the database as well as other information in the database
that can help to reproduce an optimization result.

The logging Argument
====================


``logging`` can be a string or pathlib.Path that specifies the path to a sqlite3
database. Typically, those files have the file extension ``.db``. If the file does not
exist, it will be created for you.


The log_options Argument
========================

``log_options`` is a dictionary with keyword arguments that influence the logging
behavior. The following options are available:


- "suffix": A string that is appended to the default table names, separated
  by an underscore. You can use this if you want to write the log into an
  existing database where the default names "optimization_iterations",
  "optimization_status" and "optimization_problem" are already in use.
- "fast_logging": A boolean that determines if "unsafe" settings are used
  to speed up write processes to the database. This should only be used for
  very short running criterion functions where the main purpose of the log
  is a real-time dashboard and it would not be catastrophic to get a
  corrupted database in case of a sudden system shutdown. If one evaluation
  of the criterion function (and gradient if applicable) takes more than
  100 ms, the logging overhead is negligible.
- "if_exists": (str) One of "extend", "replace", "raise"
