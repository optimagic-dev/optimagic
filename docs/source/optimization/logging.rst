.. _logging:

=========================================
The *logging* and *log_options* Arguments
=========================================

Estimagic can keep a persistent log of the parameter and criterion values tried out by
an optimizer. For this we use an sqlite database, which makes it easy to read from and
write to the log-file from several processes or threads. Moreover, it is possible to
retrieve data from the log-file without ever loading it into memory, which might be
relevant for very long running.

The log-file is updated instantly when new information becomes available. Thus no data
is lost when an optimization has to be aborted or a server is shut down for maintenance.

The sqlite database is also used to exchange data between the optimization and the
dashboard. Thus whenever the dashboard is used, a log file is created, even if no
logging is specified by the user.

In addition to parameters and criterion values, we also save all arguments to an
maximize or minimize in the database as well as other information that can help to
reproduce an optimization result in the database.

The logging Argument
====================


``logging`` can be a string or pathlib.Path that specifies the path to a sqlite3
database. Typically those files have the file extension ``.db``. If the file does not
exist, it will be created for you. If it exists, we will create and potentially
overwrite tables that are used to log the optimization. The details of what estimagic
will do with your database file are documented in the following function.


.. autofunction:: estimagic.logging.create_database.prepare_database


The log_options Argument
========================

``log_options`` is a dictionary with keyword arguments that influence the logging
behavior. The following options are available:

- ``"readme"``: A string with a description of the optimization. This can be helpful to
  send a message to your future self who might have forgotten why he ran this particular
  optimization.
