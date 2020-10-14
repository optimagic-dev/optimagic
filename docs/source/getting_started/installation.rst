============
Installation
============


Basic Installation
==================

The package can be installed via conda. To do so, type the following commands in
a terminal or shell:

``$ conda config --add channels conda-forge``
``$ conda install -c opensourceeconomics estimagic``

The first line adds conda-forge to your conda channels. This is necessary for
conda to find all dependencies of estimagic. The second line installs estimagic
and its mandatory dependencies.


Installing Optional Dependencies
================================

Only the scipy optimizers are a mandatory dependency of estimagic. Other algorithms
become available if you install more packages. The reason why we make this optional
is not that we think scipy is enough! Most of the time you will use at least one
additional package, but only very rarely you will need all of them.


For an overview of all optimizers and the packages you need to install to enable them
see :ref:`list_of_algorithms`.


To enable all algorithms at once, do the following:

``conda install nlopt``

``pip install Py-BOBYQA``

``pip install DFOLS``

``conda install petsc4py`` (Not available on windows)

``conda install cyipopt`` (Not available on windows)

``conda install pygmo``
