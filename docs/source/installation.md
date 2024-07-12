# Installation

## Basic installation

The package can be installed via conda. To do so, type the following commands in a
terminal or shell:

```
conda config --add channels conda-forge
```

```
conda install estimagic
```

The first line adds conda-forge to your conda channels. This is necessary for conda to
find all dependencies of estimagic. The second line installs estimagic and its mandatory
dependencies.

## Installing optional dependencies

Only `scipy` is a mandatory dependency of estimagic. Other algorithms become available
if you install more packages. We make this optional because most of the time you will
use at least one additional package, but only very rarely will you need all of them.

For an overview of all optimizers and the packages you need to install to enable them,
see {ref}`list_of_algorithms`.

To enable all algorithms at once, do the following:

```
conda install nlopt
```

```
pip install Py-BOBYQA
```

```
pip install DFO-LS
```

```
conda install petsc4py
```

*Note*: `` `petsc4py` `` is not available on Windows.

```
conda install cyipopt
```

```
conda install pygmo
```

```
pip install fides>=0.7.4
```

*Note*: Make sure you have at least 0.7.1.
