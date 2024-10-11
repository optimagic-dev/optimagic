# optimagic

![PyPI - Version](https://img.shields.io/pypi/v/optimagic)
[![image](https://img.shields.io/pypi/pyversions/optimagic)](https://pypi.org/project/optimagic)
[![image](https://img.shields.io/conda/vn/conda-forge/optimagic.svg)](https://anaconda.org/conda-forge/optimagic)
[![image](https://img.shields.io/conda/pn/conda-forge/optimagic.svg)](https://anaconda.org/conda-forge/optimagic)
[![image](https://img.shields.io/pypi/l/optimagic)](https://pypi.org/project/optimagic)
[![image](https://readthedocs.org/projects/optimagic/badge/?version=latest)](https://optimagic.readthedocs.io/en/latest)
[![image](https://img.shields.io/github/actions/workflow/status/optimagic-dev/optimagic/main.yml?branch=main)](https://github.com/optimagic-dev/optimagic/actions?query=branch%3Amain)
[![image](https://codecov.io/gh/optimagic-dev/optimagic/branch/main/graph/badge.svg)](https://codecov.io/gh/optimagic-dev/optimagic)
[![image](https://results.pre-commit.ci/badge/github/optimagic-dev/optimagic/main.svg)](https://results.pre-commit.ci/latest/github/optimagic-dev/optimagic/main)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![image](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![image](https://pepy.tech/badge/optimagic/month)](https://pepy.tech/project/optimagic)
[![image](https://img.shields.io/badge/NumFOCUS-affiliated%20project-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)](https://numfocus.org/sponsored-projects/affiliated-projects)
[![image](https://img.shields.io/twitter/follow/aiidateam.svg?style=social&label=Follow)](https://x.com/optimagic)

## Introduction

*optimagic* is a Python package for numerical optimization. It is a unified interface to
optimizers from SciPy, NlOpt and many other Python packages.

*optimagic*'s `minimize` function works just like SciPy's, so you don't have to adjust
your code. You simply get more optimizers for free. On top you get powerful diagnostic
tools, parallel numerical derivatives and more.

*optimagic* was formerly called *estimagic*, because it also provides functionality to
perform statistical inference on estimated parameters. *estimagic* is now a subpackage
of *optimagic*.

## Documentation

The documentation is hosted at https://optimagic.readthedocs.io

## Installation

The package can be installed via pip or conda. To do so, type the following commands in
a terminal:

```bash
pip install optimagic
```

or

```bash
$ conda config --add channels conda-forge
$ conda install optimagic
```

The first line adds conda-forge to your conda channels. This is necessary for conda to
find all dependencies of optimagic. The second line installs optimagic and its
dependencies.

## Installing optional dependencies

Only `scipy` is a mandatory dependency of optimagic. Other algorithms become available
if you install more packages. We make this optional because most of the time you will
use at least one additional package, but only very rarely will you need all of them.

For an overview of all optimizers and the packages you need to install to enable them
see {ref}`list_of_algorithms`.

To enable all algorithms at once, do the following:

`conda install nlopt`

`pip install Py-BOBYQA`

`pip install DFO-LS`

`conda install petsc4py` (Not available on Windows)

`conda install cyipopt`

`conda install pygmo`

`pip install fides>=0.7.4 (Make sure you have at least 0.7.1)`

## Citation

If you use optimagic for your research, please do not forget to cite it.

```
@Unpublished{Gabler2024,
  Title  = {optimagic: A library for nonlinear optimization},
  Author = {Janos Gabler},
  Year   = {2022},
  Url    = {https://github.com/optimagic-dev/optimagic}
}
```

## Acknowledgements

We thank all institutions that have funded or supported optimagic (formerly estimagic)

<img src="docs/source/_static/images/aai-institute-logo.svg" width="185">
<img src="docs/source/_static/images/numfocus_logo.png" width="200">
<img src="docs/source/_static/images/tra_logo.png" width="240">

<img src="docs/source/_static/images/hoover_logo.png" width="192">
<img src="docs/source/_static/images/transferlab-logo.svg" width="400">
