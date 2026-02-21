<a href="https://optimagic.readthedocs.io">
    <p align="center">
        <img src="https://raw.githubusercontent.com/optimagic-dev/optimagic/main/docs/source/_static/images/optimagic_logo.svg" width=50% alt="optimagic">
    </p>
</a>

______________________________________________________________________

[![PyPI](https://img.shields.io/pypi/v/optimagic?color=blue)](https://pypi.org/project/optimagic)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/optimagic)](https://pypi.org/project/optimagic)
[![image](https://img.shields.io/conda/vn/conda-forge/optimagic.svg)](https://anaconda.org/conda-forge/optimagic)
[![image](https://img.shields.io/conda/pn/conda-forge/optimagic.svg)](https://anaconda.org/conda-forge/optimagic)
[![PyPI - License](https://img.shields.io/pypi/l/optimagic)](https://pypi.org/project/optimagic)
[![image](https://readthedocs.org/projects/optimagic/badge/?version=latest)](https://optimagic.readthedocs.io/en/latest)
[![image](https://img.shields.io/github/actions/workflow/status/optimagic-dev/optimagic/main.yml?branch=main)](https://github.com/optimagic-dev/optimagic/actions?query=branch%3Amain)
[![image](https://codecov.io/gh/optimagic-dev/optimagic/branch/main/graph/badge.svg)](https://codecov.io/gh/optimagic-dev/optimagic)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/optimagic-dev/optimagic/main.svg)](https://results.pre-commit.ci/latest/github/optimagic-dev/optimagic/main)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Downloads](https://pepy.tech/badge/optimagic/month)](https://pepy.tech/project/optimagic)
[![NumFOCUS](https://img.shields.io/badge/NumFOCUS-affiliated%20project-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)](https://numfocus.org/sponsored-projects/affiliated-projects)

optimagic is a Python package for numerical optimization. It is a unified interface to
optimizers from SciPy, NlOpt, and many other Python packages. Its features include:

- **SciPy-compatible API.** optimagic's `minimize` function works just like SciPy's, so
  you don't have to adjust your code. You simply get more optimizers for free.
- **Powerful diagnostic tools.** Visualize optimizer histories, compare runs, and
  diagnose convergence problems.
- **Parallel numerical derivatives.** Compute gradients, jacobians, and hessians with
  parallel execution.
- **Bounded, constrained, and unconstrained optimization.** Support for bounds, linear
  constraints, nonlinear constraints, fixed parameters, and more.
- **Statistical inference on estimated parameters.** The estimagic subpackage provides
  functionality for confidence intervals, standard errors, and p-values.

# Installation

optimagic is available on [PyPI](https://pypi.org/project/optimagic) and on
[conda-forge](https://anaconda.org/conda-forge/optimagic). Install the package with

```console
$ pip install optimagic
```

or

```console
$ conda install -c conda-forge optimagic
```

optimagic ships with all `scipy` optimizers out of the box. Additional algorithms become
available if you install optional packages. For an overview of all supported optimizers
and how to enable them, see the
[list of algorithms](https://optimagic.readthedocs.io/en/latest/algorithms.html).

# Usage

```python
import optimagic as om
import numpy as np


def fun(x):
    return x @ x


result = om.minimize(fun, params=np.array([1, 2, 3]), algorithm="scipy_lbfgsb")
result.params.round(9)  # np.array([0., 0., 0.])
```

# Documentation

You find the documentation at <https://optimagic.readthedocs.io> with
[tutorials](https://optimagic.readthedocs.io/en/latest/tutorials/index.html) and
[how-to guides](https://optimagic.readthedocs.io/en/latest/how_to/index.html).

# Changes

Consult the
[release notes](https://optimagic.readthedocs.io/en/latest/development/changes.html) to
find out about what is new.

# License

optimagic is distributed under the terms of the [MIT license](LICENSE).

# Citation

If you use optimagic for your research, please cite it with the following key to help
others discover the tool.

```bibtex
@Unpublished{Gabler2024,
    Title  = {optimagic: A library for nonlinear optimization},
    Author = {Janos Gabler},
    Year   = {2022},
    Url    = {https://github.com/optimagic-dev/optimagic}
}
```

# Acknowledgment

We thank all institutions that have funded or supported optimagic (formerly estimagic).

<table>
  <tc>
    <td><img src="docs/source/_static/images/numfocus_logo.png" width="200"></td>
    <td><img src="docs/source/_static/images/aai-institute-logo.svg" width="185"></td>
    <td><img src="docs/source/_static/images/tra_logo.png" width="240"></td>
    <td><img src="docs/source/_static/images/hoover_logo.png" width="192"></td>

</tc>
</table>
