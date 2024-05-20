<div style="padding-top: 50px;">
</div>

```{raw} html
<img src="_static/images/estimagic_logo.svg" class="only-light center" style="display:block; margin-left:auto; margin-right:auto; width:300px; height:auto;"/>

<img src="_static/images/estimagic_logo_dark_mode.svg" class="only-dark center" style="display:block; margin-left:auto; margin-right:auto; width:300px; height:auto;"/>
```

<br>
<br>

`estimagic` is a Python package for nonlinear optimization with or without constraints.
It is particularly suited to solve difficult nonlinear estimation problems. On top, it
provides functionality to perform statistical inference on estimated parameters.

For a complete introduction to optimization in estimagic, check out the
{ref}`estimagic_scipy2022`

If you want to learn more about estimagic, dive into one of the following topics

`````{grid} 1 2 2 2
---
gutter: 3
---
````{grid-item-card}
:text-align: center
:img-top: _static/images/light-bulb.svg
:class-img-top: index-card-image
:shadow: md

```{button-link} getting_started/index.html
---
click-parent:
ref-type: ref
class: stretched-link index-card-link sd-text-primary
---
Getting Started
```

New users of estimagic should read this first.

````

````{grid-item-card}
:text-align: center
:img-top: _static/images/book.svg
:class-img-top: index-card-image
:shadow: md

```{button-link} how_to_guides/index.html
---
click-parent:
ref-type: ref
class: stretched-link index-card-link sd-text-primary
---
How-to Guides
```

Detailed instructions for specific and advanced tasks.

````

````{grid-item-card}
:text-align: center
:img-top: _static/images/installation.svg
:class-img-top: index-card-image
:shadow: md

```{button-link} getting_started/installation.html
---
click-parent:
ref-type: ref
class: stretched-link index-card-link sd-text-primary
---
Installation
```

Installation instructions for estimagic and optional dependencies.

````

````{grid-item-card}
:text-align: center
:img-top: _static/images/optimization.svg
:class-img-top: index-card-image
:shadow: md

```{button-link} algorithms.html
---
click-parent:
ref-type: ref
class: stretched-link index-card-link sd-text-primary
---
Optimization Algorithms
```

List of numerical optimizers and their optional parameters.

````


````{grid-item-card}
:text-align: center
:img-top: _static/images/books.svg
:class-img-top: index-card-image
:shadow: md

```{button-link} explanations/index.html
---
click-parent:
ref-type: ref
class: stretched-link index-card-link sd-text-primary
---
Explanations
```

Background information on key topics central to the package.

````

````{grid-item-card}
:text-align: center
:img-top: _static/images/coding.svg
:class-img-top: index-card-image
:shadow: md

```{button-link} reference_guides/index.html
---
click-parent:
ref-type: ref
class: stretched-link index-card-link sd-text-primary
---
API Reference
```

Detailed description of the estimagic API.

````

````{grid-item-card}
:text-align: center
:columns: 12
:img-top: _static/images/video.svg
:class-img-top: index-card-image
:shadow: md

```{button-link} videos.html
---
click-parent:
ref-type: ref
class: stretched-link index-card-link sd-text-primary
---
Videos
```

Collection of tutorials, talks, and screencasts on estimagic.

````

`````

```{toctree}
---
hidden: true
maxdepth: 1
---
getting_started/index
how_to_guides/index
explanations/index
reference_guides/index
development/index
videos
algorithms
```

## Highlights

### Optimization

- estimagic wraps algorithms from *scipy.optimize*, *nlopt*, *pygmo* and more. See
  {ref}`list_of_algorithms`
- estimagic implements constraints efficiently via reparametrization, so you can solve
  constrained problems with any optimzer that supports bounds. See {ref}`constraints`
- The parameters of an optimization problem can be arbitrary pytrees. See {ref}`params`.
- The complete history of parameters and function evaluations can be saved in a database
  for maximum reproducibility. See [How to use logging]
- Painless and efficient multistart optimization. See [How to do multistart]
- The progress of the optimization is displayed in real time via an interactive
  dashboard. See {ref}`dashboard`.

### Estimation and Inference

- You can estimate a model using method of simulated moments (MSM), calculate standard
  errors and do sensitivity analysis with just one function call. See [MSM Tutorial]
- Asymptotic standard errors for maximum likelihood estimation.
- estimagic also provides bootstrap confidence intervals and standard errors. Of course
  the bootstrap procedures are parallelized.

### Numerical differentiation

- estimagic can calculate precise numerical derivatives using
  [Richardson extrapolations](https://en.wikipedia.org/wiki/Richardson_extrapolation).
- Function evaluations needed for numerical derivatives can be done in parallel with
  pre-implemented or user provided batch evaluators.

**Useful links for search:** {ref}`genindex` | {ref}`modindex` | {ref}`search`

[how to do multistart]: how_to_guides/optimization/how_to_do_multistart_optimizations
[how to use logging]: how_to_guides/optimization/how_to_use_logging
[msm tutorial]: getting_started/estimation/first_msm_estimation_with_estimagic
