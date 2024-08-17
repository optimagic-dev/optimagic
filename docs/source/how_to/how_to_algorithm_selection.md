# Which optimizer to use

This is a short and a simplified guide on selecting an optimization algorithm based on the properties of your problem.

Knowledge of those properties can significantly narrow the set of algorithms best suited for your problem. 

### Choosing a local optimizer
```mermaid
graph LR
    classDef highlight fill:#FF4500;
    A["Do you have<br/>nonlinear constraints?"] -- yes --> B["differentiable?"]
    B["differentiable?"] -- yes --> C["'ipopt', 'nlopt_slsqp', 'scipy_trust_constr'"]
    B["differentiable?"] -- no --> D["'scipy_cobyla', 'nlopt_cobyla'"]

    A["Do you have<br/>nonlinear constraints?"] -- no --> E["Can you exploit<br/>a least-squares<br/>structure?"]
    E["Can you exploit<br/>a least-squares<br/>structure?"] -- yes --> F["differentiable?"]
    E["Can you exploit<br/>a least-squares<br/>structure?"] -- no --> G["differentiable?"]

    F["differentiable?"] -- yes --> H["'scipy_ls_lm', 'scipy_ls_trf', 'scipy_ls_dogleg'"]
    F["differentiable?"] -- no --> I["'nag_dflos', 'pounders', 'tao_pounders'"]

    G["differentiable?"] -- yes --> J["'scipy_lbfgsb', 'fides'"]
    G["differentiable?"] -- no --> K["'nlopt_bobyqa', 'nlopt_neldermead', 'neldermead_parallel'"]
```

Almost always, you will have more than one algorithm to try out choose the best by comparing them via the `criterion_plot` [insert link]. 

Remember, no amount of theory can replace experimentation!

```{eval-rst}
.. tabbed:: Differentiable Scalar Function
    As an example of unconstrained optimization problem with a scalar differentiable objective function, consider the minimization of the sphere function: 
    .. code-block:: python
        import numpy as np
        
        def sphere(params):
            return params@params
        
        def sphere_gradient(params):
            return params*2

        start_params = np.arange(5)

    For this problem, the algorithm choice lies between ``'scipy_lbfgs'`` and ``'fides'``. 